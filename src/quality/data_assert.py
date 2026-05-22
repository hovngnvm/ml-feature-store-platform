import os
import sys
import logging
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from pandera.errors import SchemaErrors, SchemaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_assert")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DLQ_DIR = os.path.join(PROJECT_DIR, "data", "lakehouse", "dlq")
BATCH_DLQ_PATH = os.path.join(DLQ_DIR, "batch_errors.parquet")

# -----------------------------------------------------------------------------
# 1. Pandera Schemas
# -----------------------------------------------------------------------------
BatchFeatureSchema = DataFrameSchema(
    columns={
        "card_id": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0, name="non_empty_card_id"),
                Check(lambda s: s != "unknown_card", name="valid_card_id")
            ]
        ),
        "trans_count_7d": Column(int, nullable=False, checks=Check.ge(0)),
        "trans_count_30d": Column(int, nullable=False, checks=Check.ge(0)),
        "avg_amount_30d": Column(
            float,
            nullable=False,
            checks=[Check.gt(0.0), Check.le(500000.0)]
        ),
        "max_amount_30d": Column(
            float,
            nullable=False,
            checks=[Check.gt(0.0), Check.le(500000.0)]
        ),
        "distinct_addr_7d": Column(int, nullable=False, checks=Check.ge(0)),
        "days_since_last_trans": Column(float, nullable=False, checks=Check.ge(0.0)),
    },
    coerce=True,
    strict=False
)

StreamEventSchema = DataFrameSchema(
    columns={
        "transaction_id": Column(int, nullable=False, checks=Check.gt(0)),
        "card_id": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0, name="non_empty_card_id"),
                Check(lambda s: s != "unknown_card", name="valid_card_id")
            ]
        ),
        "amount": Column(
            float,
            nullable=False,
            checks=[Check.gt(0.0), Check.le(500000.0)]
        ),
    },
    coerce=True,
    strict=False
)

# -----------------------------------------------------------------------------
# 2. Validation Engine Functions
# -----------------------------------------------------------------------------
def validate_batch_dataframe(df: pd.DataFrame):
    """
    Validates batch feature DataFrame using Pandera BatchFeatureSchema.
    Uses lazy=True to evaluate all rules and capture bad rows.
    
    Returns:
        (is_valid: bool, clean_df: pd.DataFrame, error_df: pd.DataFrame or None)
    """
    if df.empty:
        logger.error("[DQ Gate] DataFrame is EMPTY! Data Quality assertion failed.")
        return False, df, None

    # Check null ratio
    null_ratio = df.isnull().mean().max()
    if null_ratio > 0.001:  # > 0.1% null
        logger.warning(f"[DQ Gate Warning] Max null ratio across columns is {null_ratio:.4%}")

    try:
        validated_df = BatchFeatureSchema.validate(df, lazy=True)
        logger.info(f"[DQ Gate PASSED] All {len(validated_df):,} records satisfy BatchFeatureSchema assertions.")
        return True, validated_df, None
    except SchemaErrors as err:
        logger.error(f"[DQ Gate FAILED] Found schema violations in {len(err.failure_cases)} instances.")
        
        # Extract bad row indices
        failure_cases = err.failure_cases
        invalid_indices = failure_cases["index"].dropna().astype(int).unique()
        
        error_df = df.iloc[invalid_indices].copy()
        clean_df = df.drop(index=invalid_indices).copy()
        
        # Quarantine invalid rows into Batch DLQ Parquet
        os.makedirs(DLQ_DIR, exist_ok=True)
        error_df.to_parquet(BATCH_DLQ_PATH, index=False)
        logger.warning(f"[DLQ Isolation] Quarantined {len(error_df)} invalid records to '{BATCH_DLQ_PATH}'")
        
        return False, clean_df, error_df

if __name__ == "__main__":
    logger.info("Running Data Quality Assertion Engine Self-Test...")
    
    # Valid Test Sample
    sample_valid = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": 5,
        "trans_count_30d": 20,
        "avg_amount_30d": 150.50,
        "max_amount_30d": 500.00,
        "distinct_addr_7d": 2,
        "days_since_last_trans": 1.2
    }])
    
    is_valid, clean, errs = validate_batch_dataframe(sample_valid)
    assert is_valid, "Valid sample failed validation!"
    
    # Corrupt Test Sample
    sample_corrupt = pd.DataFrame([
        {
            "card_id": "11556",
            "trans_count_7d": 5,
            "trans_count_30d": 20,
            "avg_amount_30d": 150.50,
            "max_amount_30d": 500.00,
            "distinct_addr_7d": 2,
            "days_since_last_trans": 1.2
        },
        {
            "card_id": "unknown_card",  # Invalid card_id
            "trans_count_7d": -1,       # Invalid count < 0
            "trans_count_30d": 10,
            "avg_amount_30d": -99.0,    # Invalid amount < 0
            "max_amount_30d": 50.0,
            "distinct_addr_7d": 1,
            "days_since_last_trans": 0.0
        }
    ])
    
    is_valid_c, clean_c, errs_c = validate_batch_dataframe(sample_corrupt)
    assert not is_valid_c, "Corrupt sample incorrectly passed validation!"
    assert len(clean_c) == 1, "Expected 1 clean row"
    assert len(errs_c) == 1, "Expected 1 corrupt row in DLQ"
    logger.info("Self-test completed successfully! All Pandera assertion gates verified.")
