"""Data Quality Assertion Engine (Pandera Schemas).

Validates DataFrame structure, value bounds, and non-null constraints before committing to Lakehouse.
Quarantines invalid rows into Batch DLQ Parquet files.
"""

from pathlib import Path
import pandas as pd
from pandera.pandas import Column, Check, DataFrameSchema
from pandera.errors import SchemaErrors

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_VALID_TRANSACTION_AMOUNT: float = 500_000.0
INVALID_CARD_SENTINEL: str = "unknown_card"
MAX_ALLOWED_NULL_RATIO: float = 0.001

BATCH_DLQ_PATH = Path(settings.dlq_dir) / "batch_errors.parquet"

BatchFeatureSchema = DataFrameSchema(
    columns={
        "card_id": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0, name="non_empty_card_id"),
                Check(lambda s: s != INVALID_CARD_SENTINEL, name="valid_card_id"),
            ]
        ),
        "trans_count_7d": Column(int, nullable=False, checks=Check.ge(0)),
        "trans_count_30d": Column(int, nullable=False, checks=Check.ge(0)),
        "avg_amount_30d": Column(
            float,
            nullable=False,
            checks=[Check.gt(0.0), Check.le(MAX_VALID_TRANSACTION_AMOUNT)]
        ),
        "max_amount_30d": Column(
            float,
            nullable=False,
            checks=[Check.gt(0.0), Check.le(MAX_VALID_TRANSACTION_AMOUNT)]
        ),
        "distinct_addr_7d": Column(int, nullable=False, checks=Check.ge(0)),
        "days_since_last_trans": Column(float, nullable=False, checks=Check.ge(0.0)),
    },
    coerce=True,
    strict=False
)


def validate_batch_dataframe(
    df: pd.DataFrame,
    dlq_path: Path | str | None = None,
) -> tuple[bool, pd.DataFrame, pd.DataFrame | None]:
    """Validates batch feature DataFrame using Pandera schema and splits valid vs quarantined records."""
    if df.empty:
        logger.error("[DQ Gate] DataFrame is EMPTY! Data Quality assertion failed.")
        return False, df, None

    null_ratio = float(df.isnull().mean().max())
    if null_ratio > MAX_ALLOWED_NULL_RATIO:
        logger.warning(f"[DQ Gate Warning] Max null ratio across columns is {null_ratio:.4%}")

    try:
        validated_df = BatchFeatureSchema.validate(df, lazy=True)
        logger.info(f"[DQ Gate PASSED] All {len(validated_df):,} records satisfy BatchFeatureSchema assertions.")
        return True, validated_df, None
    except SchemaErrors as err:
        logger.error(f"[DQ Gate FAILED] Found schema violations in {len(err.failure_cases)} instances.")

        failure_cases = err.failure_cases
        invalid_indices = failure_cases["index"].dropna().unique()

        error_df = df.loc[df.index.isin(invalid_indices)].copy()
        clean_df = df.loc[~df.index.isin(invalid_indices)].copy()

        target_dlq = Path(dlq_path) if dlq_path else BATCH_DLQ_PATH
        target_dlq.parent.mkdir(parents=True, exist_ok=True)
        if not error_df.empty:
            error_df.to_parquet(target_dlq, index=False)
            logger.warning(f"[DLQ Isolation] Quarantined {len(error_df)} invalid records to '{target_dlq}'")

        return False, clean_df, error_df


if __name__ == "__main__":
    logger.info("Running Data Quality Assertion Engine Self-Test...")

    sample_valid = pd.DataFrame([{
        "card_id": "11556",
        "trans_count_7d": 5,
        "trans_count_30d": 20,
        "avg_amount_30d": 150.50,
        "max_amount_30d": 500.00,
        "distinct_addr_7d": 2,
        "days_since_last_trans": 1.2,
    }])

    is_valid, clean, errs = validate_batch_dataframe(sample_valid)
    assert is_valid, "Valid sample failed validation!"

    sample_corrupt = pd.DataFrame([
        {
            "card_id": "11556",
            "trans_count_7d": 5,
            "trans_count_30d": 20,
            "avg_amount_30d": 150.50,
            "max_amount_30d": 500.00,
            "distinct_addr_7d": 2,
            "days_since_last_trans": 1.2,
        },
        {
            "card_id": "unknown_card",
            "trans_count_7d": -1,
            "trans_count_30d": 10,
            "avg_amount_30d": -99.0,
            "max_amount_30d": 50.0,
            "distinct_addr_7d": 1,
            "days_since_last_trans": 0.0,
        }
    ])

    is_valid_c, clean_c, errs_c = validate_batch_dataframe(sample_corrupt)
    assert not is_valid_c, "Corrupt sample incorrectly passed validation!"
    assert len(clean_c) == 1, "Expected 1 clean row"
    assert len(errs_c) == 1, "Expected 1 corrupt row in DLQ"
    logger.info("Self-test completed successfully! All Pandera assertion gates verified.")
