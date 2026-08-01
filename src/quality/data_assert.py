"""Data Quality Assertion Engine (Vectorized Pandas).

Validates DataFrame structure, value bounds, and non-null constraints before committing to Lakehouse.
Quarantines invalid rows into Batch DLQ Parquet files.
"""

from pathlib import Path
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_VALID_TRANSACTION_AMOUNT: float = 500_000.0
INVALID_CARD_SENTINEL: str = "unknown_card"
MAX_ALLOWED_NULL_RATIO: float = 0.001

BATCH_DLQ_PATH = Path(settings.dlq_dir) / "batch_errors.parquet"


def validate_batch_dataframe(
    df: pd.DataFrame,
    dlq_path: Path | str | None = None,
) -> tuple[bool, pd.DataFrame, pd.DataFrame | None]:
    """Validates batch feature DataFrame using vectorized pandas assertions and splits valid vs quarantined records."""
    if df.empty:
        logger.error("[DQ Gate] DataFrame is EMPTY! Data Quality assertion failed.")
        return False, df, None

    null_ratio = float(df.isnull().mean().max())
    if null_ratio > MAX_ALLOWED_NULL_RATIO:
        logger.warning(f"[DQ Gate Warning] Max null ratio across columns is {null_ratio:.4%}")

    required_cols = [
        "card_id", "trans_count_7d", "trans_count_30d",
        "avg_amount_30d", "max_amount_30d", "distinct_addr_7d", "days_since_last_trans"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"[DQ Gate FAILED] Missing required columns: {missing}")
        return False, df.iloc[0:0].copy(), df.copy()

    card_str = df["card_id"].astype(str).str.strip()
    num = df[required_cols[1:]].apply(pd.to_numeric, errors="coerce")

    valid_mask = (
        df["card_id"].notna()
        & (card_str != "")
        & (card_str != INVALID_CARD_SENTINEL)
        & (num["trans_count_7d"] >= 0)
        & (num["trans_count_30d"] >= 0)
        & (num["avg_amount_30d"] > 0.0)
        & (num["avg_amount_30d"] <= MAX_VALID_TRANSACTION_AMOUNT)
        & (num["max_amount_30d"] > 0.0)
        & (num["max_amount_30d"] <= MAX_VALID_TRANSACTION_AMOUNT)
        & (num["distinct_addr_7d"] >= 0)
        & (num["days_since_last_trans"] >= 0.0)
    ).fillna(False)

    clean_df = df.loc[valid_mask].copy()
    error_df = df.loc[~valid_mask].copy()

    if error_df.empty:
        logger.info(f"[DQ Gate PASSED] All {len(clean_df):,} records satisfy data quality assertions.")
        return True, clean_df, None

    logger.warning(f"[DQ Gate FAILED] Quarantining {len(error_df):,} invalid records ({len(clean_df):,} clean).")
    target_dlq = Path(dlq_path) if dlq_path else BATCH_DLQ_PATH
    target_dlq.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_parquet(target_dlq, index=False)
    logger.warning(f"[DLQ Isolation] Quarantined {len(error_df)} invalid records to '{target_dlq}'")

    return False, clean_df, error_df

