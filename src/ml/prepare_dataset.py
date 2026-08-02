"""Offline Training Dataset Builder.

Assembles unified ML Training Dataset by joining historical batch features with raw labels (is_fraud).
"""

from pathlib import Path
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger
from src.ml.ensemble import derive_transaction_features

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "trans_count_7d",
    "trans_count_30d",
    "avg_amount_30d",
    "max_amount_30d",
    "distinct_addr_7d",
    "days_since_last_trans",
    "TransactionAmt",
    "amount_ratio_30d",
    "is_amount_gt_30d_max",
]

TARGET_COLUMN = "is_fraud"


def prepare_training_dataset(
    csv_path: str | Path = settings.raw_csv_path,
    parquet_path: str | Path = settings.batch_parquet_path,
    output_path: str | Path = settings.ml_dataset_path,
    sample_size: int = 50000,
) -> pd.DataFrame:
    """Joins historical batch features with raw transaction labels to produce the standardized model training dataset."""
    path_csv = Path(csv_path)
    logger.info(f"Loading raw transactions from '{path_csv}' (Sampling max {sample_size:,} rows)...")
    if not path_csv.is_file():
        raise FileNotFoundError(f"Raw transaction CSV not found at: {path_csv}")

    df_raw = pd.read_csv(
        path_csv,
        usecols=["TransactionID", "isFraud", "TransactionAmt", "card1"],
        nrows=sample_size,
    )
    df_raw = df_raw.rename(columns={"isFraud": "is_fraud", "card1": "card_id"})
    df_raw["card_id"] = df_raw["card_id"].astype(str)

    path_parquet = Path(parquet_path)
    logger.info(f"Loading batch features from '{path_parquet}'...")
    if not path_parquet.is_file():
        raise FileNotFoundError(f"Batch features parquet not found at: {path_parquet}")

    df_batch = pd.read_parquet(path_parquet)
    df_batch["card_id"] = df_batch["card_id"].astype(str)

    logger.info("Performing Feature Join (Batch Features + Target Labels)...")
    df_joined = pd.merge(df_raw, df_batch, on="card_id", how="inner")

    df_joined["amount_ratio_30d"], df_joined["is_amount_gt_30d_max"] = derive_transaction_features(
        df_joined["TransactionAmt"], df_joined["avg_amount_30d"], df_joined["max_amount_30d"]
    )

    for col in FEATURE_COLUMNS:
        if col in df_joined.columns:
            df_joined[col] = df_joined[col].fillna(0.0)

    final_cols = ["card_id", "TransactionID", TARGET_COLUMN] + FEATURE_COLUMNS
    df_final = df_joined[final_cols].drop_duplicates(subset=["TransactionID"])

    path_output = Path(output_path)
    path_output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(path_output, index=False)

    fraud_count = int(df_final[TARGET_COLUMN].sum())
    fraud_ratio = (fraud_count / len(df_final)) * 100 if len(df_final) > 0 else 0.0
    logger.info(f"Training Dataset saved to '{path_output}'")
    logger.info(f"   Total Samples: {len(df_final):,} rows")
    logger.info(f"   Fraud Labels: {fraud_count:,} ({fraud_ratio:.2f}%)")
    logger.info(f"   Feature Columns: {FEATURE_COLUMNS}")

    return df_final


if __name__ == "__main__":
    logger.info("Preparing ML training dataset...")
    prepare_training_dataset()
