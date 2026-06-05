"""
ML Training Dataset Preparation Engine.

Assembles unified training dataset by joining historical offline batch features 
with raw labels (is_fraud).
"""

import os
from typing import Optional, List
import pandas as pd

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("prepare_dataset")

RAW_CSV_PATH = settings.get_dataset_path()
BATCH_PARQUET_PATH = os.path.join(settings.project_dir, "data", "batch_features.parquet")
OUTPUT_DATASET_PATH = os.path.join(settings.project_dir, "data", "ml_training_dataset.parquet")

FEATURE_COLUMNS: List[str] = [
    "trans_count_7d",
    "trans_count_30d",
    "avg_amount_30d",
    "max_amount_30d",
    "distinct_addr_7d",
    "days_since_last_trans",
    "TransactionAmt",
    "amount_ratio_30d",
    "is_amount_gt_30d_max"
]

TARGET_COLUMN: str = "is_fraud"


def prepare_training_dataset(
    csv_path: str = RAW_CSV_PATH,
    parquet_path: str = BATCH_PARQUET_PATH,
    output_path: str = OUTPUT_DATASET_PATH,
    sample_size: int = 50000
) -> pd.DataFrame:
    """Assembles unified ML training dataset by joining batch features with target labels.

    Args:
        csv_path: Absolute or relative file path to raw transactions CSV.
        parquet_path: Absolute or relative file path to batch features Parquet.
        output_path: Target destination path for joined dataset.
        sample_size: Maximum row count sample from raw dataset.

    Returns:
        Joined training dataset pandas DataFrame.
    """
    logger.info(f"Loading raw transactions from '{csv_path}' (Sampling max {sample_size:,} rows)...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw transaction CSV not found at: {csv_path}")

    df_raw = pd.read_csv(
        csv_path,
        usecols=["TransactionID", "isFraud", "TransactionAmt", "card1"],
        nrows=sample_size
    )
    df_raw = df_raw.rename(columns={"isFraud": "is_fraud", "card1": "card_id"})
    df_raw["card_id"] = df_raw["card_id"].astype(str)

    logger.info(f"Loading batch features from '{parquet_path}'...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Batch features parquet not found at: {parquet_path}")

    df_batch = pd.read_parquet(parquet_path)
    df_batch["card_id"] = df_batch["card_id"].astype(str)

    logger.info("Performing feature join between batch features and target labels...")
    df_joined = pd.merge(df_raw, df_batch, on="card_id", how="inner")

    df_joined["amount_ratio_30d"] = df_joined["TransactionAmt"] / (df_joined["avg_amount_30d"] + 1.0)
    df_joined["is_amount_gt_30d_max"] = (
        df_joined["TransactionAmt"] > df_joined["max_amount_30d"]
    ).astype(float)

    for col in FEATURE_COLUMNS:
        if col in df_joined.columns:
            df_joined[col] = df_joined[col].fillna(0.0)

    final_cols = ["card_id", "TransactionID", TARGET_COLUMN] + FEATURE_COLUMNS
    df_final = df_joined[final_cols].drop_duplicates(subset=["TransactionID"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    fraud_count = int(df_final[TARGET_COLUMN].sum())
    fraud_ratio = (fraud_count / len(df_final)) * 100 if len(df_final) > 0 else 0.0
    logger.info(f"Training dataset saved to '{output_path}'")
    logger.info(f"Total Samples: {len(df_final):,} rows")
    logger.info(f"Fraud Labels: {fraud_count:,} ({fraud_ratio:.2f}%)")

    return df_final


if __name__ == "__main__":
    logger.info("Executing Phase 1: Feature Dataset Preparation...")
    prepare_training_dataset()
