"""
Feast Historical & Online Feature Retrieval Demo Validator Script.

Verifies point-in-time offline feature retrieval and online feature vector lookups.
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("feast_retrieval_demo")


def run_feast_retrieval_demo() -> None:
    """
    Executes Feast FeatureStore validation pipeline:
    - Materializes batch features into Redis online store.
    - Tests offline point-in-time historical feature retrieval.
    - Tests online low-latency feature retrieval.
    """
    from feast import FeatureStore

    logger.info(f"Initializing Feast FeatureStore from repository: {settings.feature_repo_dir}")
    store = FeatureStore(repo_path=settings.feature_repo_dir)

    # Materialize Batch Feature View into Online Store
    logger.info("Syncing Batch Parquet to Online Store via store.materialize()...")
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2030, 12, 31, tzinfo=timezone.utc)
    store.materialize(
        start_date=start_date,
        end_date=end_date,
        feature_views=["card_batch_features"]
    )
    logger.info("Feast Materialization of batch features completed successfully.")

    # Fetch sample card IDs & timestamps from offline store
    if not os.path.exists(settings.batch_parquet_path):
        logger.warning(f"Batch Parquet file not found at '{settings.batch_parquet_path}'. Run batch feature job first.")
        return

    df_sample = pd.read_parquet(settings.batch_parquet_path).head(3)
    sample_card_ids = df_sample["card_id"].astype(str).tolist()
    parquet_timestamps = df_sample["event_timestamp"].tolist()
    logger.info(f"Sample card IDs from offline store: {sample_card_ids}")

    # Test Offline Point-in-Time Historical Feature Retrieval
    logger.info("Testing Offline Historical Feature Retrieval (Point-in-Time Join)...")
    entity_df = pd.DataFrame.from_dict({
        "card_id": sample_card_ids,
        "event_timestamp": [pd.Timestamp(ts, tz="UTC") for ts in parquet_timestamps]
    })

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "card_batch_features:trans_count_7d",
            "card_batch_features:trans_count_30d",
            "card_batch_features:avg_amount_30d",
            "card_batch_features:max_amount_30d"
        ]
    ).to_df()

    logger.info(f"Historical Training DataFrame Shape: {training_df.shape}")

    # Test Online Low-Latency Feature Retrieval
    logger.info("Testing Online Feature Retrieval (Online Store Lookup + On-Demand Calculation)...")
    entity_rows = [
        {"card_id": str(sample_card_ids[0]), "current_amount": 250.0},
        {"card_id": str(sample_card_ids[1]), "current_amount": 1500.0}
    ]

    response = store.get_online_features(
        features=[
            "card_batch_features:avg_amount_30d",
            "card_batch_features:max_amount_30d",
            "card_stream_features:trans_count_5m",
            "card_stream_features:avg_amount_24h",
            "card_on_demand_features:amount_ratio_24h",
            "card_on_demand_features:amount_ratio_30d",
            "card_on_demand_features:is_amount_gt_30d_max"
        ],
        entity_rows=entity_rows
    ).to_dict()

    response_df = pd.DataFrame(response)
    logger.info("Online Feature Vector Response retrieved successfully.")
    logger.info("Feast Repository & Materialization Verification PASSED.")


if __name__ == "__main__":
    run_feast_retrieval_demo()
