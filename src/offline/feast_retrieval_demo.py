"""Feast Historical & Online Feature Retrieval Demo Validator Script.

Verifies point-in-time offline feature retrieval and online feature vector lookups.
"""

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_feast_retrieval_demo() -> None:
    """Executes Feast FeatureStore validation pipeline."""
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
    batch_parquet = Path(settings.batch_parquet_path)
    if not batch_parquet.exists():
        logger.warning(f"Batch Parquet file not found at '{batch_parquet}'. Run batch feature job first.")
        return

    df_sample = pd.read_parquet(batch_parquet).head(3)
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
    assert not training_df.empty, "Historical training DataFrame must not be empty"

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

    assert "card_id" in response, "Response must contain 'card_id' field"
    logger.info("Online Feature Vector Response retrieved successfully.")
    logger.info("Feast Repository & Materialization Verification PASSED.")


if __name__ == "__main__":
    run_feast_retrieval_demo()
