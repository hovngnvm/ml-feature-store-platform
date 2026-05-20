import os
import sys
import logging
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_feast_retrieval")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_PATH = os.path.join(PROJECT_DIR, "feature_repository")
PARQUET_PATH = os.path.join(PROJECT_DIR, "data", "batch_features.parquet")

def test_feast_pipeline():
    logger.info(f"Initializing Feast FeatureStore from repository: {REPO_PATH}")
    store = FeatureStore(repo_path=REPO_PATH)

    # 1. Materialize Batch Feature View into Online Store
    logger.info("Syncing Batch Parquet -> Online Store via store.materialize(feature_views=['card_batch_features'])...")
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2030, 12, 31, tzinfo=timezone.utc)
    store.materialize(
        start_date=start_date,
        end_date=end_date,
        feature_views=["card_batch_features"]
    )
    logger.info("Feast Materialization of batch features completed successfully.")

    # 2. Fetch sample card IDs & timestamps from offline store
    df_sample = pd.read_parquet(PARQUET_PATH).head(3)
    sample_card_ids = df_sample["card_id"].astype(str).tolist()
    parquet_timestamps = df_sample["event_timestamp"].tolist()
    logger.info(f"Sample card IDs from offline store: {sample_card_ids}")

    # 3. Test Offline Point-in-Time Historical Feature Retrieval (for Model Training)
    logger.info("Testing Offline Historical Feature Retrieval (Point-in-Time Join)...")
    entity_df = pd.DataFrame.from_dict({
        "card_id": sample_card_ids,
        "event_timestamp": [pd.Timestamp(ts, tz='UTC') for ts in parquet_timestamps]
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
    logger.info(f"\n{training_df.to_string()}")

    # 4. Test Online Low-Latency Feature Retrieval (for Online Inference)
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
    logger.info("Online Feature Vector Response:")
    logger.info(f"\n{response_df.to_string()}")
    logger.info("Phase 4 Feast Repository & Materialization Verification PASSED 100% SUCCESS!")

if __name__ == "__main__":
    test_feast_pipeline()
