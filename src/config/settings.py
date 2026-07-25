"""
System Settings & Centralized Configuration Module.

Provides single-source-of-truth for project paths, environment variables,
and infrastructure connection parameters across all data pipelines.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_DIR / ".env"

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
else:
    load_dotenv()


class SystemSettings:
    """Centralized Settings Instance container."""

    def __init__(self):
        # Base System Paths
        self.project_dir = str(PROJECT_DIR)
        self.data_dir = str(PROJECT_DIR / "data")
        self.model_dir = str(PROJECT_DIR / "models")
        self.feature_repo_dir = str(PROJECT_DIR / "feature_repository")
        self.dashboard_dir = str(PROJECT_DIR / "dashboards")

        # Lakehouse & Data Artifact Paths
        self.lakehouse_base_dir = str(PROJECT_DIR / "data" / "lakehouse" / "batch_features")
        self.dlq_dir = str(PROJECT_DIR / "data" / "lakehouse" / "dlq")
        self.raw_csv_path = os.getenv("DATASET_PATH", str(PROJECT_DIR / "data" / "train_transaction.csv"))
        self.batch_parquet_path = str(PROJECT_DIR / "data" / "batch_features.parquet")
        self.ml_dataset_path = str(PROJECT_DIR / "data" / "ml_training_dataset.parquet")
        self.raw_events_parquet_path = str(PROJECT_DIR / "data" / "raw_events_stream.parquet")
        self.stream_dlq_parquet_path = str(PROJECT_DIR / "data" / "lakehouse" / "dlq" / "stream_errors.parquet")

        # Machine Learning Artifact Paths
        self.model_artifact_path = str(PROJECT_DIR / "models" / "ensemble_fraud_model.joblib")
        self.report_json_path = str(PROJECT_DIR / "models" / "evaluation_report.json")

        # Infrastructure Service Configurations
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))

        self.kafka_broker = os.getenv("KAFKA_BROKER", "localhost:19092")
        self.kafka_topic = os.getenv("KAFKA_TOPIC", "raw_transactions")
        self.kafka_dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "raw_transactions_dlq")
        self.stream_engine = os.getenv("STREAM_ENGINE", "direct").lower()

        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000").replace("http://", "").replace("https://", "")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadminpassword")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "feature-store-offline")

        # Fraud Decision & Cost Matrix Parameters
        self.default_fp_cost = float(os.getenv("DEFAULT_FP_COST", 2.0))
        self.default_decision_threshold = float(os.getenv("DEFAULT_DECISION_THRESHOLD", 0.5))

        # Auto-bootstrap runtime directory structure
        for path_obj in (
            PROJECT_DIR / "data",
            PROJECT_DIR / "models",
            PROJECT_DIR / "dashboards",
            PROJECT_DIR / "logs",
            PROJECT_DIR / "data" / "lakehouse" / "batch_features",
            PROJECT_DIR / "data" / "lakehouse" / "dlq",
        ):
            path_obj.mkdir(parents=True, exist_ok=True)


settings = SystemSettings()

