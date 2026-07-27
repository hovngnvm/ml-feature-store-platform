"""
System Settings & Centralized Configuration Module.

Provides single-source-of-truth for project paths, environment variables,
and infrastructure connection parameters across all data pipelines.
Uses Pydantic Settings with Pathlib Project Root Resolution and @lru_cache Singleton.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class SystemSettings(BaseSettings):
    """Centralized Settings Instance container."""

    # Base System Paths
    project_dir: str = str(PROJECT_DIR)
    data_dir: str = str(PROJECT_DIR / "data")
    model_dir: str = str(PROJECT_DIR / "models")
    feature_repo_dir: str = str(PROJECT_DIR / "feature_repository")
    dashboard_dir: str = str(PROJECT_DIR / "dashboards")

    # Lakehouse & Data Artifact Paths
    lakehouse_base_dir: str = str(PROJECT_DIR / "data" / "lakehouse" / "batch_features")
    dlq_dir: str = str(PROJECT_DIR / "data" / "lakehouse" / "dlq")
    raw_csv_path: str = Field(default=str(PROJECT_DIR / "data" / "train_transaction.csv"), alias="DATASET_PATH")
    batch_parquet_path: str = str(PROJECT_DIR / "data" / "batch_features.parquet")
    ml_dataset_path: str = str(PROJECT_DIR / "data" / "ml_training_dataset.parquet")
    raw_events_parquet_path: str = str(PROJECT_DIR / "data" / "raw_events_stream.parquet")
    stream_dlq_parquet_path: str = str(PROJECT_DIR / "data" / "lakehouse" / "dlq" / "stream_errors.parquet")

    # Machine Learning Artifact Paths
    model_artifact_path: str = str(PROJECT_DIR / "models" / "ensemble_fraud_model.joblib")
    report_json_path: str = str(PROJECT_DIR / "models" / "evaluation_report.json")

    # Infrastructure Service Configurations
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    kafka_broker: str = Field(default="localhost:19092", alias="KAFKA_BROKER")
    kafka_topic: str = Field(default="raw_transactions", alias="KAFKA_TOPIC")
    kafka_dlq_topic: str = Field(default="raw_transactions_dlq", alias="KAFKA_DLQ_TOPIC")
    stream_engine: str = Field(default="direct", alias="STREAM_ENGINE")

    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadminpassword", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="feature-store-offline", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # Application Ports & Service Endpoints
    api_host: str = Field(default="0.0.0.0", alias="FASTAPI_HOST")
    api_port: int = Field(default=8000, alias="FASTAPI_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    metrics_port: int = Field(default=9091, alias="METRICS_PORT")

    # Fraud Decision & Cost Matrix Parameters
    default_fp_cost: float = Field(default=2.0, alias="DEFAULT_FP_COST")
    default_decision_threshold: float = Field(default=0.5, alias="DEFAULT_DECISION_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("minio_endpoint", mode="after")
    @classmethod
    def clean_minio_endpoint(cls, v: str) -> str:
        return v.replace("http://", "").replace("https://", "")

    @field_validator("log_level", mode="after")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("stream_engine", mode="after")
    @classmethod
    def normalize_stream_engine(cls, v: str) -> str:
        return v.lower()


@lru_cache()
def get_settings() -> SystemSettings:
    """Returns singleton SystemSettings instance cached in memory."""
    return SystemSettings()


settings = get_settings()

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

