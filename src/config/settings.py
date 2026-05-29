"""
Centralized Configuration Settings Module.

Loads and validates environment variables from .env using Pydantic BaseSettings.
Provides type-safe configuration parameters for Kafka, Redis, MinIO, and App logging.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    # Project directory paths
    project_dir: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Logging settings
    log_level: str = "INFO"

    # Kafka / Redpanda Stream Broker Configuration
    kafka_broker: str = "localhost:19092"
    kafka_topic: str = "raw_transactions"
    kafka_dlq_topic: str = "raw_transactions_dlq"
    kafka_consumer_group: str = "pyflink_feature_table_group"

    # Redis Online Feature Store Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_socket_timeout: float = 5.0

    # MinIO S3 Offline Feature Store Configuration
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadminpassword"
    minio_bucket: str = "feature-store-offline"
    minio_secure: bool = False

    # Application Ports
    fastapi_port: int = 8000
    streamlit_port: int = 8501
    prometheus_metrics_port: int = 9091

    # Default Data Paths
    dataset_path: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    def get_dataset_path(self) -> str:
        """Returns default dataset path if not explicitly provided in environment."""
        if self.dataset_path:
            return self.dataset_path
        return os.path.join(self.project_dir, "data", "train_transaction.csv")


# Global singleton settings instance
settings = Settings()
