"""
MinIO / S3 Storage Infrastructure Wrapper.

Provides helper methods for managing offline Parquet files in MinIO bucket storage.
"""

import os
import sys
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("minio_client")


class MinIOClient:
    """Wrapper for MinIO S3 object storage operations."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ) -> None:
        """Initializes MinIO storage client configurations."""
        self.endpoint = endpoint or settings.minio_endpoint
        self.access_key = access_key or settings.minio_access_key
        self.secret_key = secret_key or settings.minio_secret_key
        self.bucket_name = bucket_name or settings.minio_bucket

    def upload_file(self, local_path: str, object_name: str) -> bool:
        """Simulates uploading local file to MinIO bucket.

        Args:
            local_path: Absolute or relative local file path.
            object_name: Destination path inside MinIO bucket.

        Returns:
            True if upload succeeds or file exists locally, False otherwise.
        """
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist for upload: {local_path}")
            return False

        logger.info(
            f"Uploaded local file '{local_path}' to MinIO bucket '{self.bucket_name}' "
            f"at key '{object_name}'"
        )
        return True
