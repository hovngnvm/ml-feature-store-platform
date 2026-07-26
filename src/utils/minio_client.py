"""Centralized MinIO S3 Client Utility Module."""

from pathlib import Path
from minio import Minio
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_minio_client: Minio | None = None


def get_minio_client() -> Minio:
    """Returns a configured MinIO client instance."""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _minio_client


def reset_minio_client() -> None:
    """Resets cached MinIO client for testing and lifecycle management."""
    global _minio_client
    _minio_client = None


def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """Idempotently ensures the target S3 bucket exists."""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"Created MinIO bucket: {bucket_name}")


def upload_file_to_minio(
    local_path: str | Path,
    object_name: str,
    bucket_name: str = settings.minio_bucket,
) -> bool:
    """Uploads a local file to MinIO S3 storage."""
    path = Path(local_path)
    if not path.is_file():
        logger.error(f"Local file does not exist for upload: '{path}'")
        return False

    client = get_minio_client()
    ensure_bucket_exists(client, bucket_name)
    try:
        client.fput_object(bucket_name, object_name, str(path))
        logger.info(f"Uploaded '{path}' to MinIO: '{bucket_name}/{object_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to upload '{path}' to MinIO: {e}")
        return False


def upload_folder_to_minio(
    local_folder: str | Path,
    bucket_name: str = settings.minio_bucket,
    minio_prefix: str = "batch_features",
) -> int:
    """Recursively uploads a local directory to MinIO S3 storage, returning the uploaded file count."""
    folder_path = Path(local_folder)
    if not folder_path.exists():
        logger.warning(f"Local folder does not exist for MinIO upload: '{folder_path}'")
        return 0

    client = get_minio_client()
    ensure_bucket_exists(client, bucket_name)

    count = 0
    for parquet_file in folder_path.rglob("*.parquet"):
        rel_path = parquet_file.relative_to(folder_path).as_posix()
        s3_object_name = f"{minio_prefix}/{rel_path}" if minio_prefix else rel_path
        try:
            client.fput_object(bucket_name, s3_object_name, str(parquet_file))
            count += 1
        except Exception as e:
            logger.error(f"Failed to upload '{parquet_file}' to MinIO: {e}")

    logger.info(f"[Lakehouse Sync] Uploaded {count} partitioned Parquet files -> MinIO S3 's3://{bucket_name}/{minio_prefix}/'")
    return count
