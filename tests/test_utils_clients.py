"""Automated Test Suite for MinIO S3 and Redis Shared Utilities."""

from unittest.mock import MagicMock
from src.utils.minio_client import ensure_bucket_exists
from src.utils.redis_client import check_redis_health


def test_minio_ensure_bucket_exists_idempotent() -> None:
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False

    ensure_bucket_exists(mock_client, "test-bucket")
    mock_client.make_bucket.assert_called_once_with("test-bucket")


def test_minio_ensure_bucket_exists_when_already_exists() -> None:
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True

    ensure_bucket_exists(mock_client, "existing-bucket")
    mock_client.make_bucket.assert_not_called()


def test_redis_client_healthcheck_fallback() -> None:
    mock_redis = MagicMock()
    mock_redis.ping.side_effect = Exception("Connection refused")

    is_healthy = check_redis_health(mock_redis)
    assert is_healthy is False
