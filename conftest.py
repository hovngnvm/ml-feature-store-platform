import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_redis():
    """Provides a mocked Redis client instance."""
    client = MagicMock()
    client.ping.return_value = True
    return client


@pytest.fixture
def mock_minio():
    """Provides a mocked MinIO client instance."""
    client = MagicMock()
    client.bucket_exists.return_value = True
    return client
