"""
Redis Client Infrastructure Wrapper.

Manages thread-safe Redis connection pooling, health checks, 
and batch online feature storage operations for Feast.
"""

import os
import sys
import redis
from typing import Dict, Any, Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("redis_client")


class RedisClient:
    """Thread-safe Redis client wrapper with connection pooling."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None
    ) -> None:
        """Initializes Redis connection pool from settings or parameters."""
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db if db is not None else settings.redis_db
        self.password = password or settings.redis_password

        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            socket_timeout=settings.redis_socket_timeout,
            decode_responses=True
        )
        self._client: Optional[redis.Redis] = None

    def get_client(self) -> redis.Redis:
        """Returns active Redis client instance from connection pool."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
        return self._client

    def ping(self) -> bool:
        """Checks connection status to Redis server.

        Returns:
            True if Redis server responds to PING, False otherwise.
        """
        try:
            client = self.get_client()
            return bool(client.ping())
        except Exception as e:
            logger.error(f"Redis health check ping failed: {e}")
            return False

    def hset_features(self, key: str, mapping: Dict[str, Any]) -> int:
        """Stores feature key-value pairs into Redis hash.

        Args:
            key: Redis hash key (e.g., 'card_id:12345').
            mapping: Dictionary of feature names and values.

        Returns:
            Number of fields added to the hash.
        """
        try:
            client = self.get_client()
            str_mapping = {k: str(v) for k, v in mapping.items()}
            return int(client.hset(key, mapping=str_mapping))
        except Exception as e:
            logger.error(f"Failed to execute HSET for key '{key}': {e}")
            raise

    def close(self) -> None:
        """Closes active Redis connection pool."""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connection pool disconnected.")
