"""
Centralized Redis Connection Pool & Client Utility Module.
"""

import redis
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("redis_client")

_redis_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    """Returns a thread-safe Redis client instance backed by a shared connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return redis.Redis(connection_pool=_redis_pool)


def check_redis_health() -> bool:
    """Probes Redis connection health."""
    try:
        client = get_redis_client()
        return bool(client.ping())
    except Exception as e:
        logger.warning(f"Redis healthcheck probe failed: {e}")
        return False
