"""Centralized Redis Connection Pool & Client Utility Module."""

import redis
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_redis_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    """Returns a thread-safe Redis client instance backed by a shared connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            max_connections=50,
            decode_responses=True,
        )
    return redis.Redis(connection_pool=_redis_pool)


def reset_redis_pool() -> None:
    """Closes and resets the connection pool for testing and lifecycle management."""
    global _redis_pool
    if _redis_pool is not None:
        _redis_pool.disconnect()
        _redis_pool = None


def check_redis_health(client: redis.Redis | None = None) -> bool:
    """Probes Redis connection health."""
    try:
        r = client or get_redis_client()
        return bool(r.ping())
    except (redis.RedisError, ConnectionError, OSError, Exception) as e:
        logger.warning(f"Redis healthcheck probe failed: {e}")
        return False
