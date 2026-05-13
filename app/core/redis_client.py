"""
META-STAMP V3 Async Redis Client Module

This module provides comprehensive async Redis client implementation for the META-STAMP V3
platform, supporting caching and session management with configurable TTL values. It includes:

- Connection management with automatic retry logic (3 attempts, exponential backoff)
- Core key-value operations (get, set, delete, exists, expire, ttl)
- JSON serialization operations for complex data structures
- Hash operations for session management
- Caching decorator for service method results

Cache TTL Configurations (per Agent Action Plan section 0.8):
- Asset metadata: 5 minutes (300 seconds) - redis_cache_ttl_seconds
- Fingerprint data: 1 hour (3600 seconds) - fingerprint_cache_ttl_seconds
- Conversation context: 1 hour (3600 seconds)

Usage:
    ```python
    from app.core.redis_client import init_redis, get_redis_client, cache_result

    # Initialize on application startup
    await init_redis()

    # Get client instance
    client = get_redis_client()
    await client.set("key", "value", ttl=300)
    value = await client.get("key")

    # Use caching decorator
    @cache_result(ttl=300, key_prefix="assets")
    async def get_asset_metadata(asset_id: str):
        return await fetch_from_database(asset_id)
    ```
"""

import asyncio
import json
import logging

from collections.abc import Callable
from functools import wraps
from typing import Any

import redis.asyncio as redis

from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from app.config import Settings, get_settings


# Configure module logger
logger = logging.getLogger(__name__)


# Module-level singleton instance
class _RedisClientContainer:
    """Container for Redis client singleton to avoid global statements."""

    client: "RedisClient | None" = None


_container = _RedisClientContainer()


class RedisClient:
    """
    Async Redis client wrapper providing caching and session management operations.

    This class encapsulates the redis-py async client with additional features:
    - Connection management with retry logic
    - JSON serialization for complex objects
    - Hash operations for session data
    - TTL management for cache expiration
    - Comprehensive error handling and logging

    Attributes:
        settings: Application settings containing Redis configuration
        _client: Underlying redis async client instance
        _connected: Connection state flag

    Example:
        ```python
        settings = get_settings()
        client = RedisClient(settings)
        await client.connect()

        # Store and retrieve data
        await client.set("user:123", "data", ttl=3600)
        data = await client.get("user:123")

        # Store JSON
        await client.set_json("profile:123", {"name": "John", "age": 30})
        profile = await client.get_json("profile:123")

        await client.close()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize RedisClient with application settings.

        Args:
            settings: Optional Settings instance. If not provided, will be loaded
                     from environment using get_settings().

        The client is configured with:
        - decode_responses=True for automatic string decoding
        - Connection pool parameters for efficient connection reuse
        """
        self.settings = settings or get_settings()
        self._client: redis.Redis | None = None
        self._connected: bool = False

        logger.info(
            "RedisClient initialized with URL: %s",
            self._mask_url(self.settings.redis_url),
        )

    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of Redis URL for safe logging."""
        # Simple masking - hide password if present
        if "@" in url:
            parts = url.split("@")
            redis_scheme = "redis" + "://"
            return redis_scheme + "***@" + parts[-1]
        return url

    async def connect(self) -> bool:
        """
        Establish connection to Redis with retry logic.

        Implements exponential backoff with 3 connection attempts:
        - Attempt 1: immediate
        - Attempt 2: after 1 second
        - Attempt 3: after 2 seconds

        Returns:
            bool: True if connection successful, False otherwise.

        Raises:
            No exceptions raised - errors are logged and False is returned.
        """
        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Attempting Redis connection (attempt %d/%d)",
                    attempt,
                    max_retries,
                )

                self._client = redis.from_url(  # type: ignore[no-untyped-call]
                    self.settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                    retry_on_timeout=True,
                )

                # Test connection with ping
                await self._client.ping()  # type: ignore[misc]
                self._connected = True

                logger.info("Successfully connected to Redis")
                return True

            except RedisConnectionError as e:
                logger.warning(
                    "Redis connection failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    str(e),
                )
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                else:
                    logger.exception("Failed to connect to Redis after %d attempts", max_retries)

            except RedisError:
                logger.exception("Redis error during connection")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        self._connected = False
        return False

    async def close(self) -> None:
        """
        Close Redis connection gracefully.

        Releases connection pool resources and resets connection state.
        Safe to call multiple times.
        """
        if self._client:
            try:
                await self._client.close()
                logger.info("Redis connection closed")
            except RedisError:
                logger.exception("Error closing Redis connection")
            finally:
                self._client = None
                self._connected = False

    async def ping(self) -> bool:
        """
        Health check method to verify Redis connection is alive.

        Returns:
            bool: True if Redis responds to PING, False otherwise.
        """
        if not self._client:
            return False

        try:
            result = await self._client.ping()  # type: ignore[misc]
            return result is True or result == "PONG"
        except (RedisError, RedisConnectionError) as e:
            logger.warning("Redis ping failed: %s", str(e))
            self._connected = False
            return False

    async def is_connected(self) -> bool:
        """
        Check current connection status.

        Performs an actual ping to verify connection is still alive,
        not just checking the internal flag.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if not self._connected or not self._client:
            return False
        return await self.ping()

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def get(self, key: str) -> str | None:
        """
        Get value by key from Redis.

        Args:
            key: The key to retrieve.

        Returns:
            Optional[str]: The value if found, None if key doesn't exist or on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return None

        try:
            result: str | None = await self._client.get(key)
            return result
        except RedisError:
            logger.exception("Failed to get key '%s'", key)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Set value with optional TTL.

        Args:
            key: The key to set.
            value: The value to store (will be converted to string).
            ttl: Optional time-to-live in seconds. If None, no expiration is set.

        Returns:
            bool: True if set successfully, False on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            # Convert value to string if necessary
            str_value = str(value) if not isinstance(value, str) else value

            if ttl is not None and ttl > 0:
                await self._client.setex(key, ttl, str_value)
            else:
                await self._client.set(key, str_value)

            logger.debug("Set key '%s' with TTL=%s", key, ttl)
            return True

        except RedisError:
            logger.exception("Failed to set key '%s'", key)
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from Redis.

        Args:
            key: The key to delete.

        Returns:
            bool: True if key was deleted, False if key didn't exist or on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            result: int = await self._client.delete(key)
            deleted: bool = result > 0
            if deleted:
                logger.debug("Deleted key '%s'", key)
            return deleted
        except RedisError:
            logger.exception("Failed to delete key '%s'", key)
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: The key to check.

        Returns:
            bool: True if key exists, False otherwise or on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            result: int = await self._client.exists(key)
            return bool(result > 0)
        except RedisError:
            logger.exception("Failed to check existence of key '%s'", key)
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration on an existing key.

        Args:
            key: The key to set expiration on.
            seconds: Time-to-live in seconds.

        Returns:
            bool: True if expiration was set, False if key doesn't exist or on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            result = await self._client.expire(key, seconds)
            return bool(result)
        except RedisError:
            logger.exception("Failed to set expiration on key '%s'", key)
            return False

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL of a key.

        Args:
            key: The key to check.

        Returns:
            int: Remaining TTL in seconds, -1 if no expiration, -2 if key doesn't exist.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return -2

        try:
            result: int = await self._client.ttl(key)
            return result
        except RedisError:
            logger.exception("Failed to get TTL for key '%s'", key)
            return -2

    # =========================================================================
    # JSON Operations
    # =========================================================================

    async def get_json(self, key: str) -> Any | None:
        """
        Get and deserialize JSON value from Redis.

        Retrieves the string value and deserializes it from JSON format.
        Useful for caching complex data structures like user profiles,
        asset metadata, and conversation context.

        Args:
            key: The key to retrieve.

        Returns:
            Optional[Any]: Deserialized Python object if found, None otherwise.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return None

        try:
            value = await self._client.get(key)
            if value is None:
                return None

            return json.loads(value)
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON for key '%s'", key)
            return None
        except RedisError:
            logger.exception("Failed to get JSON key '%s'", key)
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Serialize and set JSON value in Redis.

        Serializes the Python object to JSON format before storing.
        Useful for caching complex data structures.

        Args:
            key: The key to set.
            value: The Python object to serialize and store.
            ttl: Optional time-to-live in seconds.

        Returns:
            bool: True if set successfully, False on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            json_value = json.dumps(value, default=str)

            if ttl is not None and ttl > 0:
                await self._client.setex(key, ttl, json_value)
            else:
                await self._client.set(key, json_value)

            logger.debug("Set JSON key '%s' with TTL=%s", key, ttl)
            return True

        except (TypeError, ValueError):
            logger.exception("Failed to serialize JSON for key '%s'", key)
            return False
        except RedisError:
            logger.exception("Failed to set JSON key '%s'", key)
            return False

    # =========================================================================
    # Hash Operations (for session management)
    # =========================================================================

    async def hget(self, name: str, key: str) -> str | None:
        """
        Get a field value from a Redis hash.

        Args:
            name: The hash name.
            key: The field key within the hash.

        Returns:
            Optional[str]: Field value if found, None otherwise.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return None

        try:
            result: str | None = await self._client.hget(name, key)  # type: ignore[misc]
            return result
        except RedisError:
            logger.exception("Failed to hget '%s.%s'", name, key)
            return None

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Set a field value in a Redis hash.

        Args:
            name: The hash name.
            key: The field key within the hash.
            value: The value to store (will be converted to string).

        Returns:
            bool: True if set successfully, False on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return False

        try:
            str_value = str(value) if not isinstance(value, str) else value
            await self._client.hset(name, key, str_value)  # type: ignore[misc]
            logger.debug("Set hash field '%s.%s'", name, key)
            return True
        except RedisError:
            logger.exception("Failed to hset '%s.%s'", name, key)
            return False

    async def hgetall(self, name: str) -> dict[str, str]:
        """
        Get all fields and values from a Redis hash.

        Useful for retrieving complete session data or user preferences.

        Args:
            name: The hash name.

        Returns:
            Dict[str, str]: Dictionary of all field-value pairs, empty dict on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return {}

        try:
            result: dict[str, str] = await self._client.hgetall(name)  # type: ignore[misc]
            return result or {}
        except RedisError:
            logger.exception("Failed to hgetall '%s'", name)
            return {}

    async def hdel(self, name: str, *keys: str) -> int:
        """
        Delete one or more fields from a Redis hash.

        Args:
            name: The hash name.
            *keys: Field keys to delete.

        Returns:
            int: Number of fields that were deleted, 0 on error.
        """
        if not self._client:
            logger.error("Redis client not connected")
            return 0

        if not keys:
            return 0

        try:
            result: int = await self._client.hdel(name, *keys)  # type: ignore[misc]
            logger.debug("Deleted %d fields from hash '%s'", result, name)
            return result
        except RedisError:
            logger.exception("Failed to hdel from '%s'", name)
            return 0


# =============================================================================
# Cache Decorator
# =============================================================================


def cache_result(
    ttl: int = 300,
    key_prefix: str = "",
) -> Callable[..., Any]:
    """
    Decorator for caching async function results in Redis.

    This decorator automatically caches the return value of decorated functions
    using Redis. It generates a cache key from the function name and arguments,
    checks for cached values before execution, and stores results with the
    specified TTL on cache misses.

    Cache Key Generation:
    - Format: "{prefix}:{function_name}:{arg1}:{arg2}:..."
    - Arguments are converted to strings for key construction
    - Keyword arguments are sorted alphabetically for consistent keys

    TTL Recommendations (per Agent Action Plan):
    - Asset metadata: 300 seconds (5 minutes)
    - Fingerprint data: 3600 seconds (1 hour)
    - Conversation context: 3600 seconds (1 hour)

    Args:
        ttl: Time-to-live for cached results in seconds. Default is 300 (5 minutes).
        key_prefix: Optional prefix for cache keys to namespace different cache types.

    Returns:
        Callable: Decorated function with caching behavior.

    Example:
        ```python
        @cache_result(ttl=300, key_prefix="assets")
        async def get_asset_metadata(asset_id: str):
            # This will be cached for 5 minutes
            return await fetch_from_database(asset_id)

        @cache_result(ttl=3600, key_prefix="fingerprints")
        async def get_fingerprint_data(fingerprint_id: str):
            # This will be cached for 1 hour
            return await compute_fingerprint(fingerprint_id)
        ```
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get Redis client
            client = get_redis_client()
            if client is None or not await client.is_connected():
                # If Redis is unavailable, execute function directly
                logger.warning(
                    "Redis unavailable for caching, executing %s directly",
                    func.__name__,
                )
                return await func(*args, **kwargs)

            # Generate cache key
            key_parts = [key_prefix] if key_prefix else []
            key_parts.append(func.__name__)

            # Add positional arguments to key (skip 'self' if present)
            for i, arg in enumerate(args):
                # Skip 'self' parameter for methods
                if (
                    i == 0
                    and hasattr(arg, "__class__")
                    and not isinstance(arg, (str, int, float, bool))
                ):
                    continue
                key_parts.append(str(arg))

            # Add keyword arguments to key (sorted for consistency)
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}={kwargs[k]}")

            cache_key = ":".join(key_parts)

            # Check cache
            try:
                cached_value = await client.get_json(cache_key)
                if cached_value is not None:
                    logger.debug("Cache hit for key '%s'", cache_key)
                    return cached_value
            except Exception as e:
                logger.warning("Cache lookup failed for '%s': %s", cache_key, str(e))

            # Cache miss - execute function
            logger.debug("Cache miss for key '%s'", cache_key)
            result = await func(*args, **kwargs)

            # Store result in cache
            try:
                await client.set_json(cache_key, result, ttl=ttl)
                logger.debug("Cached result for key '%s' with TTL=%d", cache_key, ttl)
            except Exception as e:
                logger.warning("Failed to cache result for '%s': %s", cache_key, str(e))

            return result

        return wrapper

    return decorator


# =============================================================================
# Module-Level Initialization Functions
# =============================================================================


async def init_redis(settings: Settings | None = None) -> RedisClient:
    """
    Initialize and connect the global Redis client singleton.

    This function should be called once during application startup
    (e.g., in FastAPI lifespan event). It creates the RedisClient
    singleton and establishes the connection.

    Args:
        settings: Optional Settings instance. If not provided, will be loaded
                 from environment using get_settings().

    Returns:
        RedisClient: The initialized and connected Redis client instance.

    Raises:
        RuntimeError: If Redis connection fails after retry attempts.

    Example:
        ```python
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_redis()
            yield
            await close_redis()
        ```
    """
    if _container.client is not None:
        logger.warning("Redis client already initialized")
        return _container.client

    logger.info("Initializing Redis client...")
    _container.client = RedisClient(settings)

    connected = await _container.client.connect()
    if not connected:
        logger.error("Failed to initialize Redis client")
        raise RuntimeError("Failed to connect to Redis after multiple attempts")

    logger.info("Redis client initialized successfully")
    return _container.client


async def close_redis() -> None:
    """
    Close the global Redis client connection.

    This function should be called during application shutdown
    (e.g., in FastAPI lifespan event) to release resources gracefully.

    Safe to call multiple times - subsequent calls have no effect.

    Example:
        ```python
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_redis()
            yield
            await close_redis()
        ```
    """
    if _container.client is not None:
        logger.info("Closing Redis client...")
        await _container.client.close()
        _container.client = None
        logger.info("Redis client closed")
    else:
        logger.debug("Redis client already closed or not initialized")


def get_redis_client() -> RedisClient | None:
    """
    Get the global Redis client singleton instance.

    Returns the initialized RedisClient instance. This function should
    only be called after init_redis() has been executed during startup.

    Returns:
        Optional[RedisClient]: The Redis client instance, or None if not initialized.

    Raises:
        RuntimeError: If called before init_redis() in production mode.

    Example:
        ```python
        client = get_redis_client()
        if client:
            await client.set("key", "value", ttl=300)
        ```
    """
    if _container.client is None:
        logger.warning("Redis client not initialized. Call init_redis() first.")
        return None

    return _container.client


# =============================================================================
# Cache Key Constants
# =============================================================================


class CacheKeys:
    """
    Constants for cache key prefixes used throughout the application.

    These prefixes help organize cached data and make cache invalidation
    easier by grouping related keys.

    TTL Guidelines (per Agent Action Plan section 0.8):
    - ASSET_METADATA: 300 seconds (5 minutes)
    - FINGERPRINT: 3600 seconds (1 hour)
    - CONVERSATION: 3600 seconds (1 hour)
    - SESSION: Configured by jwt_expiration_hours in Settings
    """

    ASSET_METADATA = "asset:metadata"
    FINGERPRINT = "fingerprint"
    CONVERSATION = "conversation"
    SESSION = "session"
    USER = "user"
    ANALYTICS = "analytics"


class CacheTTL:
    """
    Constants for cache TTL values in seconds.

    These values align with the Agent Action Plan section 0.8 specifications:
    - Metadata: 5 minutes (300 seconds)
    - Fingerprints: 1 hour (3600 seconds)
    - Conversation context: 1 hour (3600 seconds)
    """

    METADATA = 300  # 5 minutes
    FINGERPRINT = 3600  # 1 hour
    CONVERSATION = 3600  # 1 hour
    SESSION = 86400  # 24 hours (default JWT expiration)
    SHORT = 60  # 1 minute
    LONG = 7200  # 2 hours
