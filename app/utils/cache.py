"""
META-STAMP V3 Redis Caching Utilities Module

This module provides comprehensive Redis-based caching utilities for optimizing
frequently accessed data across backend services. It implements:

- Decorator-based caching with configurable TTL for async functions
- Cache key generation from function arguments using deterministic hashing
- Async cache operations (get, set, delete) with JSON serialization
- Pattern-based cache invalidation for user and asset data
- Pre-defined TTL constants per Agent Action Plan specifications

Cache TTL Configurations (per Agent Action Plan section 0.4, 0.8):
- Asset metadata: 5 minutes (300 seconds) - CACHE_TTL_METADATA
- Fingerprint data: 1 hour (3600 seconds) - CACHE_TTL_FINGERPRINTS
- Session/conversation context: 1 hour (3600 seconds) - CACHE_TTL_SESSION

Usage:
    ```python
    from app.utils.cache import (
        cache_decorator,
        get_cached_value,
        set_cached_value,
        delete_cached_value,
        clear_cache_pattern,
        invalidate_user_cache,
        invalidate_asset_cache,
        CACHE_TTL_METADATA,
        CACHE_TTL_FINGERPRINTS,
        CACHE_TTL_SESSION,
    )

    # Use decorator for automatic caching
    @cache_decorator(ttl_seconds=CACHE_TTL_METADATA, key_prefix="asset")
    async def get_asset_metadata(asset_id: str):
        return await fetch_from_database(asset_id)

    # Manual cache operations
    await set_cached_value("my_key", {"data": "value"}, ttl_seconds=300)
    value = await get_cached_value("my_key")
    await delete_cached_value("my_key")

    # Invalidate user-related cache entries
    await invalidate_user_cache("user123")
    ```
"""

import hashlib
import json
import logging

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from app.core.redis_client import get_redis_client


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# TTL Constants (per Agent Action Plan section 0.4, 0.8, 0.10)
# =============================================================================

# Cache TTL for asset metadata - 5 minutes
CACHE_TTL_METADATA: int = 300

# Cache TTL for fingerprint data - 1 hour
CACHE_TTL_FINGERPRINTS: int = 3600

# Cache TTL for session and conversation context - 1 hour
CACHE_TTL_SESSION: int = 3600


# Type variable for generic function return types in decorator
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Cache Key Generation
# =============================================================================


def generate_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a deterministic cache key from prefix and function arguments.

    This function creates a consistent, unique cache key by combining a prefix
    with a hash of the provided arguments. It handles various argument types
    including unhashable types (dicts, lists) by converting them to JSON before
    hashing.

    Key Format: "{prefix}:{md5_hash_of_args}"

    The MD5 hash ensures:
    - Consistent key length regardless of argument complexity
    - Deterministic output for identical inputs
    - No special characters that might conflict with Redis key patterns

    Args:
        prefix: A string prefix for namespacing the cache key (e.g., "asset", "user").
        *args: Positional arguments to include in the key hash.
        **kwargs: Keyword arguments to include in the key hash.

    Returns:
        str: A cache key in the format "{prefix}:{hash}".

    Example:
        >>> generate_cache_key("asset", "12345", include_metadata=True)
        "asset:a1b2c3d4e5f6..."

        >>> generate_cache_key("user", {"id": 123, "role": "admin"})
        "user:f7e8d9c0b1a2..."
    """
    # Build a list of argument representations for hashing
    parts: list[str] = []

    # Process positional arguments
    for arg in args:
        try:
            # Attempt direct string conversion for simple types
            if isinstance(arg, (str, int, float, bool, type(None))):
                parts.append(str(arg))
            else:
                # Convert complex types to JSON for consistent representation
                parts.append(json.dumps(arg, sort_keys=True, default=str))
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            logger.warning(
                "Failed to serialize argument for cache key, using repr: %s",
                str(e),
            )
            parts.append(repr(arg))

    # Process keyword arguments (sorted for consistency)
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={json.dumps(value, sort_keys=True, default=str)}")
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to serialize kwarg '%s' for cache key, using repr: %s",
                key,
                str(e),
            )
            parts.append(f"{key}={value!r}")

    # Create deterministic hash of all arguments
    args_string = ":".join(parts) if parts else ""
    hash_input = args_string.encode("utf-8")
    args_hash = hashlib.md5(hash_input).hexdigest()

    # Combine prefix with hash
    cache_key = f"{prefix}:{args_hash}" if prefix else args_hash

    logger.debug("Generated cache key: %s", cache_key)
    return cache_key


# =============================================================================
# Cache Decorator
# =============================================================================


def cache_decorator(
    ttl_seconds: int = 300,
    key_prefix: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator for caching async function results in Redis with configurable TTL.

    This decorator automatically caches the return value of decorated async functions.
    It generates a cache key from the function name and arguments, checks for cached
    values before execution, and stores results with the specified TTL on cache misses.

    The decorator preserves the original function's metadata (name, docstring,
    annotations) using functools.wraps for proper introspection and documentation.

    Cache Key Generation:
    - If key_prefix provided: "{prefix}:{function_name}:{hash_of_args}"
    - If no prefix: "{function_name}:{hash_of_args}"

    TTL Recommendations (per Agent Action Plan section 0.4, 0.8):
    - Asset metadata: 300 seconds (5 minutes) - use CACHE_TTL_METADATA
    - Fingerprint data: 3600 seconds (1 hour) - use CACHE_TTL_FINGERPRINTS
    - Session context: 3600 seconds (1 hour) - use CACHE_TTL_SESSION

    Args:
        ttl_seconds: Time-to-live for cached results in seconds. Default is 300
                    (5 minutes). Must be a positive integer.
        key_prefix: Optional prefix for cache keys to namespace different cache types.
                   If None, only the function name is used as the prefix.

    Returns:
        Callable: A decorator that wraps async functions with caching behavior.

    Raises:
        No exceptions are raised by the decorator itself. If Redis is unavailable,
        the decorated function executes normally without caching.

    Example:
        ```python
        @cache_decorator(ttl_seconds=CACHE_TTL_METADATA, key_prefix="asset")
        async def get_asset_metadata(asset_id: str) -> dict:
            # This will be cached for 5 minutes
            return await fetch_from_database(asset_id)

        @cache_decorator(ttl_seconds=CACHE_TTL_FINGERPRINTS, key_prefix="fingerprint")
        async def get_fingerprint_data(fingerprint_id: str) -> dict:
            # This will be cached for 1 hour
            return await compute_fingerprint(fingerprint_id)

        # Using default TTL (5 minutes)
        @cache_decorator(key_prefix="user")
        async def get_user_profile(user_id: str) -> dict:
            return await get_user_from_db(user_id)
        ```
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get Redis client singleton
            client = get_redis_client()

            if client is None:
                # Redis not initialized, execute function directly
                logger.warning(
                    "Redis client not available for caching, executing %s directly",
                    func.__name__,
                )
                return await func(*args, **kwargs)

            # Check connection status
            try:
                connected = await client.is_connected()
                if not connected:
                    logger.warning(
                        "Redis not connected for caching, executing %s directly",
                        func.__name__,
                    )
                    return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Error checking Redis connection for %s: %s",
                    func.__name__,
                    str(e),
                )
                return await func(*args, **kwargs)

            # Build cache key prefix
            effective_prefix = f"{key_prefix}:{func.__name__}" if key_prefix else func.__name__

            # Filter out 'self' parameter for method caching
            filtered_args = []
            for i, arg in enumerate(args):
                # Skip 'self' or 'cls' parameter (first arg that's an object instance)
                if (
                    i == 0
                    and hasattr(arg, "__class__")
                    and not isinstance(
                        arg, (str, int, float, bool, bytes, type(None), list, dict, tuple)
                    )
                ):
                    continue
                filtered_args.append(arg)

            # Generate cache key from arguments
            cache_key = generate_cache_key(effective_prefix, *filtered_args, **kwargs)

            # Try to get cached value
            try:
                cached_value = await client.get_json(cache_key)
                if cached_value is not None:
                    logger.debug(
                        "Cache hit for %s with key '%s'",
                        func.__name__,
                        cache_key,
                    )
                    return cached_value
            except Exception as e:
                logger.warning(
                    "Cache lookup failed for %s key '%s': %s",
                    func.__name__,
                    cache_key,
                    str(e),
                )

            # Cache miss - execute the original function
            logger.debug(
                "Cache miss for %s with key '%s'",
                func.__name__,
                cache_key,
            )
            result = await func(*args, **kwargs)

            # Store result in cache
            try:
                success = await client.set_json(cache_key, result, ttl=ttl_seconds)
                if success:
                    logger.debug(
                        "Cached result for %s key '%s' with TTL=%d seconds",
                        func.__name__,
                        cache_key,
                        ttl_seconds,
                    )
                else:
                    logger.warning(
                        "Failed to cache result for %s key '%s'",
                        func.__name__,
                        cache_key,
                    )
            except Exception as e:
                logger.warning(
                    "Error caching result for %s key '%s': %s",
                    func.__name__,
                    cache_key,
                    str(e),
                )

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Async Cache Operations
# =============================================================================


async def get_cached_value(key: str) -> Any | None:
    """
    Retrieve a cached value from Redis by key.

    This function fetches the value associated with the given key from Redis
    and deserializes it from JSON format. Returns None if the key doesn't exist
    or if there's an error during retrieval.

    Args:
        key: The cache key to retrieve.

    Returns:
        Optional[Any]: The cached value (deserialized from JSON) if found,
                      None if the key doesn't exist, Redis is unavailable,
                      or an error occurs.

    Example:
        ```python
        # Store a value
        await set_cached_value("user:123:profile", {"name": "John", "age": 30})

        # Retrieve the value
        profile = await get_cached_value("user:123:profile")
        if profile:
            print(f"Name: {profile['name']}")
        else:
            print("Profile not found in cache")
        ```
    """
    client = get_redis_client()

    if client is None:
        logger.warning("Redis client not available for get_cached_value")
        return None

    try:
        value = await client.get_json(key)
        if value is not None:
            logger.debug("Retrieved cached value for key '%s'", key)
        else:
            logger.debug("Cache miss for key '%s'", key)
        return value

    except json.JSONDecodeError:
        logger.exception("Failed to decode JSON for key '%s'", key)
        return None
    except Exception:
        logger.exception("Error retrieving cached value for key '%s'", key)
        return None


async def set_cached_value(
    key: str,
    value: Any,
    ttl_seconds: int = 300,
) -> bool:
    """
    Store a value in Redis cache with optional TTL.

    This function serializes the provided value to JSON format and stores it
    in Redis with the specified key and time-to-live. The value can be any
    JSON-serializable Python object (dict, list, str, int, etc.).

    Args:
        key: The cache key to set.
        value: The value to cache (must be JSON-serializable).
        ttl_seconds: Time-to-live in seconds. Default is 300 (5 minutes).
                    Must be a positive integer.

    Returns:
        bool: True if the value was successfully cached, False otherwise.

    Example:
        ```python
        # Cache asset metadata for 5 minutes
        success = await set_cached_value(
            "asset:12345:metadata",
            {"filename": "image.png", "size": 1024, "type": "image/png"},
            ttl_seconds=CACHE_TTL_METADATA
        )
        if success:
            print("Asset metadata cached successfully")

        # Cache fingerprint data for 1 hour
        await set_cached_value(
            "fingerprint:67890",
            {"hash": "abc123", "embeddings": [0.1, 0.2, 0.3]},
            ttl_seconds=CACHE_TTL_FINGERPRINTS
        )
        ```
    """
    client = get_redis_client()

    if client is None:
        logger.warning("Redis client not available for set_cached_value")
        return False

    try:
        success = await client.set_json(key, value, ttl=ttl_seconds)
        if success:
            logger.debug("Set cached value for key '%s' with TTL=%d", key, ttl_seconds)
        else:
            logger.warning("Failed to set cached value for key '%s'", key)
        return success

    except (TypeError, ValueError):
        logger.exception("Failed to serialize value for key '%s'", key)
        return False
    except Exception:
        logger.exception("Error setting cached value for key '%s'", key)
        return False


async def delete_cached_value(key: str) -> bool:
    """
    Delete a cached value from Redis by key.

    This function removes the value associated with the given key from Redis.
    Returns True if the key was deleted, False if the key didn't exist or
    if there was an error.

    Args:
        key: The cache key to delete.

    Returns:
        bool: True if the key was successfully deleted, False if the key
              didn't exist, Redis is unavailable, or an error occurred.

    Example:
        ```python
        # Delete a specific cache entry
        deleted = await delete_cached_value("user:123:profile")
        if deleted:
            print("Cache entry deleted")
        else:
            print("Cache entry not found or already deleted")
        ```
    """
    client = get_redis_client()

    if client is None:
        logger.warning("Redis client not available for delete_cached_value")
        return False

    try:
        deleted = await client.delete(key)
        if deleted:
            logger.debug("Deleted cached value for key '%s'", key)
        else:
            logger.debug("Key '%s' not found or already deleted", key)
        return deleted

    except Exception:
        logger.exception("Error deleting cached value for key '%s'", key)
        return False


# =============================================================================
# Pattern-Based Cache Invalidation
# =============================================================================


async def clear_cache_pattern(pattern: str) -> int:
    """
    Delete all cache keys matching a pattern.

    This function finds all Redis keys matching the specified glob-style pattern
    and deletes them. Useful for bulk cache invalidation based on key prefixes
    or patterns.

    Pattern Syntax:
    - * matches any number of characters
    - ? matches a single character
    - [abc] matches a, b, or c

    Common Patterns:
    - "user:*" - All user-related cache entries
    - "asset:123:*" - All cache entries for asset 123
    - "fingerprint:*" - All fingerprint cache entries

    Args:
        pattern: A glob-style pattern to match cache keys.

    Returns:
        int: The number of keys that were deleted. Returns 0 if no keys matched,
             Redis is unavailable, or an error occurred.

    Warning:
        Use this function with caution in production as it can delete many keys
        and may impact performance if matching a large number of keys.

    Example:
        ```python
        # Clear all user-related cache
        deleted_count = await clear_cache_pattern("user:*")
        print(f"Deleted {deleted_count} user cache entries")

        # Clear all cache for a specific asset
        deleted_count = await clear_cache_pattern("asset:12345:*")
        print(f"Deleted {deleted_count} asset cache entries")
        ```
    """
    client = get_redis_client()

    if client is None:
        logger.warning("Redis client not available for clear_cache_pattern")
        return 0

    try:
        # Access underlying Redis client to use keys() method
        # The RedisClient wrapper doesn't expose keys(), so we access _client directly
        if client._client is None:
            logger.warning("Redis client connection not established")
            return 0

        # Find all keys matching the pattern
        keys = await client._client.keys(pattern)

        if not keys:
            logger.debug("No keys found matching pattern '%s'", pattern)
            return 0

        # Delete all matching keys
        deleted_count = await client._client.delete(*keys)
        logger.info(
            "Cleared %d cache entries matching pattern '%s'",
            deleted_count,
            pattern,
        )
        return int(deleted_count)

    except Exception:
        logger.exception("Error clearing cache pattern '%s'", pattern)
        return 0


# =============================================================================
# Cache Invalidation Helpers
# =============================================================================


async def invalidate_user_cache(user_id: str) -> int:
    """
    Invalidate all cache entries associated with a specific user.

    This function clears all cached data related to a user, including profile
    information, preferences, session data, and any other user-specific cache
    entries. It uses pattern matching to find all keys with the user ID.

    Pattern Used: "user:{user_id}:*" and "*:user:{user_id}:*"

    Common cache entries invalidated:
    - User profile data
    - User preferences
    - User-specific analytics
    - User session information

    Args:
        user_id: The unique identifier of the user whose cache should be cleared.

    Returns:
        int: The total number of cache entries that were deleted.

    Example:
        ```python
        # User updated their profile, invalidate related cache
        deleted = await invalidate_user_cache("user123")
        print(f"Invalidated {deleted} cache entries for user123")

        # User logged out, clear their cache
        await invalidate_user_cache(str(current_user.id))
        ```
    """
    if not user_id:
        logger.warning("Empty user_id provided to invalidate_user_cache")
        return 0

    total_deleted = 0

    # Pattern 1: Keys starting with user:{user_id}:
    pattern1 = f"user:{user_id}:*"
    deleted1 = await clear_cache_pattern(pattern1)
    total_deleted += deleted1

    # Pattern 2: Keys containing user:{user_id} anywhere
    pattern2 = f"*:user:{user_id}:*"
    deleted2 = await clear_cache_pattern(pattern2)
    total_deleted += deleted2

    # Pattern 3: Exact user key
    pattern3 = f"user:{user_id}"
    client = get_redis_client()
    if client is not None:
        try:
            if await client.delete(pattern3):
                total_deleted += 1
        except Exception as e:
            logger.warning("Error deleting exact user key: %s", str(e))

    logger.info(
        "Invalidated %d cache entries for user '%s'",
        total_deleted,
        user_id,
    )
    return total_deleted


async def invalidate_asset_cache(asset_id: str) -> int:
    """
    Invalidate all cache entries associated with a specific asset.

    This function clears all cached data related to an asset, including metadata,
    fingerprint data, analytics, and any other asset-specific cache entries.
    It uses pattern matching to find all keys containing the asset ID.

    Pattern Used: "asset:{asset_id}:*" and "*:asset:{asset_id}:*"

    Common cache entries invalidated:
    - Asset metadata
    - Asset fingerprints
    - Asset analytics data
    - AI Touch Value calculations

    Args:
        asset_id: The unique identifier of the asset whose cache should be cleared.

    Returns:
        int: The total number of cache entries that were deleted.

    Example:
        ```python
        # Asset was updated, invalidate related cache
        deleted = await invalidate_asset_cache("asset12345")
        print(f"Invalidated {deleted} cache entries for asset12345")

        # Asset was deleted, clear all related cache
        await invalidate_asset_cache(str(deleted_asset.id))
        ```
    """
    if not asset_id:
        logger.warning("Empty asset_id provided to invalidate_asset_cache")
        return 0

    total_deleted = 0

    # Pattern 1: Keys starting with asset:{asset_id}:
    pattern1 = f"asset:{asset_id}:*"
    deleted1 = await clear_cache_pattern(pattern1)
    total_deleted += deleted1

    # Pattern 2: Keys containing asset:{asset_id} anywhere
    pattern2 = f"*:asset:{asset_id}:*"
    deleted2 = await clear_cache_pattern(pattern2)
    total_deleted += deleted2

    # Pattern 3: Exact asset key
    pattern3 = f"asset:{asset_id}"
    client = get_redis_client()
    if client is not None:
        try:
            if await client.delete(pattern3):
                total_deleted += 1
        except Exception as e:
            logger.warning("Error deleting exact asset key: %s", str(e))

    # Pattern 4: Fingerprint keys for this asset
    pattern4 = f"fingerprint:{asset_id}:*"
    deleted4 = await clear_cache_pattern(pattern4)
    total_deleted += deleted4

    # Pattern 5: Exact fingerprint key
    pattern5 = f"fingerprint:{asset_id}"
    if client is not None:
        try:
            if await client.delete(pattern5):
                total_deleted += 1
        except Exception as e:
            logger.warning("Error deleting exact fingerprint key: %s", str(e))

    logger.info(
        "Invalidated %d cache entries for asset '%s'",
        total_deleted,
        asset_id,
    )
    return total_deleted


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CACHE_TTL_FINGERPRINTS",
    # TTL Constants
    "CACHE_TTL_METADATA",
    "CACHE_TTL_SESSION",
    # Decorator
    "cache_decorator",
    # Pattern Invalidation
    "clear_cache_pattern",
    "delete_cached_value",
    # Key Generation
    "generate_cache_key",
    # Async Operations
    "get_cached_value",
    "invalidate_asset_cache",
    "invalidate_user_cache",
    "set_cached_value",
]
