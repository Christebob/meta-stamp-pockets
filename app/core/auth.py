"""
META-STAMP V3 Authentication Module

This module provides comprehensive authentication services for the META-STAMP V3 platform,
implementing Auth0 JWT validation with automatic fallback to local HS256 JWT generation
when Auth0 is not configured. Key features include:

- Auth0 RS256 JWT validation using public keys from JWKS endpoint
- Local HS256 JWT generation and validation with 24-hour expiration
- FastAPI dependency injection for route protection
- Redis-based user caching (5-minute TTL) and session management
- Automatic authentication strategy selection based on configuration
- Optional authentication for endpoints that work with or without auth

Per Agent Action Plan section 0.3:
- Primary Auth0 integration with automatic fallback to local JWT stub (HS256, 24h expiry)
- JWT tokens must use HS256 for local fallback with 24-hour expiration
- Auth0 integration must validate JWT signatures using public keys

Per Agent Action Plan section 0.8:
- User caching with 5-minute TTL in Redis
- Session management using Redis with JWT expiration TTL

Usage:
    ```python
    from fastapi import Depends
    from app.core.auth import get_current_user, security

    @router.get("/protected")
    async def protected_route(user: dict = Depends(get_current_user)):
        return {"user_id": user["_id"]}

    @router.get("/optional-auth")
    async def optional_auth_route(user: Optional[dict] = Depends(get_current_user_optional)):
        if user:
            return {"user_id": user["_id"]}
        return {"message": "Anonymous user"}
    ```
"""

import contextlib
import json
import logging

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import requests

from bson import ObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.utils import base64url_decode

from app.config import Settings, get_settings
from app.core.database import get_db_client
from app.core.redis_client import get_redis_client


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Security Scheme
# =============================================================================

# HTTPBearer security scheme for extracting Bearer tokens from Authorization header
security = HTTPBearer(
    scheme_name="Bearer",
    description="JWT Bearer token authentication. "
    "Use Auth0 tokens in production or local JWT tokens in development.",
    auto_error=True,
)


# =============================================================================
# Cache Configuration
# =============================================================================

# User cache TTL in seconds (5 minutes per Agent Action Plan section 0.8)
USER_CACHE_TTL = 300

# Session cache key prefix
SESSION_KEY_PREFIX = "session"

# User cache key prefix
USER_KEY_PREFIX = "user"

# =============================================================================
# JWT Format Constants
# =============================================================================

# JWT format: header.payload.signature (3 parts)
JWT_PARTS_COUNT = 3

# Base64 padding byte boundary
BASE64_PADDING_BOUNDARY = 4


# =============================================================================
# Auth0 Token Validator
# =============================================================================


class Auth0TokenValidator:
    """
    Auth0 JWT token validator using RS256 algorithm with public key verification.

    This class handles Auth0 token validation by:
    1. Extracting the key ID (kid) from the token header
    2. Fetching the corresponding public key from Auth0's JWKS endpoint
    3. Verifying the token signature using RS256 algorithm
    4. Validating token claims (audience, issuer, expiration)

    The JWKS (JSON Web Key Set) is fetched from Auth0's well-known endpoint
    and cached for subsequent validations.

    Attributes:
        settings: Application settings containing Auth0 configuration
        _jwks_cache: Cached JWKS data to avoid repeated HTTP requests
        _jwks_cache_time: Timestamp when JWKS was cached

    Example:
        ```python
        settings = get_settings()
        validator = Auth0TokenValidator(settings)
        payload = await validator.validate_auth0_token(token)
        print(f"User ID: {payload['sub']}")
        ```
    """

    # JWKS cache duration in seconds (1 hour)
    JWKS_CACHE_DURATION = 3600

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Auth0TokenValidator with application settings.

        Args:
            settings: Settings instance containing Auth0 configuration including
                     auth0_domain, auth0_api_audience, and auth0_client_id.
        """
        self.settings = settings
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: datetime | None = None

        logger.info(
            "Auth0TokenValidator initialized for domain: %s",
            settings.auth0_domain,
        )

    def _get_jwks_url(self) -> str:
        """
        Construct the JWKS endpoint URL for the configured Auth0 domain.

        Returns:
            str: Full URL to the Auth0 JWKS endpoint.
        """
        return f"https://{self.settings.auth0_domain}/.well-known/jwks.json"

    def _fetch_jwks(self) -> dict[str, Any]:
        """
        Fetch JWKS from Auth0 with caching to reduce HTTP requests.

        Implements caching with 1-hour TTL to avoid hitting Auth0's
        rate limits and improve performance.

        Returns:
            dict: The JWKS containing public keys for token verification.

        Raises:
            HTTPException: If JWKS fetch fails.
        """
        now = datetime.now(UTC)

        # Check if cache is still valid
        if self._jwks_cache is not None and self._jwks_cache_time is not None:
            cache_age = (now - self._jwks_cache_time).total_seconds()
            if cache_age < self.JWKS_CACHE_DURATION:
                logger.debug("Using cached JWKS (age: %.0f seconds)", cache_age)
                return self._jwks_cache

        # Fetch fresh JWKS from Auth0
        jwks_url = self._get_jwks_url()
        logger.info("Fetching JWKS from: %s", jwks_url)

        try:
            response = requests.get(jwks_url, timeout=10)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_time = now
            logger.info("JWKS fetched and cached successfully")
            return self._jwks_cache
        except requests.RequestException as e:
            logger.exception("Failed to fetch JWKS from Auth0")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to verify token: Auth0 JWKS endpoint unavailable",
            ) from e

    async def get_auth0_public_key(self, token: str) -> dict[str, Any]:
        """
        Get the Auth0 public key corresponding to the token's key ID (kid).

        Parses the token header without verification to extract the key ID,
        then fetches the matching public key from Auth0's JWKS endpoint.

        Args:
            token: The JWT token string to find the public key for.

        Returns:
            dict: The RSA public key in JWK format for token verification.

        Raises:
            HTTPException: If the token header is invalid, kid is missing,
                          or no matching key is found in JWKS.
        """
        try:
            # Decode token header without verification to get kid
            unverified_header = decode_token_without_verification(token)

            kid = unverified_header.get("kid")
            if not kid:
                logger.warning("Token header missing 'kid' claim")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing key ID",
                )

            # Fetch JWKS and find matching key
            jwks = self._fetch_jwks()
            keys_list: list[dict[str, Any]] = jwks.get("keys", [])
            for key in keys_list:
                if key.get("kid") == kid:
                    logger.debug("Found matching public key for kid: %s", kid)
                    return dict(key)

            logger.warning("No matching key found for kid: %s", kid)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: key not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error getting Auth0 public key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
            ) from e

    async def validate_auth0_token(self, token: str) -> dict[str, Any]:
        """
        Validate an Auth0 JWT token with full verification.

        Performs complete token validation:
        1. Fetches the appropriate public key from Auth0 JWKS
        2. Verifies the token signature using RS256 algorithm
        3. Validates audience matches configured API audience
        4. Validates issuer matches configured Auth0 domain
        5. Validates token expiration

        Args:
            token: The JWT token string to validate.

        Returns:
            dict: The decoded and verified token payload containing claims.

        Raises:
            HTTPException: With 401 status if token validation fails for any reason.
        """
        try:
            # Get the public key for this token
            rsa_key = await self.get_auth0_public_key(token)

            # Decode and verify the token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self.settings.auth0_api_audience,
                issuer=f"https://{self.settings.auth0_domain}/",
            )

            logger.info(
                "Auth0 token validated successfully for subject: %s",
                payload.get("sub", "unknown"),
            )
            result: dict[str, Any] = dict(payload) if payload else {}
            return result

        except jwt.ExpiredSignatureError as e:
            logger.warning("Auth0 token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            ) from e
        except jwt.JWTClaimsError as e:
            logger.warning("Auth0 token claims validation failed: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims",
            ) from e
        except JWTError as e:
            logger.warning("Auth0 token validation failed: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error validating Auth0 token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed",
            ) from e


# =============================================================================
# Local JWT Functions
# =============================================================================


def create_local_jwt(user_id: str, email: str, settings: Settings) -> str:
    """
    Create a local JWT token using HS256 algorithm.

    Generates a JWT for local authentication when Auth0 is not configured.
    The token includes standard claims and a type indicator to distinguish
    from Auth0 tokens.

    Token claims:
    - sub: User ID (subject)
    - email: User's email address
    - exp: Expiration timestamp (24 hours from now per Agent Action Plan)
    - iat: Issued at timestamp
    - type: "local" to indicate local authentication

    Args:
        user_id: The user's unique identifier.
        email: The user's email address.
        settings: Settings instance containing secret_key and jwt_expiration_hours.

    Returns:
        str: The encoded JWT token string.

    Example:
        ```python
        settings = get_settings()
        token = create_local_jwt("user123", "user@example.com", settings)
        ```
    """
    now = datetime.now(UTC)
    expire = now + timedelta(hours=settings.jwt_expiration_hours)

    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": now,
        "type": "local",
    }

    encoded_token = jwt.encode(
        payload,
        settings.secret_key,
        algorithm="HS256",
    )
    # jwt.encode returns str in python-jose, ensure it's a string
    token_str: str = str(encoded_token) if not isinstance(encoded_token, str) else encoded_token

    logger.info("Created local JWT for user: %s (expires: %s)", user_id, expire.isoformat())
    return token_str


def validate_local_jwt(token: str, settings: Settings) -> dict[str, Any]:
    """
    Validate a local JWT token using HS256 algorithm.

    Verifies the token signature and expiration for locally-issued JWTs.

    Args:
        token: The JWT token string to validate.
        settings: Settings instance containing secret_key for verification.

    Returns:
        dict: The decoded token payload containing claims.

    Raises:
        JWTError: If the token is invalid, expired, or signature verification fails.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
        )
        logger.debug("Local JWT validated for subject: %s", payload.get("sub", "unknown"))
        result: dict[str, Any] = dict(payload) if payload else {}
        return result
    except jwt.ExpiredSignatureError:
        logger.warning("Local JWT has expired")
        raise
    except JWTError as e:
        logger.warning("Local JWT validation failed: %s", str(e))
        raise


def create_access_token(
    user_id: str,
    email: str,
    settings: Settings | None = None,
) -> str:
    """
    Create an access token for the given user.

    Convenience function that creates a local JWT token for authentication.
    Used primarily during login and token refresh operations.

    Args:
        user_id: The user's unique identifier.
        email: The user's email address.
        settings: Optional Settings instance. If not provided, uses get_settings().

    Returns:
        str: The encoded JWT access token.

    Example:
        ```python
        token = create_access_token("user123", "user@example.com")
        ```
    """
    if settings is None:
        settings = get_settings()
    return create_local_jwt(user_id, email, settings)


# =============================================================================
# Token Verification Functions
# =============================================================================


async def verify_auth0_token(
    token: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Verify an Auth0 JWT token.

    Convenience function for Auth0 token validation.

    Args:
        token: The Auth0 JWT token string to validate.
        settings: Optional Settings instance. If not provided, uses get_settings().

    Returns:
        dict: The decoded and verified token payload.

    Raises:
        HTTPException: With 401 status if token validation fails.
    """
    if settings is None:
        settings = get_settings()
    validator = Auth0TokenValidator(settings)
    return await validator.validate_auth0_token(token)


def decode_token_without_verification(token: str) -> dict[str, Any]:
    """
    Decode a JWT token header without signature verification.

    Used internally for extracting the key ID (kid) from Auth0 tokens
    before fetching the corresponding public key for verification.

    WARNING: This function does NOT verify the token signature and should
    only be used for key lookup purposes, not for authentication.

    Args:
        token: The JWT token string.

    Returns:
        dict: The decoded token header containing algorithm and key ID.

    Raises:
        ValueError: If the token format is invalid.
    """
    try:
        # JWT format: header.payload.signature
        parts = token.split(".")
        if len(parts) != JWT_PARTS_COUNT:
            raise ValueError(f"Invalid JWT format: expected {JWT_PARTS_COUNT} parts")

        # Decode header (first part)
        # Add padding if necessary
        header_b64 = parts[0]
        padding = BASE64_PADDING_BOUNDARY - len(header_b64) % BASE64_PADDING_BOUNDARY
        if padding != BASE64_PADDING_BOUNDARY:
            header_b64 += "=" * padding

        header_bytes = base64url_decode(header_b64.encode("utf-8"))
        header_data = json.loads(header_bytes.decode("utf-8"))
        result: dict[str, Any] = dict(header_data) if isinstance(header_data, dict) else {}
        return result

    except Exception as e:
        logger.warning("Failed to decode token header: %s", str(e))
        raise ValueError(f"Invalid token format: {e!s}") from e


# =============================================================================
# Authentication Dependencies
# =============================================================================


async def authenticate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Authenticate a Bearer token using the appropriate strategy.

    Automatically selects the authentication method based on configuration:
    - If Auth0 is enabled: Validates token using Auth0 RS256 verification
    - If Auth0 is disabled: Validates token using local HS256 verification

    This is the primary authentication dependency for protected routes.

    Args:
        credentials: HTTP Authorization credentials extracted by HTTPBearer.
        settings: Application settings (injected via FastAPI dependency).

    Returns:
        dict: The validated token payload containing user claims.

    Raises:
        HTTPException: With 401 status if token validation fails.

    Example:
        ```python
        @router.get("/protected")
        async def protected_route(
            token_data: dict = Depends(authenticate_token)
        ):
            return {"user_id": token_data["sub"]}
        ```
    """
    token = credentials.credentials

    try:
        if settings.is_auth0_enabled:
            logger.debug("Using Auth0 authentication strategy")
            validator = Auth0TokenValidator(settings)
            return await validator.validate_auth0_token(token)
        logger.debug("Using local JWT authentication strategy")
        return validate_local_jwt(token, settings)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected authentication error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user(
    token_data: dict[str, Any] = Depends(authenticate_token),
) -> dict[str, Any]:
    """
    Get the current authenticated user from the database.

    Retrieves the user document from MongoDB based on the validated token claims.
    Implements Redis caching with 5-minute TTL for performance optimization.

    The user lookup strategy:
    1. Check Redis cache first (key: user:{user_id})
    2. If not cached, query MongoDB users collection
    3. Cache the result in Redis with 5-minute TTL

    Args:
        token_data: Validated token payload from authenticate_token dependency.

    Returns:
        dict: The user document from MongoDB with all profile fields.

    Raises:
        HTTPException: With 404 status if user is not found in database.

    Example:
        ```python
        @router.get("/me")
        async def get_current_user_profile(
            user: dict = Depends(get_current_user)
        ):
            return {
                "id": str(user["_id"]),
                "email": user["email"]
            }
        ```
    """
    # Extract user identifier from token
    user_id = token_data.get("sub")
    if not user_id:
        logger.warning("Token missing 'sub' claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user identifier",
        )

    # Check Redis cache first
    redis_client = get_redis_client()
    cache_key = f"{USER_KEY_PREFIX}:{user_id}"

    if redis_client:
        try:
            cached_user = await redis_client.get_json(cache_key)
            if cached_user:
                logger.debug("User found in cache: %s", user_id)
                user_data: dict[str, Any] = dict(cached_user) if cached_user else {}
                return user_data
        except Exception as e:
            logger.warning("Redis cache lookup failed: %s", str(e))

    # Query database for user
    try:
        db_client = get_db_client()
        users_collection = db_client.get_users_collection()

        # Try to find by MongoDB _id first (for local auth), then by auth0_id
        user = None

        # Check if user_id looks like a MongoDB ObjectId
        if ObjectId.is_valid(user_id):
            user = await users_collection.find_one({"_id": ObjectId(user_id)})

        # If not found, try auth0_id (for Auth0 tokens)
        if user is None:
            user = await users_collection.find_one({"auth0_id": user_id})

        # Still not found, try by email if token contains email
        if user is None and token_data.get("email"):
            user = await users_collection.find_one({"email": token_data.get("email")})

        if user is None:
            logger.warning("User not found in database: %s", user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Convert ObjectId to string for JSON serialization
        user["_id"] = str(user["_id"])

        # Cache user in Redis
        if redis_client:
            try:
                await redis_client.set_json(cache_key, user, ttl=USER_CACHE_TTL)
                logger.debug("User cached in Redis: %s (TTL: %d)", user_id, USER_CACHE_TTL)
            except Exception as e:
                logger.warning("Failed to cache user in Redis: %s", str(e))

        logger.info("User authenticated: %s", user.get("email", user_id))
        user_result: dict[str, Any] = dict(user) if user else {}
        return user_result

    except HTTPException:
        raise
    except RuntimeError:
        logger.exception("Database error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error getting user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from None


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any] | None:
    """
    Get the current user if authenticated, or None if not.

    Used for endpoints that work with or without authentication,
    providing different behavior for authenticated vs anonymous users.

    Args:
        credentials: Optional HTTP Authorization credentials.
        settings: Application settings (injected via FastAPI dependency).

    Returns:
        Optional[dict]: The user document if authenticated, None otherwise.

    Example:
        ```python
        @router.get("/public-with-personalization")
        async def public_route(
            user: Optional[dict] = Depends(get_current_user_optional)
        ):
            if user:
                return {"message": f"Hello, {user['email']}!"}
            return {"message": "Hello, guest!"}
        ```
    """
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        if settings.is_auth0_enabled:
            validator = Auth0TokenValidator(settings)
            token_data = await validator.validate_auth0_token(token)
        else:
            token_data = validate_local_jwt(token, settings)

        # Get user from database (uses the same logic as get_current_user)
        user_id = token_data.get("sub")
        if not user_id:
            return None

        # Check cache
        redis_client = get_redis_client()
        cache_key = f"{USER_KEY_PREFIX}:{user_id}"

        if redis_client:
            try:
                cached_user = await redis_client.get_json(cache_key)
                if cached_user:
                    cached_data: dict[str, Any] = dict(cached_user) if cached_user else {}
                    return cached_data
            except Exception:
                pass

        # Query database
        db_client = get_db_client()
        users_collection = db_client.get_users_collection()

        user = None
        if ObjectId.is_valid(user_id):
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if user is None:
            user = await users_collection.find_one({"auth0_id": user_id})
        if user is None and token_data.get("email"):
            user = await users_collection.find_one({"email": token_data.get("email")})

        if user:
            user["_id"] = str(user["_id"])
            if redis_client:
                with contextlib.suppress(Exception):
                    await redis_client.set_json(cache_key, user, ttl=USER_CACHE_TTL)
            user_data: dict[str, Any] = dict(user) if user else {}
            return user_data

        return None

    except Exception as e:
        logger.debug("Optional auth failed: %s", str(e))
        return None


# =============================================================================
# Session Management Functions
# =============================================================================


async def create_user_session(user_id: str, token: str) -> bool:
    """
    Create a user session in Redis.

    Stores the session with a TTL matching the JWT expiration time (24 hours default).
    Used for tracking active sessions and enabling token revocation.

    Args:
        user_id: The user's unique identifier.
        token: The JWT token for this session.

    Returns:
        bool: True if session created successfully, False otherwise.

    Example:
        ```python
        token = create_access_token(user_id, email)
        await create_user_session(user_id, token)
        ```
    """
    redis_client = get_redis_client()
    if redis_client is None:
        logger.warning("Redis not available for session creation")
        return False

    try:
        session_key = f"{SESSION_KEY_PREFIX}:{user_id}"
        settings = get_settings()
        ttl = settings.jwt_expiration_hours * 3600  # Convert hours to seconds

        session_data = {
            "token": token,
            "created_at": datetime.now(UTC).isoformat(),
            "user_id": user_id,
        }

        await redis_client.set_json(session_key, session_data, ttl=ttl)
        logger.info("Session created for user: %s (TTL: %d seconds)", user_id, ttl)
        return True

    except Exception:
        logger.exception("Failed to create session for user: %s", user_id)
        return False


async def revoke_user_session(user_id: str) -> bool:
    """
    Revoke a user's session by deleting it from Redis.

    Used for logout functionality to invalidate the user's session
    before the JWT expires naturally.

    Args:
        user_id: The user's unique identifier.

    Returns:
        bool: True if session was revoked successfully, False otherwise.

    Example:
        ```python
        @router.post("/logout")
        async def logout(user: dict = Depends(get_current_user)):
            await revoke_user_session(user["_id"])
            return {"message": "Logged out successfully"}
        ```
    """
    redis_client = get_redis_client()
    if redis_client is None:
        logger.warning("Redis not available for session revocation")
        return False

    try:
        session_key = f"{SESSION_KEY_PREFIX}:{user_id}"
        deleted = await redis_client.delete(session_key)

        # Also clear user cache
        user_cache_key = f"{USER_KEY_PREFIX}:{user_id}"
        await redis_client.delete(user_cache_key)

        if deleted:
            logger.info("Session revoked for user: %s", user_id)
        else:
            logger.debug("No session found to revoke for user: %s", user_id)

        return True

    except Exception:
        logger.exception("Failed to revoke session for user: %s", user_id)
        return False


async def verify_session(user_id: str) -> bool:
    """
    Verify that a user has an active session in Redis.

    Used to check if a user's session is still valid before processing
    requests that require an active session.

    Args:
        user_id: The user's unique identifier.

    Returns:
        bool: True if session exists and is valid, False otherwise.

    Example:
        ```python
        if await verify_session(user_id):
            # Process request
            pass
        else:
            # Session expired or revoked
            raise HTTPException(status_code=401)
        ```
    """
    redis_client = get_redis_client()
    if redis_client is None:
        logger.warning("Redis not available for session verification")
        # If Redis is unavailable, allow the request (token validation still occurs)
        return True

    try:
        session_key = f"{SESSION_KEY_PREFIX}:{user_id}"
        exists = await redis_client.exists(session_key)

        if exists:
            logger.debug("Session verified for user: %s", user_id)
        else:
            logger.debug("No active session for user: %s", user_id)

        return exists

    except Exception as e:
        logger.warning("Session verification failed for user %s: %s", user_id, str(e))
        # On error, allow the request (token validation still occurs)
        return True


# =============================================================================
# User Authentication Function
# =============================================================================


async def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """
    Authenticate a user with email and password.

    Used for local authentication when Auth0 is not configured.
    Validates credentials against the database and returns the user document.

    Note: This function is a placeholder for local authentication.
    In production with Auth0, authentication is handled by Auth0.

    Args:
        email: The user's email address.
        password: The user's password (will be hashed and compared).

    Returns:
        Optional[dict]: The user document if authentication succeeds, None otherwise.

    Example:
        ```python
        user = await authenticate_user("user@example.com", "password123")
        if user:
            token = create_access_token(str(user["_id"]), user["email"])
        ```
    """
    try:
        db_client = get_db_client()
        users_collection = db_client.get_users_collection()

        # Find user by email
        user = await users_collection.find_one({"email": email})
        if user is None:
            logger.warning("Authentication failed: user not found for email %s", email)
            return None

        # Verify password
        # Note: In a real implementation, we would use bcrypt or similar
        # For now, we check if hashed_password exists and matches
        stored_password = user.get("hashed_password")
        if stored_password is None:
            logger.warning("Authentication failed: no password set for user %s", email)
            return None

        # Simple password comparison for development
        # In production, use passlib with bcrypt:
        # from passlib.context import CryptContext
        # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        # if not pwd_context.verify(password, stored_password):
        #     return None

        # For development/testing, direct comparison (NOT secure for production)
        hashed_input = sha256(password.encode()).hexdigest()
        if hashed_input != stored_password:
            logger.warning("Authentication failed: invalid password for user %s", email)
            return None

        # Update last login timestamp
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.now(UTC)}},
        )

        # Convert ObjectId to string for return
        user["_id"] = str(user["_id"])
        logger.info("User authenticated successfully: %s", email)
        user_result: dict[str, Any] = dict(user) if user else {}
        return user_result

    except RuntimeError:
        logger.exception("Database error during authentication")
        return None
    except Exception:
        logger.exception("Unexpected error during authentication")
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Auth0 validator class
    "Auth0TokenValidator",
    # Authentication dependencies
    "authenticate_token",
    # User authentication
    "authenticate_user",
    "create_access_token",
    # Local JWT functions
    "create_local_jwt",
    # Session management
    "create_user_session",
    # Token utilities
    "decode_token_without_verification",
    "get_current_user",
    "get_current_user_optional",
    "revoke_user_session",
    # Security scheme
    "security",
    "validate_local_jwt",
    # Auth0 verification
    "verify_auth0_token",
    "verify_session",
]
