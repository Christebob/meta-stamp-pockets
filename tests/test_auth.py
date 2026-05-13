"""
META-STAMP V3 Authentication Module Test Suite

Comprehensive test suite for backend/app/core/auth.py covering:
- Auth0 JWT validation using public keys (RS256)
- Local JWT generation and validation with HS256 algorithm
- 24-hour token expiration enforcement
- Authentication dependencies for FastAPI route protection
- Session management using Redis with TTL support
- Automatic fallback from Auth0 to local JWT when Auth0 not configured
- User caching with 5-minute TTL

Based on Agent Action Plan sections:
- Section 0.3: Auth0/local JWT authentication requirements
- Section 0.4: Authentication implementation details
- Section 0.5: test_auth.py requirements
- Section 0.8: 24-hour JWT expiration, Redis session management
- Section 0.10: Security testing requirements
"""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import requests

from bson import ObjectId
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.config import Settings
from app.core.auth import (
    SESSION_KEY_PREFIX,
    USER_CACHE_TTL,
    USER_KEY_PREFIX,
    Auth0TokenValidator,
    authenticate_token,
    authenticate_user,
    create_access_token,
    create_local_jwt,
    create_user_session,
    decode_token_without_verification,
    get_current_user,
    get_current_user_optional,
    revoke_user_session,
    validate_local_jwt,
    verify_session,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create test Settings with test configuration.

    Returns settings with:
    - secret_key for local JWT signing/validation
    - 24-hour JWT expiration
    - Auth0 disabled (local JWT fallback)
    """
    return Settings(
        app_env="testing",
        debug=True,
        secret_key="test-secret-key-for-jwt-minimum-32-chars",
        jwt_expiration_hours=24,
        jwt_algorithm="HS256",
        auth0_domain=None,
        auth0_client_id=None,
        auth0_client_secret=None,
        auth0_api_audience=None,
        mongodb_uri="mongodb://localhost:27017/test_metastamp",
        redis_url="localhost:6379/1",
    )


@pytest.fixture
def mock_settings_with_auth0() -> Settings:
    """
    Create test Settings with Auth0 enabled.

    Returns settings with full Auth0 configuration for testing
    Auth0 JWT validation path.
    """
    return Settings(
        app_env="testing",
        debug=True,
        secret_key="test-secret-key-for-jwt-minimum-32-chars",
        jwt_expiration_hours=24,
        jwt_algorithm="RS256",
        auth0_domain="test-tenant.auth0.com",
        auth0_client_id="test-client-id",
        auth0_client_secret="test-client-secret",
        auth0_api_audience="https://api.metastamp.test",
        mongodb_uri="mongodb://localhost:27017/test_metastamp",
        redis_url="localhost:6379/1",
    )


@pytest.fixture
def test_user() -> dict[str, Any]:
    """
    Create sample user data for testing authentication.

    Returns a user document matching MongoDB schema with:
    - Valid ObjectId as _id
    - Email address
    - Optional auth0_id for Auth0 users
    - Timestamps for created_at and last_login
    """
    return {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "email": "test@example.com",
        "auth0_id": None,
        "created_at": datetime.now(UTC),
        "last_login": datetime.now(UTC),
        "hashed_password": None,
    }


@pytest.fixture
def mock_redis() -> AsyncMock:
    """
    Create AsyncMock for Redis client operations.

    Provides mocked methods for:
    - get/set operations
    - get_json/set_json for JSON data
    - delete for cache invalidation
    - exists for session verification
    """
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.exists = AsyncMock(return_value=False)
    mock.get_json = AsyncMock(return_value=None)
    mock.set_json = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_db() -> AsyncMock:
    """
    Create AsyncMock for MongoDB client operations.

    Provides mocked collection accessors and database operations
    for user lookups and session management testing.
    """
    mock = AsyncMock()
    mock_users_collection = AsyncMock()
    mock_users_collection.find_one = AsyncMock(return_value=None)
    mock_users_collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    mock.get_users_collection = Mock(return_value=mock_users_collection)
    mock.get_database = Mock(return_value=MagicMock())
    return mock


@pytest.fixture
def mock_auth0_jwks() -> dict[str, Any]:
    """
    Create mocked Auth0 JWKS response for testing RS256 validation.

    Provides a JWKS structure with a test RSA public key
    for mocking Auth0 JWKS endpoint responses.
    """
    return {
        "keys": [
            {
                "kid": "test-kid-123",
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                "e": "AQAB",
            }
        ]
    }


@pytest.fixture
def test_jwt_token(mock_settings: Settings, test_user: dict[str, Any]) -> str:
    """
    Create a valid local JWT token for testing.

    Generates a JWT with:
    - User ID as subject (sub)
    - Email claim
    - 24-hour expiration
    - type='local' indicator
    """
    user_id = str(test_user["_id"])
    return create_local_jwt(user_id, test_user["email"], mock_settings)


@pytest.fixture
def expired_jwt_token(mock_settings: Settings, test_user: dict[str, Any]) -> str:
    """
    Create an expired JWT token for testing expiration handling.

    Generates a token that expired 1 hour ago.
    """
    user_id = str(test_user["_id"])
    now = datetime.now(UTC)
    expire = now - timedelta(hours=1)  # Already expired

    payload = {
        "sub": user_id,
        "email": test_user["email"],
        "exp": expire,
        "iat": now - timedelta(hours=25),  # Issued 25 hours ago
        "type": "local",
    }

    return jwt.encode(payload, mock_settings.secret_key, algorithm="HS256")


# =============================================================================
# Test Local JWT Creation
# =============================================================================


class TestLocalJWTCreation:
    """Test suite for create_local_jwt function."""

    def test_create_local_jwt(self, mock_settings: Settings, test_user: dict[str, Any]) -> None:
        """
        Test token creation with correct payload structure.

        Verify that created token contains:
        - sub: user ID
        - email: user email
        - exp: expiration timestamp
        - iat: issued at timestamp
        - type: 'local' indicator
        """
        user_id = str(test_user["_id"])
        email = test_user["email"]

        token = create_local_jwt(user_id, email, mock_settings)

        # Decode without verification to check payload
        payload = jwt.decode(token, mock_settings.secret_key, algorithms=["HS256"])

        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert payload["type"] == "local"
        assert "exp" in payload
        assert "iat" in payload

    def test_create_local_jwt_expiration(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test 24-hour expiration time enforcement.

        Verify that token expires approximately 24 hours from creation,
        per Agent Action Plan section 0.3 24-hour JWT requirement.
        """
        user_id = str(test_user["_id"])
        email = test_user["email"]

        token = create_local_jwt(user_id, email, mock_settings)

        payload = jwt.decode(token, mock_settings.secret_key, algorithms=["HS256"])

        exp_timestamp = payload["exp"]
        iat_timestamp = payload["iat"]

        # Expiration should be ~24 hours from issue time
        expected_expiration = iat_timestamp + (24 * 3600)  # 24 hours in seconds

        # Allow 5 seconds tolerance for test execution time
        assert abs(exp_timestamp - expected_expiration) < 5

    def test_create_local_jwt_signature(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test HS256 signature with secret key.

        Verify that:
        - Token can be validated with correct secret key
        - Token validation fails with incorrect secret key
        """
        user_id = str(test_user["_id"])
        email = test_user["email"]

        token = create_local_jwt(user_id, email, mock_settings)

        # Should validate with correct key
        payload = jwt.decode(token, mock_settings.secret_key, algorithms=["HS256"])
        assert payload["sub"] == user_id

        # Should fail with incorrect key
        wrong_key = "wrong-secret-key-that-is-32-characters"
        with pytest.raises(JWTError):
            jwt.decode(token, wrong_key, algorithms=["HS256"])

    def test_create_local_jwt_returns_string(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """Verify create_local_jwt returns a string token."""
        user_id = str(test_user["_id"])
        email = test_user["email"]

        token = create_local_jwt(user_id, email, mock_settings)

        assert isinstance(token, str)
        assert len(token) > 0
        # JWT format: header.payload.signature (3 parts)
        assert len(token.split(".")) == 3


# =============================================================================
# Test Local JWT Validation
# =============================================================================


class TestLocalJWTValidation:
    """Test suite for validate_local_jwt function."""

    def test_validate_local_jwt_valid_token(
        self, mock_settings: Settings, test_jwt_token: str, test_user: dict[str, Any]
    ) -> None:
        """
        Test that valid token returns correct payload.

        Verify validate_local_jwt successfully decodes a valid token
        and returns the complete payload with all claims.
        """
        payload = validate_local_jwt(test_jwt_token, mock_settings)

        assert payload["sub"] == str(test_user["_id"])
        assert payload["email"] == test_user["email"]
        assert payload["type"] == "local"

    def test_validate_local_jwt_expired_token(
        self, mock_settings: Settings, expired_jwt_token: str
    ) -> None:
        """
        Test that expired token raises JWTError.

        Per Agent Action Plan section 0.3, tokens must expire after 24 hours.
        Verify that expired tokens are properly rejected.
        """
        with pytest.raises(JWTError):
            validate_local_jwt(expired_jwt_token, mock_settings)

    def test_validate_local_jwt_invalid_signature(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test that tampered token raises JWTError.

        Verify that a token signed with a different key is rejected.
        """
        # Create token with different key
        wrong_settings = Settings(
            app_env="testing",
            debug=True,
            secret_key="different-secret-key-32-characters",
            jwt_expiration_hours=24,
        )

        user_id = str(test_user["_id"])
        token_with_wrong_key = create_local_jwt(user_id, test_user["email"], wrong_settings)

        # Validation with correct settings should fail
        with pytest.raises(JWTError):
            validate_local_jwt(token_with_wrong_key, mock_settings)

    def test_validate_local_jwt_missing_fields(self, mock_settings: Settings) -> None:
        """
        Test that incomplete payload validates but may have missing fields.

        Note: JWT validation doesn't enforce required fields beyond signature
        and expiration - the caller must check for required claims.
        """
        # Create token with minimal payload (only sub, missing other fields)
        now = datetime.now(UTC)
        expire = now + timedelta(hours=24)

        minimal_payload = {
            "sub": "test-user-id",
            "exp": expire,
            "iat": now,
        }

        token = jwt.encode(minimal_payload, mock_settings.secret_key, algorithm="HS256")

        # Token validates but email field is missing
        payload = validate_local_jwt(token, mock_settings)
        assert "sub" in payload
        assert "email" not in payload

    def test_validate_local_jwt_malformed_token(self, mock_settings: Settings) -> None:
        """Test that malformed token string raises JWTError."""
        malformed_tokens = [
            "not-a-jwt",
            "only.two.parts.here.extra",
            "",
            "a.b.c",  # Invalid base64
        ]

        for malformed_token in malformed_tokens:
            with pytest.raises(JWTError):
                validate_local_jwt(malformed_token, mock_settings)


# =============================================================================
# Test Auth0 JWT Validation (Mocked)
# =============================================================================


class TestAuth0JWTValidation:
    """Test suite for Auth0TokenValidator class."""

    def test_auth0_token_validator_initialization(self, mock_settings_with_auth0: Settings) -> None:
        """Test Auth0TokenValidator initializes correctly with settings."""
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        assert validator.settings == mock_settings_with_auth0
        assert validator._jwks_cache is None
        assert validator._jwks_cache_time is None

    def test_auth0_jwks_url_construction(self, mock_settings_with_auth0: Settings) -> None:
        """Test JWKS URL is correctly constructed from Auth0 domain."""
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        expected_url = "https://test-tenant.auth0.com/.well-known/jwks.json"
        assert validator._get_jwks_url() == expected_url

    @pytest.mark.asyncio
    async def test_auth0_token_validation_invalid_signature(
        self, mock_settings_with_auth0: Settings, mock_auth0_jwks: dict
    ) -> None:
        """
        Test that invalid Auth0 token raises HTTPException(401).

        Mock JWKS endpoint and verify signature validation fails
        for tokens signed with incorrect key.
        """
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        # Create a fake token that won't validate
        fake_token = jwt.encode(
            {"sub": "test", "exp": datetime.now(UTC) + timedelta(hours=1)},
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "test-kid-123"},
        )

        with patch.object(validator, "_fetch_jwks", return_value=mock_auth0_jwks):
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_auth0_token(fake_token)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth0_token_missing_kid(
        self, mock_settings_with_auth0: Settings, mock_auth0_jwks: dict
    ) -> None:
        """
        Test that token without kid raises HTTPException(401).

        Per Auth0 JWT spec, tokens must have key ID in header
        to identify which key was used for signing.
        """
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        # Create token without kid header
        token_without_kid = jwt.encode(
            {"sub": "test", "exp": datetime.now(UTC) + timedelta(hours=1)},
            "fake-secret",
            algorithm="HS256",
            # No headers means no kid
        )

        with patch.object(validator, "_fetch_jwks", return_value=mock_auth0_jwks):
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_auth0_token(token_without_kid)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth0_jwks_fetch_failure(self, mock_settings_with_auth0: Settings) -> None:
        """
        Test graceful handling when Auth0 JWKS endpoint unavailable.

        Verify that network errors fetching JWKS result in
        HTTPException(503) indicating service unavailable.
        """
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        # Create a token to validate (will fail at JWKS fetch)
        token = jwt.encode(
            {"sub": "test", "exp": datetime.now(UTC) + timedelta(hours=1)},
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "test-kid-123"},
        )

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_auth0_token(token)

            assert exc_info.value.status_code == 503

    def test_auth0_jwks_caching(
        self, mock_settings_with_auth0: Settings, mock_auth0_jwks: dict
    ) -> None:
        """
        Test JWKS caching behavior.

        Verify that JWKS is cached and reused within cache duration
        to reduce HTTP requests to Auth0.
        """
        validator = Auth0TokenValidator(mock_settings_with_auth0)

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_auth0_jwks
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # First call should fetch JWKS
            jwks1 = validator._fetch_jwks()
            assert mock_get.call_count == 1

            # Second call should use cache
            jwks2 = validator._fetch_jwks()
            assert mock_get.call_count == 1  # No additional call

            # Both should return same data
            assert jwks1 == jwks2


# =============================================================================
# Test Unified Authentication Function
# =============================================================================


class TestAuthenticateToken:
    """Test suite for authenticate_token dependency."""

    @pytest.mark.asyncio
    async def test_authenticate_token_local_jwt(
        self, mock_settings: Settings, test_jwt_token: str, test_user: dict[str, Any]
    ) -> None:
        """
        Test authentication with local JWT when Auth0 disabled.

        Per Agent Action Plan section 0.3, system should automatically
        fall back to local JWT when Auth0 is not configured.
        """
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=test_jwt_token)

        with patch("app.core.auth.get_settings", return_value=mock_settings):
            payload = await authenticate_token(credentials, mock_settings)

        assert payload["sub"] == str(test_user["_id"])
        assert payload["email"] == test_user["email"]
        assert payload["type"] == "local"

    @pytest.mark.asyncio
    async def test_authenticate_token_auth0_jwt_fallback(
        self, mock_settings_with_auth0: Settings
    ) -> None:
        """
        Test authentication with Auth0 JWT when Auth0 enabled (mocked).

        Verify that when Auth0 is enabled, the authenticate_token
        function routes to Auth0 validation path.
        """
        # Create a fake Auth0-style token
        token = jwt.encode(
            {"sub": "auth0|12345", "exp": datetime.now(UTC) + timedelta(hours=1)},
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "test-kid"},
        )

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        # Mock the Auth0 validator to return a payload
        mock_payload = {"sub": "auth0|12345", "email": "auth0@example.com"}

        with (
            patch("app.core.auth.get_settings", return_value=mock_settings_with_auth0),
            patch.object(
                Auth0TokenValidator,
                "validate_auth0_token",
                new_callable=AsyncMock,
                return_value=mock_payload,
            ),
        ):
            payload = await authenticate_token(credentials, mock_settings_with_auth0)

        assert payload["sub"] == "auth0|12345"

    @pytest.mark.asyncio
    async def test_authenticate_token_invalid_token_raises_401(
        self, mock_settings: Settings
    ) -> None:
        """
        Test that invalid token raises HTTPException(401).

        Verify proper error handling for malformed tokens.
        """
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")

        with patch("app.core.auth.get_settings", return_value=mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await authenticate_token(credentials, mock_settings)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticate_token_expired_token_raises_401(
        self, mock_settings: Settings, expired_jwt_token: str
    ) -> None:
        """
        Test that expired token raises HTTPException(401).

        Per Agent Action Plan section 0.3, expired tokens must be rejected.
        """
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=expired_jwt_token)

        with patch("app.core.auth.get_settings", return_value=mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await authenticate_token(credentials, mock_settings)

            assert exc_info.value.status_code == 401


# =============================================================================
# Test get_current_user Dependency
# =============================================================================


class TestGetCurrentUser:
    """Test suite for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_success(
        self, mock_db: AsyncMock, mock_redis: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test successful user lookup from database.

        Verify that get_current_user retrieves user document
        from MongoDB when not cached in Redis.
        """
        token_data = {
            "sub": str(test_user["_id"]),
            "email": test_user["email"],
        }

        # Configure mock DB to return test user
        mock_db.get_users_collection().find_one = AsyncMock(return_value=test_user.copy())

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
        ):
            user = await get_current_user(token_data)

        assert user["email"] == test_user["email"]
        assert user["_id"] == str(test_user["_id"])

    @pytest.mark.asyncio
    async def test_get_current_user_cached(
        self, mock_redis: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test Redis cache is checked first with 5-minute TTL.

        Per Agent Action Plan section 0.8, user data should be
        cached in Redis with 5-minute TTL for performance.
        """
        token_data = {
            "sub": str(test_user["_id"]),
            "email": test_user["email"],
        }

        # Prepare cached user (with string _id as it would be stored)
        cached_user = test_user.copy()
        cached_user["_id"] = str(cached_user["_id"])

        # Configure Redis to return cached user
        mock_redis.get_json = AsyncMock(return_value=cached_user)

        mock_db = AsyncMock()

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
        ):
            user = await get_current_user(token_data)

        # Verify Redis was checked
        mock_redis.get_json.assert_called_once()

        # Verify database was NOT queried (cache hit)
        mock_db.get_users_collection.assert_not_called()

        assert user["email"] == test_user["email"]

    @pytest.mark.asyncio
    async def test_get_current_user_not_found(
        self, mock_db: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """
        Test HTTPException(404) when user doesn't exist.

        Verify proper error when user ID from token
        doesn't match any user in database.
        """
        token_data = {
            "sub": str(ObjectId()),  # Non-existent user ID
            "email": "nonexistent@example.com",
        }

        # Configure mock DB to return None (user not found)
        mock_db.get_users_collection().find_one = AsyncMock(return_value=None)

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token_data)

        assert exc_info.value.status_code == 404
        assert "User not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_cache_population(
        self, mock_db: AsyncMock, mock_redis: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test cache is populated after DB query.

        Verify that after fetching user from database,
        the result is cached in Redis with proper TTL.
        """
        token_data = {
            "sub": str(test_user["_id"]),
            "email": test_user["email"],
        }

        # Configure mock DB to return test user
        mock_db.get_users_collection().find_one = AsyncMock(return_value=test_user.copy())

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
        ):
            await get_current_user(token_data)

        # Verify Redis set_json was called to cache the user
        mock_redis.set_json.assert_called_once()

        # Verify TTL matches USER_CACHE_TTL (5 minutes = 300 seconds)
        call_args = mock_redis.set_json.call_args
        assert call_args[1]["ttl"] == USER_CACHE_TTL
        assert USER_CACHE_TTL == 300  # Verify constant is 5 minutes

    @pytest.mark.asyncio
    async def test_get_current_user_missing_sub_claim(self, mock_redis: AsyncMock) -> None:
        """
        Test HTTPException(401) when token missing 'sub' claim.

        Verify that tokens without user identifier are rejected.
        """
        token_data = {
            "email": "test@example.com",
            # Missing 'sub' claim
        }

        mock_db = AsyncMock()

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token_data)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_redis_failure_continues(
        self, mock_db: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test authentication works even when Redis down (no caching).

        Verify that Redis connection failures don't block authentication;
        the system should gracefully fall back to direct DB queries.
        """
        token_data = {
            "sub": str(test_user["_id"]),
            "email": test_user["email"],
        }

        # Configure Redis to raise exception
        mock_redis = AsyncMock()
        mock_redis.get_json = AsyncMock(side_effect=Exception("Redis connection failed"))
        mock_redis.set_json = AsyncMock(side_effect=Exception("Redis connection failed"))

        # Configure mock DB to return test user
        mock_db.get_users_collection().find_one = AsyncMock(return_value=test_user.copy())

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
        ):
            # Should still succeed despite Redis failure
            user = await get_current_user(token_data)

        assert user["email"] == test_user["email"]


# =============================================================================
# Test Session Management
# =============================================================================


class TestSessionManagement:
    """Test suite for session management functions."""

    @pytest.mark.asyncio
    async def test_create_user_session(
        self, mock_redis: AsyncMock, mock_settings: Settings
    ) -> None:
        """
        Test session stored in Redis with TTL.

        Per Agent Action Plan section 0.8, sessions should be
        stored in Redis with TTL matching JWT expiration.
        """
        user_id = "test-user-id"
        token = "test-jwt-token"

        with (
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
            patch("app.core.auth.get_settings", return_value=mock_settings),
        ):
            result = await create_user_session(user_id, token)

        assert result is True

        # Verify Redis set_json was called
        mock_redis.set_json.assert_called_once()

        # Verify session key format
        call_args = mock_redis.set_json.call_args
        session_key = call_args[0][0]
        assert session_key == f"{SESSION_KEY_PREFIX}:{user_id}"

        # Verify TTL matches JWT expiration (24 hours = 86400 seconds)
        assert call_args[1]["ttl"] == mock_settings.jwt_expiration_hours * 3600

    @pytest.mark.asyncio
    async def test_revoke_user_session(self, mock_redis: AsyncMock) -> None:
        """
        Test session deleted from Redis on logout.

        Per Agent Action Plan section 0.8, revoke_user_session
        should delete session and user cache from Redis.
        """
        user_id = "test-user-id"

        with patch("app.core.auth.get_redis_client", return_value=mock_redis):
            result = await revoke_user_session(user_id)

        assert result is True

        # Verify both session and user cache are deleted
        assert mock_redis.delete.call_count == 2

        # Check the keys that were deleted
        deleted_keys = [call[0][0] for call in mock_redis.delete.call_args_list]
        assert f"{SESSION_KEY_PREFIX}:{user_id}" in deleted_keys
        assert f"{USER_KEY_PREFIX}:{user_id}" in deleted_keys

    @pytest.mark.asyncio
    async def test_verify_session_exists(self, mock_redis: AsyncMock) -> None:
        """
        Test valid session returns True.

        Verify that verify_session correctly identifies
        when a user has an active session.
        """
        user_id = "test-user-id"

        # Configure Redis to indicate session exists
        mock_redis.exists = AsyncMock(return_value=True)

        with patch("app.core.auth.get_redis_client", return_value=mock_redis):
            result = await verify_session(user_id)

        assert result is True
        mock_redis.exists.assert_called_once_with(f"{SESSION_KEY_PREFIX}:{user_id}")

    @pytest.mark.asyncio
    async def test_verify_session_not_exists(self, mock_redis: AsyncMock) -> None:
        """
        Test invalid/expired session returns False.

        Verify that verify_session returns False when
        no active session exists for the user.
        """
        user_id = "test-user-id"

        # Configure Redis to indicate session doesn't exist
        mock_redis.exists = AsyncMock(return_value=False)

        with patch("app.core.auth.get_redis_client", return_value=mock_redis):
            result = await verify_session(user_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_create_user_session_redis_unavailable(self) -> None:
        """
        Test session creation returns False when Redis unavailable.

        Verify graceful handling when Redis connection fails.
        """
        user_id = "test-user-id"
        token = "test-jwt-token"

        with patch("app.core.auth.get_redis_client", return_value=None):
            result = await create_user_session(user_id, token)

        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_user_session_redis_unavailable(self) -> None:
        """
        Test session revocation returns False when Redis unavailable.

        Verify graceful handling when Redis connection fails.
        """
        user_id = "test-user-id"

        with patch("app.core.auth.get_redis_client", return_value=None):
            result = await revoke_user_session(user_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_session_redis_unavailable_returns_true(self) -> None:
        """
        Test session verification returns True when Redis unavailable.

        When Redis is down, allow the request through since
        token validation still occurs (graceful degradation).
        """
        user_id = "test-user-id"

        with patch("app.core.auth.get_redis_client", return_value=None):
            result = await verify_session(user_id)

        # Returns True to allow request when Redis unavailable
        assert result is True


# =============================================================================
# Test Authentication Edge Cases
# =============================================================================


class TestAuthenticationEdgeCases:
    """Test edge cases in authentication handling."""

    def test_authentication_with_whitespace_in_token(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test tokens with leading/trailing whitespace.

        Note: HTTPAuthorizationCredentials typically strips whitespace,
        but we test that our validation handles it gracefully.
        """
        user_id = str(test_user["_id"])
        token = create_local_jwt(user_id, test_user["email"], mock_settings)

        # Token with whitespace should fail validation (not a valid JWT format)
        token_with_whitespace = f"  {token}  "

        with pytest.raises(JWTError):
            validate_local_jwt(token_with_whitespace, mock_settings)

        # Clean token should work
        payload = validate_local_jwt(token.strip(), mock_settings)
        assert payload["sub"] == user_id

    def test_authentication_case_sensitivity(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test case-sensitive token handling.

        Verify that JWT tokens are treated as case-sensitive,
        as base64 encoding is case-sensitive.
        """
        user_id = str(test_user["_id"])
        token = create_local_jwt(user_id, test_user["email"], mock_settings)

        # Original token works
        payload = validate_local_jwt(token, mock_settings)
        assert payload["sub"] == user_id

        # Modified case token should fail
        modified_token = token.swapcase()
        with pytest.raises(JWTError):
            validate_local_jwt(modified_token, mock_settings)

    def test_decode_token_without_verification(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test decoding token header without verification.

        Verify that decode_token_without_verification correctly
        extracts the header (including algorithm and kid).
        """
        # Create token with custom header
        user_id = str(test_user["_id"])
        now = datetime.now(UTC)
        expire = now + timedelta(hours=24)

        payload = {
            "sub": user_id,
            "email": test_user["email"],
            "exp": expire,
            "iat": now,
        }

        token = jwt.encode(
            payload,
            mock_settings.secret_key,
            algorithm="HS256",
            headers={"kid": "test-kid-header"},
        )

        header = decode_token_without_verification(token)

        assert header["alg"] == "HS256"
        assert header["kid"] == "test-kid-header"

    def test_decode_token_without_verification_invalid_format(self) -> None:
        """
        Test decode_token_without_verification with invalid token format.

        Verify ValueError is raised for malformed tokens.
        """
        invalid_tokens = [
            "not.a.jwt.token.here",  # Too many parts
            "only-one-part",  # Too few parts
            "",  # Empty
        ]

        for invalid_token in invalid_tokens:
            with pytest.raises(ValueError, match="Invalid token format"):
                decode_token_without_verification(invalid_token)


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_failure(
        self, mock_redis: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test appropriate error when database unavailable.

        Verify HTTPException(503) when MongoDB connection fails
        during user lookup.
        """
        token_data = {
            "sub": str(test_user["_id"]),
            "email": test_user["email"],
        }

        # Configure mock DB to raise RuntimeError
        mock_db = AsyncMock()
        mock_db.get_users_collection = Mock(side_effect=RuntimeError("Database unavailable"))

        with (
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token_data)

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_session_creation_exception_handling(self, mock_settings: Settings) -> None:
        """
        Test graceful handling of session creation exceptions.

        Verify that exceptions during session creation return False
        rather than propagating the error.
        """
        user_id = "test-user-id"
        token = "test-jwt-token"

        mock_redis = AsyncMock()
        mock_redis.set_json = AsyncMock(side_effect=Exception("Unexpected error"))

        with (
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
            patch("app.core.auth.get_settings", return_value=mock_settings),
        ):
            result = await create_user_session(user_id, token)

        # Should return False on error, not raise exception
        assert result is False


# =============================================================================
# Test Authentication Fallback
# =============================================================================


class TestAuthenticationFallback:
    """Test automatic fallback from Auth0 to local JWT."""

    def test_is_auth0_enabled_when_configured(self, mock_settings_with_auth0: Settings) -> None:
        """
        Test is_auth0_enabled returns True when Auth0 fully configured.

        Per Agent Action Plan section 0.3, Auth0 requires:
        - auth0_domain
        - auth0_client_id
        - auth0_client_secret
        """
        assert mock_settings_with_auth0.is_auth0_enabled is True

    def test_is_auth0_enabled_when_not_configured(self, mock_settings: Settings) -> None:
        """
        Test is_auth0_enabled returns False when Auth0 not configured.

        When Auth0 settings are None, system should use local JWT.
        """
        assert mock_settings.is_auth0_enabled is False

    def test_is_auth0_enabled_partial_config(self) -> None:
        """
        Test is_auth0_enabled returns False with partial configuration.

        All three Auth0 settings must be present for Auth0 to be enabled.
        """
        partial_settings = Settings(
            app_env="testing",
            debug=True,
            secret_key="test-secret-key-for-jwt-minimum-32-chars",
            auth0_domain="test-tenant.auth0.com",
            auth0_client_id="test-client-id",
            auth0_client_secret=None,  # Missing secret
        )

        assert partial_settings.is_auth0_enabled is False


# =============================================================================
# Test User Authentication Function
# =============================================================================


class TestAuthenticateUser:
    """Test authenticate_user function for local auth."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success(
        self, mock_db: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test successful user authentication with correct credentials.
        """
        # Set up user with hashed password
        password = "test-password-123"
        user_with_password = test_user.copy()
        user_with_password["hashed_password"] = sha256(password.encode()).hexdigest()

        mock_db.get_users_collection().find_one = AsyncMock(return_value=user_with_password)

        with patch("app.core.auth.get_db_client", return_value=mock_db):
            user = await authenticate_user(test_user["email"], password)

        assert user is not None
        assert user["email"] == test_user["email"]

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(
        self, mock_db: AsyncMock, test_user: dict[str, Any]
    ) -> None:
        """
        Test authentication fails with incorrect password.
        """
        # Set up user with hashed password
        user_with_password = test_user.copy()
        user_with_password["hashed_password"] = sha256(b"correct-password").hexdigest()

        mock_db.get_users_collection().find_one = AsyncMock(return_value=user_with_password)

        with patch("app.core.auth.get_db_client", return_value=mock_db):
            user = await authenticate_user(test_user["email"], "wrong-password")

        assert user is None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, mock_db: AsyncMock) -> None:
        """
        Test authentication fails when user doesn't exist.
        """
        mock_db.get_users_collection().find_one = AsyncMock(return_value=None)

        with patch("app.core.auth.get_db_client", return_value=mock_db):
            user = await authenticate_user("nonexistent@example.com", "any-password")

        assert user is None


# =============================================================================
# Test create_access_token
# =============================================================================


class TestCreateAccessToken:
    """Test create_access_token convenience function."""

    def test_create_access_token_without_settings(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test create_access_token gets settings automatically.
        """
        user_id = str(test_user["_id"])
        email = test_user["email"]

        with patch("app.core.auth.get_settings", return_value=mock_settings):
            token = create_access_token(user_id, email)

        # Verify token is valid
        payload = jwt.decode(token, mock_settings.secret_key, algorithms=["HS256"])
        assert payload["sub"] == user_id
        assert payload["email"] == email

    def test_create_access_token_with_settings(
        self, mock_settings: Settings, test_user: dict[str, Any]
    ) -> None:
        """
        Test create_access_token with explicit settings.
        """
        user_id = str(test_user["_id"])
        email = test_user["email"]

        token = create_access_token(user_id, email, mock_settings)

        payload = jwt.decode(token, mock_settings.secret_key, algorithms=["HS256"])
        assert payload["sub"] == user_id
        assert payload["email"] == email


# =============================================================================
# Test get_current_user_optional
# =============================================================================


class TestGetCurrentUserOptional:
    """Test get_current_user_optional dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_optional_no_credentials(self, mock_settings: Settings) -> None:
        """
        Test returns None when no credentials provided.
        """
        with patch("app.core.auth.get_settings", return_value=mock_settings):
            result = await get_current_user_optional(None, mock_settings)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_optional_invalid_token(self, mock_settings: Settings) -> None:
        """
        Test returns None for invalid tokens (no exception).
        """
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")

        with patch("app.core.auth.get_settings", return_value=mock_settings):
            result = await get_current_user_optional(credentials, mock_settings)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_optional_valid_token(
        self,
        mock_settings: Settings,
        test_jwt_token: str,
        test_user: dict[str, Any],
        mock_redis: AsyncMock,
        mock_db: AsyncMock,
    ) -> None:
        """
        Test returns user when valid credentials provided.
        """
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=test_jwt_token)

        # Configure mocks
        mock_db.get_users_collection().find_one = AsyncMock(return_value=test_user.copy())

        with (
            patch("app.core.auth.get_settings", return_value=mock_settings),
            patch("app.core.auth.get_db_client", return_value=mock_db),
            patch("app.core.auth.get_redis_client", return_value=mock_redis),
        ):
            result = await get_current_user_optional(credentials, mock_settings)

        assert result is not None
        assert result["email"] == test_user["email"]


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Test authentication constants are correctly defined."""

    def test_user_cache_ttl(self) -> None:
        """
        Test USER_CACHE_TTL is 5 minutes (300 seconds).

        Per Agent Action Plan section 0.8.
        """
        assert USER_CACHE_TTL == 300

    def test_session_key_prefix(self) -> None:
        """Test SESSION_KEY_PREFIX is 'session'."""
        assert SESSION_KEY_PREFIX == "session"

    def test_user_key_prefix(self) -> None:
        """Test USER_KEY_PREFIX is 'user'."""
        assert USER_KEY_PREFIX == "user"
