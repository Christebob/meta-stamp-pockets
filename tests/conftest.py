"""
Pytest Configuration and Test Fixtures for META-STAMP V3 Backend

This module provides comprehensive test fixtures including:
- Test MongoDB and Redis instances with proper cleanup
- Mocked S3/MinIO storage client for upload testing
- FastAPI TestClient with sync and async support
- Mocked Auth0 tokens and authentication fixtures
- Sample test data (users, assets, fingerprints)
- Async event loop configuration for pytest-asyncio
- Reusable mock objects for all backend services

Based on Agent Action Plan sections 0.5, 0.6, 0.7, and 0.10 requirements.
"""

import asyncio

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from jose import jwt
from PIL import Image

from app.config import Settings
from app.core.database import DatabaseClient
from app.core.redis_client import RedisClient
from app.core.storage import StorageClient
from app.main import app


# ==============================================================================
# Pytest Plugins and Configuration
# ==============================================================================

pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers for test categorization.

    Markers defined:
    - asyncio: For async test functions
    - integration: For integration tests requiring external services
    - unit: For unit tests (isolated, no external dependencies)
    - slow: For slow-running tests that may be skipped in quick test runs
    """
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")


# ==============================================================================
# Event Loop Fixture (Session-Scoped)
# ==============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create a session-scoped event loop for async test execution.

    This fixture creates a new event loop for the entire test session,
    ensuring all async fixtures and test cases run properly with pytest-asyncio.
    The event loop is closed after the session completes.

    Yields:
        asyncio.AbstractEventLoop: The event loop for async test execution
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# Settings Fixtures
# ==============================================================================


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create a Settings instance with test-specific configuration values.

    Provides isolated test environment configuration including:
    - Test MongoDB URI pointing to test_metastamp database
    - Redis URL using DB 1 instead of production DB 0
    - Test S3/MinIO credentials and bucket
    - Disabled Auth0 for local JWT testing
    - Standard upload limits and equity factor

    Returns:
        Settings: Configured Settings instance for testing
    """
    return Settings(
        app_env="testing",
        app_name="META-STAMP-V3-Test",
        debug=True,
        secret_key="test-secret-key-for-jwt-signing-minimum-32-chars",
        mongodb_uri="mongodb://localhost:27017/test_metastamp",
        mongodb_db_name="test_metastamp",
        mongodb_min_pool_size=1,
        mongodb_max_pool_size=10,
        redis_url="localhost:6379/1",
        redis_cache_ttl_seconds=300,
        s3_endpoint_url="http://localhost:9000",
        s3_access_key_id="test-access-key",
        s3_secret_access_key="test-secret-key",
        s3_bucket_name="test-bucket",
        s3_region="us-east-1",
        auth0_domain=None,  # Use local JWT for testing
        auth0_client_id=None,
        auth0_client_secret=None,
        auth0_api_audience=None,
        jwt_algorithm="HS256",
        jwt_expiration_hours=24,
        max_upload_size_mb=500,
        direct_upload_threshold_mb=10,
        presigned_url_expiration_seconds=900,
        equity_factor=0.25,
        allowed_file_extensions=[
            ".txt",
            ".md",
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".mp3",
            ".wav",
            ".aac",
            ".mp4",
            ".mov",
            ".avi",
        ],
    )


@pytest.fixture
def mock_settings_with_auth0() -> Settings:
    """
    Create a Settings instance with Auth0 enabled for testing Auth0 integration.

    Returns:
        Settings: Configured Settings instance with Auth0 credentials
    """
    return Settings(
        app_env="testing",
        app_name="META-STAMP-V3-Test",
        debug=True,
        secret_key="test-secret-key-for-jwt-signing-minimum-32-chars",
        mongodb_uri="mongodb://localhost:27017/test_metastamp",
        mongodb_db_name="test_metastamp",
        redis_url="localhost:6379/1",
        s3_endpoint_url="http://localhost:9000",
        s3_access_key_id="test-access-key",
        s3_secret_access_key="test-secret-key",
        s3_bucket_name="test-bucket",
        auth0_domain="test-tenant.auth0.com",
        auth0_client_id="test-client-id",
        auth0_client_secret="test-client-secret",
        auth0_api_audience="https://api.metastamp.test",
        jwt_algorithm="HS256",
        jwt_expiration_hours=24,
        max_upload_size_mb=500,
        direct_upload_threshold_mb=10,
        equity_factor=0.25,
    )


# ==============================================================================
# MongoDB Fixtures
# ==============================================================================


@pytest.fixture
async def test_db_client(mock_settings: Settings) -> AsyncGenerator[DatabaseClient, None]:
    """
    Create a real DatabaseClient connected to a test MongoDB instance.

    This fixture initializes a MongoDB connection with test settings,
    provides access to test collections, and performs cleanup by
    dropping the test database after tests complete.

    Note: Requires a running MongoDB instance for integration tests.

    Args:
        mock_settings: Test configuration settings

    Yields:
        DatabaseClient: Connected database client for testing
    """
    client = DatabaseClient(mock_settings)
    try:
        await client.connect()
        yield client
    finally:
        # Cleanup: drop test database
        if client._client is not None:
            await client._client.drop_database(mock_settings.mongodb_database_name)
        await client.close()


@pytest.fixture
def mock_db() -> AsyncMock:
    """
    Create a mocked DatabaseClient for unit testing without MongoDB.

    Provides AsyncMock objects for all collection accessor methods,
    allowing tests to verify database operations without requiring
    an actual MongoDB connection.

    Returns:
        AsyncMock: Mocked DatabaseClient with collection accessors
    """
    mock = AsyncMock(spec=DatabaseClient)

    # Create mock collections with common MongoDB operations
    mock_assets_collection = AsyncMock()
    mock_assets_collection.find_one = AsyncMock(return_value=None)
    mock_assets_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="test-asset-id")
    )
    mock_assets_collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    mock_assets_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    mock_assets_collection.find = MagicMock(return_value=AsyncMock())
    mock_assets_collection.count_documents = AsyncMock(return_value=0)

    mock_users_collection = AsyncMock()
    mock_users_collection.find_one = AsyncMock(return_value=None)
    mock_users_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-user-id"))
    mock_users_collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))

    mock_fingerprints_collection = AsyncMock()
    mock_fingerprints_collection.find_one = AsyncMock(return_value=None)
    mock_fingerprints_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="test-fingerprint-id")
    )
    mock_fingerprints_collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))

    mock_wallet_collection = AsyncMock()
    mock_wallet_collection.find_one = AsyncMock(return_value=None)
    mock_wallet_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="test-wallet-id")
    )
    mock_wallet_collection.find = MagicMock(return_value=AsyncMock())

    mock_analytics_collection = AsyncMock()
    mock_analytics_collection.find_one = AsyncMock(return_value=None)
    mock_analytics_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="test-analytics-id")
    )
    mock_analytics_collection.find = MagicMock(return_value=AsyncMock())

    # Assign mock collections to accessor methods
    mock.get_assets_collection.return_value = mock_assets_collection
    mock.get_users_collection.return_value = mock_users_collection
    mock.get_fingerprints_collection.return_value = mock_fingerprints_collection
    mock.get_wallet_collection.return_value = mock_wallet_collection
    mock.get_analytics_collection.return_value = mock_analytics_collection

    # Mock database getter
    mock_database = MagicMock()
    mock_database.client = MagicMock()
    mock.get_database.return_value = mock_database

    # Mock connection methods
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.is_connected = True

    return mock


# ==============================================================================
# Redis Fixtures
# ==============================================================================


@pytest.fixture
async def test_redis_client(mock_settings: Settings) -> AsyncGenerator[RedisClient, None]:
    """
    Create a real RedisClient connected to a test Redis instance.

    This fixture initializes a Redis connection with test settings (DB 1),
    provides caching operations for tests, and performs cleanup by
    flushing the test database after tests complete.

    Note: Requires a running Redis instance for integration tests.

    Args:
        mock_settings: Test configuration settings

    Yields:
        RedisClient: Connected Redis client for testing
    """
    client = RedisClient(mock_settings)
    try:
        await client.connect()
        yield client
    finally:
        # Cleanup: flush test Redis DB
        if client._client is not None:
            await client._client.flushdb()
        await client.close()


@pytest.fixture
def mock_redis() -> AsyncMock:
    """
    Create a mocked RedisClient for unit testing without Redis.

    Provides AsyncMock objects for all Redis operations including
    get, set, delete, exists, and JSON operations. Allows tests to
    verify caching behavior without requiring an actual Redis connection.

    Returns:
        AsyncMock: Mocked RedisClient with all operations
    """
    mock = AsyncMock(spec=RedisClient)

    # Mock basic operations
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.exists = AsyncMock(return_value=False)
    mock.expire = AsyncMock(return_value=True)

    # Mock JSON operations
    mock.get_json = AsyncMock(return_value=None)
    mock.set_json = AsyncMock(return_value=True)

    # Mock hash operations
    mock.hget = AsyncMock(return_value=None)
    mock.hset = AsyncMock(return_value=True)
    mock.hgetall = AsyncMock(return_value={})

    # Mock connection state
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.is_connected = True

    # Mock underlying client for direct access
    mock_client = AsyncMock()
    mock_client.flushdb = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=True)
    mock.client = mock_client
    mock._client = mock_client

    return mock


# ==============================================================================
# S3/Storage Fixtures
# ==============================================================================


@pytest.fixture
def mock_storage() -> Mock:
    """
    Create a mocked StorageClient for testing without S3/MinIO.

    Provides Mock objects for all S3-compatible operations including
    presigned URL generation, multipart uploads, file operations,
    and existence checks. Uses spec=StorageClient to ensure the mock
    has the same interface as the real StorageClient.

    Returns:
        Mock: Mocked StorageClient with all S3 operations
    """
    mock = Mock(spec=StorageClient)

    # Mock presigned URL generation
    mock.generate_presigned_upload_url = Mock(
        return_value={
            "url": "https://s3.example.com/test-bucket/assets/test-file.png?X-Amz-Signature=test",
            "key": "assets/test-file.png",
            "expires_in": 900,
        }
    )
    mock.generate_presigned_download_url = Mock(return_value="https://s3.example.com/download-url")

    # Mock multipart upload operations
    mock.initiate_multipart_upload = Mock(return_value="test-upload-id-12345")
    mock.generate_presigned_part_url = Mock(return_value="https://s3.example.com/part-upload-url")
    mock.complete_multipart_upload = Mock(return_value=True)
    mock.abort_multipart_upload = Mock(return_value=True)

    # Mock file operations
    mock.upload_file = Mock(return_value=True)
    mock.upload_fileobj = Mock(return_value=True)
    mock.download_file = Mock(return_value=True)
    mock.download_fileobj = Mock(return_value=BytesIO(b"test file content"))
    mock.delete_file = Mock(return_value=True)

    # Mock file metadata and existence
    mock.file_exists = Mock(return_value=True)
    mock.get_file_metadata = Mock(
        return_value={
            "ContentLength": 1024,
            "ContentType": "image/png",
            "LastModified": datetime.now(UTC),
            "ETag": '"abc123def456"',
        }
    )

    # Mock bucket operations
    mock.ensure_bucket_exists = Mock(return_value=True)

    return mock


# ==============================================================================
# Authentication Fixtures
# ==============================================================================


@pytest.fixture
def test_user() -> dict[str, Any]:
    """
    Create sample test user data for authentication testing.

    Returns:
        Dict[str, Any]: User data dictionary with standard fields
    """
    return {
        "_id": "test-user-id-12345",
        "email": "test@example.com",
        "name": "Test User",
        "auth0_id": None,  # Local authentication
        "created_at": datetime.now(UTC),
        "last_login": datetime.now(UTC),
        "is_active": True,
        "preferences": {
            "theme": "light",
            "notifications": True,
        },
    }


@pytest.fixture
def test_user_with_auth0() -> dict[str, Any]:
    """
    Create sample test user data with Auth0 ID for Auth0 integration testing.

    Returns:
        Dict[str, Any]: User data dictionary with Auth0 ID
    """
    return {
        "_id": "test-user-id-auth0-12345",
        "email": "auth0user@example.com",
        "name": "Auth0 Test User",
        "auth0_id": "auth0|123456789",
        "created_at": datetime.now(UTC),
        "last_login": datetime.now(UTC),
        "is_active": True,
    }


@pytest.fixture
def test_jwt_token(mock_settings: Settings, test_user: dict[str, Any]) -> str:
    """
    Create a valid local JWT token for authentication testing.

    Generates a JWT token with HS256 algorithm, 24-hour expiration,
    and payload containing user ID, email, and token type.

    Args:
        mock_settings: Test configuration with secret key
        test_user: Test user data for token payload

    Returns:
        str: Encoded JWT token string
    """
    payload = {
        "sub": test_user["_id"],
        "email": test_user["email"],
        "name": test_user.get("name", "Test User"),
        "exp": datetime.now(UTC) + timedelta(hours=mock_settings.jwt_expiration_hours),
        "iat": datetime.now(UTC),
        "type": "local",
    }
    return jwt.encode(payload, mock_settings.secret_key, algorithm=mock_settings.jwt_algorithm)


@pytest.fixture
def test_expired_jwt_token(mock_settings: Settings, test_user: dict[str, Any]) -> str:
    """
    Create an expired JWT token for testing expiration handling.

    Args:
        mock_settings: Test configuration with secret key
        test_user: Test user data for token payload

    Returns:
        str: Encoded expired JWT token string
    """
    payload = {
        "sub": test_user["_id"],
        "email": test_user["email"],
        "exp": datetime.now(UTC) - timedelta(hours=1),  # Expired 1 hour ago
        "iat": datetime.now(UTC) - timedelta(hours=25),  # Issued 25 hours ago
        "type": "local",
    }
    return jwt.encode(payload, mock_settings.secret_key, algorithm=mock_settings.jwt_algorithm)


@pytest.fixture
def mock_auth(test_user: dict[str, Any], test_jwt_token: str) -> dict[str, Any]:
    """
    Create a complete mock authentication context for testing.

    Provides user data, valid JWT token, and pre-configured
    Authorization headers for use with test clients.

    Args:
        test_user: Test user data
        test_jwt_token: Valid JWT token for the user

    Returns:
        Dict[str, Any]: Authentication context with user, token, and headers
    """
    return {
        "user": test_user,
        "token": test_jwt_token,
        "headers": {"Authorization": f"Bearer {test_jwt_token}"},
    }


@pytest.fixture
def mock_auth0_jwks() -> dict[str, Any]:
    """
    Create mock Auth0 JWKS response for testing RS256 validation.

    Returns:
        Dict[str, Any]: Mock JWKS structure with RSA public key components
    """
    return {
        "keys": [
            {
                "kid": "test-key-id-12345",
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "n": "test-modulus-n-value-base64url-encoded",
                "e": "AQAB",  # Standard RSA public exponent (65537)
            }
        ]
    }


# ==============================================================================
# FastAPI Test Client Fixtures
# ==============================================================================


@pytest.fixture
def test_client() -> TestClient:
    """
    Create a synchronous FastAPI TestClient for endpoint testing.

    The TestClient allows making HTTP requests to the FastAPI application
    without running an actual server. Supports all HTTP methods and
    includes automatic cookie handling.

    Returns:
        TestClient: Configured test client for the FastAPI app
    """
    return TestClient(app)


@pytest.fixture
async def async_test_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Create an asynchronous HTTP client for async endpoint testing.

    Uses httpx.AsyncClient with ASGI transport to make async requests
    to the FastAPI application. Essential for testing streaming
    responses and async endpoints like the AI assistant.

    Yields:
        AsyncClient: Async HTTP client for the FastAPI app
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ==============================================================================
# Sample Test Data Fixtures - Files
# ==============================================================================


@pytest.fixture
def test_image() -> BytesIO:
    """
    Create a synthetic test image for upload and fingerprinting tests.

    Generates a 100x100 RGB image with a red color fill,
    saved as PNG format in a BytesIO buffer.

    Returns:
        BytesIO: In-memory buffer containing PNG image data
    """
    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "test_image.png"
    return buf


@pytest.fixture
def test_image_large() -> BytesIO:
    """
    Create a larger synthetic test image for testing size limits.

    Generates a 1000x1000 RGB image to test upload handling
    for larger files while still being under direct upload threshold.

    Returns:
        BytesIO: In-memory buffer containing larger PNG image data
    """
    img = Image.new("RGB", (1000, 1000), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "test_image_large.png"
    return buf


@pytest.fixture
def test_image_jpg() -> BytesIO:
    """
    Create a test JPEG image for format-specific testing.

    Returns:
        BytesIO: In-memory buffer containing JPEG image data
    """
    img = Image.new("RGB", (100, 100), color="green")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    buf.name = "test_image.jpg"
    return buf


@pytest.fixture
def test_text_file() -> BytesIO:
    """
    Create a test text file for text upload testing.

    Returns:
        BytesIO: In-memory buffer containing text data
    """
    content = b"This is a sample text file for testing.\nIt has multiple lines.\nMETA-STAMP V3 Test Content."
    buf = BytesIO(content)
    buf.name = "test_document.txt"
    return buf


@pytest.fixture
def test_pdf_content() -> bytes:
    """
    Create minimal PDF content for PDF upload testing.

    Returns:
        bytes: Minimal valid PDF file bytes
    """
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
193
%%EOF"""


@pytest.fixture
def test_audio_file_path() -> str:
    """
    Return the path to a sample audio file for testing.

    Note: Tests should create this fixture file or mock audio processing.

    Returns:
        str: Relative path to test audio fixture
    """
    return "tests/fixtures/sample.mp3"


@pytest.fixture
def test_video_file_path() -> str:
    """
    Return the path to a sample video file for testing.

    Note: Tests should create this fixture file or mock video processing.

    Returns:
        str: Relative path to test video fixture
    """
    return "tests/fixtures/sample.mp4"


# ==============================================================================
# Sample Test Data Fixtures - Database Records
# ==============================================================================


@pytest.fixture
def test_asset() -> dict[str, Any]:
    """
    Create sample asset data for database and API testing.

    Returns:
        Dict[str, Any]: Asset data dictionary matching MongoDB schema
    """
    return {
        "_id": "test-asset-id-12345",
        "user_id": "test-user-id-12345",
        "file_name": "test-file.png",
        "original_file_name": "my_original_image.png",
        "file_type": "image",
        "file_extension": ".png",
        "mime_type": "image/png",
        "file_size": 1024,
        "s3_key": "assets/test-user-id-12345/test-file.png",
        "upload_status": "ready",
        "upload_method": "direct",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "fingerprint_id": None,
        "metadata": {
            "width": 100,
            "height": 100,
            "format": "PNG",
        },
    }


@pytest.fixture
def test_asset_processing() -> dict[str, Any]:
    """
    Create sample asset data with 'processing' status.

    Returns:
        Dict[str, Any]: Asset data with processing status
    """
    return {
        "_id": "test-asset-processing-12345",
        "user_id": "test-user-id-12345",
        "file_name": "processing-file.png",
        "file_type": "image",
        "file_size": 2048,
        "s3_key": "assets/test-user-id-12345/processing-file.png",
        "upload_status": "processing",
        "created_at": datetime.now(UTC),
        "fingerprint_id": None,
    }


@pytest.fixture
def test_fingerprint() -> dict[str, Any]:
    """
    Create sample fingerprint data for fingerprinting tests.

    Returns:
        Dict[str, Any]: Fingerprint data dictionary matching MongoDB schema
    """
    return {
        "_id": "test-fingerprint-id-12345",
        "asset_id": "test-asset-id-12345",
        "perceptual_hashes": {
            "phash": "abcd1234efgh5678",
            "ahash": "1234abcd5678efgh",
            "dhash": "5678efgh1234abcd",
        },
        "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5] * 256,  # 1280-dim embedding
        "spectral_data": None,
        "video_hashes": None,
        "metadata": {
            "width": 100,
            "height": 100,
            "format": "PNG",
            "mode": "RGB",
        },
        "processing_status": "completed",
        "created_at": datetime.now(UTC),
        "processing_time_ms": 150,
    }


@pytest.fixture
def test_wallet() -> dict[str, Any]:
    """
    Create sample wallet data for wallet/earnings tests.

    Returns:
        Dict[str, Any]: Wallet data dictionary
    """
    return {
        "_id": "test-wallet-id-12345",
        "user_id": "test-user-id-12345",
        "balance": 1250.50,
        "currency": "USD",
        "pending_earnings": 350.25,
        "total_earned": 1600.75,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


@pytest.fixture
def test_transaction() -> dict[str, Any]:
    """
    Create sample transaction data for wallet history tests.

    Returns:
        Dict[str, Any]: Transaction data dictionary
    """
    return {
        "_id": "test-transaction-id-12345",
        "user_id": "test-user-id-12345",
        "wallet_id": "test-wallet-id-12345",
        "amount": 100.00,
        "currency": "USD",
        "transaction_type": "earning",
        "status": "completed",
        "description": "AI Touch Value™ compensation",
        "asset_id": "test-asset-id-12345",
        "created_at": datetime.now(UTC),
    }


@pytest.fixture
def test_analytics_calculation() -> dict[str, Any]:
    """
    Create sample AI Touch Value™ calculation data.

    Returns:
        Dict[str, Any]: Analytics calculation data dictionary
    """
    return {
        "_id": "test-analytics-id-12345",
        "user_id": "test-user-id-12345",
        "asset_id": "test-asset-id-12345",
        "model_earnings": 10000.00,
        "training_contribution_score": 75.0,
        "usage_exposure_score": 80.0,
        "equity_factor": 0.25,
        "calculated_value": 1500.00,  # 10000 * 0.75 * 0.80 * 0.25 = 1500
        "formula_breakdown": {
            "model_earnings": 10000.00,
            "contribution_factor": 0.75,
            "exposure_factor": 0.80,
            "equity_factor": 0.25,
            "result": 1500.00,
        },
        "created_at": datetime.now(UTC),
    }


# ==============================================================================
# LangChain and AI Assistant Mocks
# ==============================================================================


@pytest.fixture
def mock_langchain() -> AsyncMock:
    """
    Create a mocked LangChain chat model for AI assistant testing.

    Provides AsyncMock for invoke, stream, and bind_tools methods,
    enabling testing of AI assistant functionality without API calls.

    Returns:
        AsyncMock: Mocked LangChain chat model
    """
    mock = AsyncMock()

    # Mock invoke method for single responses
    mock.invoke = AsyncMock(
        return_value=MagicMock(
            content="Test response from AI assistant. I can help you understand your creative assets.",
            additional_kwargs={},
        )
    )

    # Mock stream method for streaming responses
    async def mock_stream(*args, **kwargs):
        chunks = [
            "Test ",
            "streaming ",
            "response ",
            "from ",
            "AI ",
            "assistant.",
        ]
        for chunk in chunks:
            yield MagicMock(content=chunk)

    mock.stream = mock_stream
    mock.astream = mock_stream

    # Mock bind_tools for tool calling
    mock_with_tools = AsyncMock()
    mock_with_tools.invoke = mock.invoke
    mock_with_tools.stream = mock_stream
    mock_with_tools.astream = mock_stream
    mock.bind_tools = Mock(return_value=mock_with_tools)

    return mock


@pytest.fixture
def mock_conversation_context() -> list[dict[str, str]]:
    """
    Create sample conversation context for multi-turn testing.

    Returns:
        List[Dict[str, str]]: List of conversation messages
    """
    return [
        {"role": "user", "content": "What is AI Touch Value?"},
        {
            "role": "assistant",
            "content": "AI Touch Value™ is a calculation that estimates the compensation owed to creators based on their content's potential contribution to AI model training.",
        },
        {"role": "user", "content": "How is it calculated?"},
        {
            "role": "assistant",
            "content": "The formula is: AI Touch Value™ = ModelEarnings x (TrainingContributionScore/100) x (UsageExposureScore/100) x EquityFactor (25%)",
        },
    ]


# ==============================================================================
# Background Task and Utility Mocks
# ==============================================================================


@pytest.fixture
def mock_background_tasks() -> Mock:
    """
    Create a mocked FastAPI BackgroundTasks for testing async operations.

    Returns:
        Mock: Mocked BackgroundTasks with add_task method
    """
    mock = Mock()
    mock.add_task = Mock()
    return mock


@pytest.fixture
def mock_datetime() -> Mock:
    """
    Create a mocked datetime module for deterministic timestamp testing.

    Returns:
        Mock: Mocked datetime with fixed utcnow
    """
    fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    mock = Mock()
    mock.utcnow = Mock(return_value=fixed_time)
    mock.now = Mock(return_value=fixed_time)
    return mock


@pytest.fixture
def mock_youtube() -> Mock:
    """
    Create a mocked YouTube transcript API for URL processing tests.

    Returns:
        Mock: Mocked YouTubeTranscriptApi with get_transcript method
    """
    mock = Mock()
    mock.get_transcript = Mock(
        return_value=[
            {"text": "Welcome to this video about creative rights.", "start": 0.0, "duration": 3.5},
            {
                "text": "Today we'll discuss AI and creator compensation.",
                "start": 3.5,
                "duration": 4.0,
            },
            {"text": "META-STAMP helps protect your content.", "start": 7.5, "duration": 3.0},
        ]
    )
    mock.list_transcripts = Mock(return_value=MagicMock())
    return mock


@pytest.fixture
def mock_requests() -> Mock:
    """
    Create a mocked requests module for URL content fetching tests.

    Returns:
        Mock: Mocked requests with get method returning HTML response
    """
    mock = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"""
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Test Content</h1>
        <p>This is sample webpage content for testing URL processing.</p>
        <p>META-STAMP V3 extracts this text for fingerprinting.</p>
    </body>
    </html>
    """
    mock_response.text = mock_response.content.decode("utf-8")
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.raise_for_status = Mock()

    mock.get = Mock(return_value=mock_response)
    return mock


@pytest.fixture
def mock_vimeo() -> Mock:
    """
    Create a mocked Vimeo API response for Vimeo URL testing.

    Returns:
        Mock: Mocked Vimeo metadata response
    """
    return {
        "video_id": "123456789",
        "title": "Test Vimeo Video",
        "description": "A test video for META-STAMP integration testing",
        "duration": 300,  # 5 minutes
        "width": 1920,
        "height": 1080,
        "created_time": "2024-01-15T12:00:00Z",
        "user": {
            "name": "Test Creator",
            "uri": "/users/12345",
        },
    }


# ==============================================================================
# Test Data Cleanup Fixtures
# ==============================================================================


@pytest.fixture(autouse=False)
async def cleanup_test_collections(test_db_client: DatabaseClient) -> AsyncGenerator[None, None]:
    """
    Clean up test database collections after test execution.

    This fixture is NOT autouse - it must be explicitly requested by tests
    that need database cleanup. It clears all documents from test collections
    after each test to ensure isolation.

    Args:
        test_db_client: Connected database client

    Yields:
        None: Yields control to test, then performs cleanup
    """
    yield

    # Cleanup after test
    if test_db_client and test_db_client.is_connected:
        collection_names = [
            "assets",
            "users",
            "fingerprints",
            "wallet",
            "analytics",
            "transactions",
        ]
        for collection_name in collection_names:
            try:
                collection = test_db_client.get_database()[collection_name]
                await collection.delete_many({})
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture(autouse=False)
async def cleanup_test_redis(test_redis_client: RedisClient) -> AsyncGenerator[None, None]:
    """
    Clean up test Redis data after test execution.

    This fixture is NOT autouse - it must be explicitly requested by tests
    that need Redis cleanup. It flushes the test Redis database after
    each test to ensure isolation.

    Args:
        test_redis_client: Connected Redis client

    Yields:
        None: Yields control to test, then performs cleanup
    """
    yield

    # Cleanup after test
    if test_redis_client and test_redis_client.is_connected:
        try:
            await test_redis_client._client.flushdb()
        except Exception:
            pass  # Ignore cleanup errors


# ==============================================================================
# Dependency Override Fixtures
# ==============================================================================


@pytest.fixture
def override_get_db(mock_db: AsyncMock):
    """
    Create a dependency override function for database injection.

    Used with app.dependency_overrides to inject mock_db into endpoints.

    Args:
        mock_db: Mocked database client

    Returns:
        Callable: Override function returning mock_db
    """

    async def _override_get_db():
        return mock_db

    return _override_get_db


@pytest.fixture
def override_get_redis(mock_redis: AsyncMock):
    """
    Create a dependency override function for Redis injection.

    Used with app.dependency_overrides to inject mock_redis into endpoints.

    Args:
        mock_redis: Mocked Redis client

    Returns:
        Callable: Override function returning mock_redis
    """

    async def _override_get_redis():
        return mock_redis

    return _override_get_redis


@pytest.fixture
def override_get_storage(mock_storage: Mock):
    """
    Create a dependency override function for storage injection.

    Used with app.dependency_overrides to inject mock_storage into endpoints.

    Args:
        mock_storage: Mocked storage client

    Returns:
        Callable: Override function returning mock_storage
    """

    def _override_get_storage():
        return mock_storage

    return _override_get_storage


# ==============================================================================
# URL Test Data Fixtures
# ==============================================================================


@pytest.fixture
def test_youtube_url() -> str:
    """
    Return a sample YouTube URL for URL processing tests.

    Returns:
        str: YouTube video URL
    """
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def test_vimeo_url() -> str:
    """
    Return a sample Vimeo URL for URL processing tests.

    Returns:
        str: Vimeo video URL
    """
    return "https://vimeo.com/123456789"


@pytest.fixture
def test_webpage_url() -> str:
    """
    Return a sample webpage URL for URL processing tests.

    Returns:
        str: Generic webpage URL
    """
    return "https://example.com/article/about-creative-rights"


@pytest.fixture
def test_invalid_urls() -> list[str]:
    """
    Return a list of invalid URLs for validation testing.

    Returns:
        List[str]: List of malformed or dangerous URLs
    """
    return [
        "not-a-url",
        "ftp://invalid-protocol.com/file.txt",
        "javascript:alert('xss')",
        "file:///etc/passwd",
        "https://example.com/malicious.exe",
        "https://example.com/archive.zip",
        "",
        "   ",
    ]


# ==============================================================================
# File Validation Test Data
# ==============================================================================


@pytest.fixture
def allowed_file_extensions() -> list[str]:
    """
    Return list of allowed file extensions for validation testing.

    Returns:
        List[str]: Allowed extensions per security requirements
    """
    return [
        ".txt",
        ".md",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp3",
        ".wav",
        ".aac",
        ".mp4",
        ".mov",
        ".avi",
    ]


@pytest.fixture
def dangerous_file_extensions() -> list[str]:
    """
    Return list of dangerous file extensions that must be rejected.

    Returns:
        List[str]: Dangerous extensions per security requirements
    """
    return [
        ".zip",
        ".rar",
        ".7z",
        ".exe",
        ".bin",
        ".sh",
        ".app",
        ".msi",
        ".dmg",
        ".iso",
        ".bat",
        ".cmd",
        ".ps1",
    ]
