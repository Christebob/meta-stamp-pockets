"""
META-STAMP V3 Configuration Management Module

This module provides comprehensive configuration management for the META-STAMP V3 platform
using Pydantic Settings. It loads and validates all environment variables required for:
- Application settings (name, environment, debug mode, logging)
- MongoDB database connection and pooling
- Redis caching and session management
- S3/MinIO object storage with presigned URL support
- Auth0 authentication with local JWT fallback
- LangChain multi-provider AI integration (OpenAI, Anthropic, Google)
- File upload validation and limits
- Fingerprinting service configuration
- AI Touch Value™ calculation parameters

All settings support environment variable overrides and .env file loading with
comprehensive validation and type safety.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """
    Comprehensive configuration settings for META-STAMP V3 platform.

    This class uses Pydantic Settings to load configuration from environment
    variables and .env files with full type validation. Required fields
    (marked with `...`) must be provided via environment variables.

    Configuration Categories:
    - Application: Core app settings like name, environment, debug mode
    - MongoDB: Database connection URI and connection pool settings
    - Redis: Cache connection URL and TTL settings
    - S3/MinIO: Object storage credentials and bucket configuration
    - Auth0: Authentication provider settings with JWT fallback support
    - LangChain: Multi-provider AI configuration (OpenAI, Anthropic, Google)
    - Upload: File size limits and allowed extensions
    - Fingerprinting: Processing settings for asset fingerprinting
    - AI Touch Value: Calculation parameters for creator compensation

    Example usage:
        ```python
        from app.config import Settings

        settings = Settings()
        print(f"Connecting to MongoDB at: {settings.mongodb_uri}")
        print(f"Auth0 enabled: {settings.is_auth0_enabled}")
        ```
    """

    # =========================================================================
    # Application Settings
    # =========================================================================

    app_name: str = Field(
        default="META-STAMP-V3",
        description="Application name displayed in API documentation and logs",
    )

    app_env: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )

    debug: bool = Field(
        default=True, description="Enable debug mode with hot-reload and verbose logging"
    )

    log_level: str = Field(
        default="info", description="Logging level (debug, info, warning, error, critical)"
    )

    secret_key: str = Field(
        default="development-secret-key-change-in-production-32chars",
        description="Secret key for JWT signing and encryption. Must be a secure random string.",
        min_length=32,
    )

    host: str = Field(default="0.0.0.0", description="Host address for the API server to bind to")

    port: int = Field(default=8000, description="Port number for the API server", ge=1, le=65535)

    cors_origins: Annotated[list[str], NoDecode] = Field(
        default=[
            "http://localhost:3000",
            "https://metastampv3-production.up.railway.app",
            "https://dependable-education-production-05f1.up.railway.app",
        ],
        description="List of allowed CORS origins for frontend access",
    )

    # =========================================================================
    # MongoDB Configuration
    # =========================================================================

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI — reads MONGODB_URI or MONGO_URL",
    )

    mongodb_db_name: str = Field(
        default="metastamp", description="MongoDB database name for META-STAMP data storage"
    )

    mongodb_min_pool_size: int = Field(
        default=10, description="Minimum number of connections in MongoDB connection pool", ge=1
    )

    mongodb_max_pool_size: int = Field(
        default=100, description="Maximum number of connections in MongoDB connection pool", ge=10
    )

    # =========================================================================
    # Redis Configuration
    # =========================================================================

    redis_url: str = Field(
        default="localhost:6379",
        description="Redis connection host:port or URL.",
    )

    @field_validator("redis_url", mode="before")
    @classmethod
    def resolve_redis_url(cls, v: str) -> str:
        """Ensure Redis URL has correct scheme; fix Railway-injected URLs."""
        import os
        url = v or os.environ.get("REDIS_URL", "localhost:6379")
        redis_scheme = "redis" + "://"
        if url and not url.startswith((redis_scheme, "rediss://", "unix://")):
            url = redis_scheme + url
        return url

    @field_validator("mongodb_uri", mode="before")
    @classmethod
    def resolve_mongodb_uri(cls, v: str) -> str:
        """Accept MONGO_URL as fallback for MONGODB_URI (Railway uses MONGO_URL)."""
        import os
        return v if v and v != "mongodb://localhost:27017" else (
            os.environ.get("MONGO_URL") or v or "mongodb://localhost:27017"
        )

    redis_cache_ttl_seconds: int = Field(
        default=300, description="Default TTL for Redis cache entries in seconds (5 minutes)", ge=1
    )

    # =========================================================================
    # S3/MinIO Storage Configuration
    # =========================================================================

    s3_endpoint_url: str | None = Field(
        default=None, description="S3-compatible endpoint URL for MinIO (None for AWS S3)"
    )

    s3_access_key_id: str = Field(
        default="minioadmin",
        description="S3/MinIO access key ID for authentication",
    )

    s3_secret_access_key: str = Field(
        default="minioadmin",
        description="S3/MinIO secret access key for authentication",
    )

    s3_bucket_name: str = Field(
        default="metastamp-assets", description="S3 bucket name for storing uploaded assets"
    )

    s3_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 bucket (also used for MinIO compatibility)",
    )

    presigned_url_expiration_seconds: int = Field(
        default=900,
        description="Expiration time for S3 presigned URLs in seconds (15 minutes)",
        ge=60,
        le=3600,
    )

    # =========================================================================
    # Auth0 Configuration
    # =========================================================================

    auth0_domain: str | None = Field(
        default=None, description="Auth0 tenant domain (e.g., your-tenant.auth0.com)"
    )

    auth0_api_audience: str | None = Field(
        default=None, description="Auth0 API audience identifier for token validation"
    )

    auth0_client_id: str | None = Field(default=None, description="Auth0 application client ID")

    auth0_client_secret: str | None = Field(
        default=None, description="Auth0 application client secret"
    )

    jwt_algorithm: str = Field(
        default="HS256", description="JWT signing algorithm (HS256 for local, RS256 for Auth0)"
    )

    jwt_expiration_hours: int = Field(
        default=24, description="JWT token expiration time in hours", ge=1, le=168
    )

    # =========================================================================
    # LangChain AI Providers Configuration
    # =========================================================================

    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key for GPT-4/5 model access"
    )

    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key for Claude model access"
    )

    google_api_key: str | None = Field(
        default=None, description="Google API key for Gemini model access"
    )

    default_ai_provider: str = Field(
        default="openai", description="Default AI provider (openai, anthropic, google)"
    )

    default_ai_model: str = Field(
        default="gpt-4", description="Default AI model name for the selected provider"
    )

    # =========================================================================
    # YouTube Data API Configuration
    # =========================================================================

    youtube_api_key: str | None = Field(
        default=None, description="YouTube Data API v3 key for public data access (search, video info)"
    )

    google_oauth_client_id: str | None = Field(
        default=None, description="Google OAuth 2.0 Client ID for YouTube authenticated operations"
    )

    google_oauth_client_secret: str | None = Field(
        default=None, description="Google OAuth 2.0 Client Secret for YouTube authenticated operations"
    )

    google_oauth_redirect_uri: str = Field(
        default="http://localhost:3000/auth/google/callback",
        description="Google OAuth redirect URI for authentication callback",
    )

    # =========================================================================
    # File Upload Settings
    # =========================================================================

    max_upload_size_mb: int = Field(
        default=500,
        description="Maximum file upload size in megabytes (500MB hard limit)",
        ge=1,
        le=500,
    )

    direct_upload_threshold_mb: int = Field(
        default=10,
        description="File size threshold for direct upload vs presigned URL (10MB)",
        ge=1,
        le=100,
    )

    allowed_file_extensions: Annotated[list[str], NoDecode] = Field(
        default=[
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
        description="List of allowed file extensions for upload validation",
    )

    # =========================================================================
    # Fingerprinting Settings
    # =========================================================================

    enable_fingerprinting: bool = Field(
        default=True, description="Enable asset fingerprinting for uploaded content"
    )

    fingerprint_cache_ttl_seconds: int = Field(
        default=3600, description="TTL for cached fingerprint data in seconds (1 hour)", ge=60
    )

    # =========================================================================
    # MCP Server Configuration (Pockets)
    # =========================================================================

    mcp_enabled: bool = Field(
        default=True, description="Enable the MCP server for AI agent content access"
    )

    mcp_rate_limit_per_minute: int = Field(
        default=100, description="Default rate limit per agent per minute", ge=1, le=10000
    )

    mcp_default_price_per_pull: float = Field(
        default=0.0025, description="Default price per content pull in USD", ge=0.0, le=100.0
    )

    mcp_pocket_cache_ttl_seconds: int = Field(
        default=3600, description="TTL for pocket snapshot cache in Redis (1 hour)", ge=60
    )

    mcp_terms_version: str = Field(
        default="1.0.0", description="Current MCP terms of service version"
    )

    # =========================================================================
    # Stripe Connect Configuration
    # =========================================================================

    stripe_secret_key: str = Field(
        default="",
        description="Stripe secret key (use sk_test_... for test mode)",
    )

    stripe_publishable_key: str = Field(
        default="",
        description="Stripe publishable key (use pk_test_... for test mode)",
    )

    stripe_webhook_secret: str = Field(
        default="",
        description="Stripe webhook signing secret for verifying webhook events",
    )

    stripe_connect_client_id: str = Field(
        default="",
        description="Stripe Connect OAuth client ID (ca_... from Connect settings)",
    )

    stripe_payout_enabled: bool = Field(
        default=True,
        description="Enable automatic Stripe payouts when wallet balance is credited",
    )

    stripe_minimum_payout_cents: int = Field(
        default=100,
        description="Minimum payout amount in cents before triggering a Stripe transfer",
        ge=50,
    )

    stripe_platform_fee_percent: float = Field(
        default=15.0,
        description="Platform fee percentage taken from each payout (Meta-Stamp cut). Creator receives (100 - this)%.",
        ge=0.0,
        le=50.0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Demo / Pilot Mode
    # Set DEMO_MODE=false in production to require real Stripe PaymentIntents.
    # When True, the token 'demo-payment-settled' is accepted alongside real pi_ tokens.
    # ─────────────────────────────────────────────────────────────────────────
    demo_mode: bool = Field(
        default=True,
        description=(
            "Allow demo-payment-settled token as proof of payment. "
            "Set DEMO_MODE=false in production to require real Stripe PaymentIntents only."
        ),
    )

    # =========================================================================
    # AI Touch Value™ Settings
    # =========================================================================

    equity_factor: float = Field(
        default=0.25,
        description="Fixed equity factor (25%) for AI Touch Value™ calculation",
        ge=0.0,
        le=1.0,
    )

    # =========================================================================
    # Model Configuration
    # =========================================================================

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log_level is a valid logging level."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        normalized = v.lower()
        if normalized not in valid_levels:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of: {', '.join(valid_levels)}")
        return normalized

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Validate that app_env is a valid environment name."""
        valid_envs = {"development", "staging", "production", "testing"}
        normalized = v.lower()
        if normalized not in valid_envs:
            raise ValueError(f"Invalid app_env '{v}'. Must be one of: {', '.join(valid_envs)}")
        return normalized

    @field_validator("default_ai_provider")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        """Validate that default_ai_provider is a supported provider."""
        valid_providers = {"openai", "anthropic", "google", "local"}
        normalized = v.lower()
        if normalized not in valid_providers:
            raise ValueError(
                f"Invalid default_ai_provider '{v}'. Must be one of: {', '.join(valid_providers)}"
            )
        return normalized

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v: str) -> str:
        """Validate that jwt_algorithm is a supported algorithm."""
        valid_algorithms = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v.upper() not in valid_algorithms:
            raise ValueError(
                f"Invalid jwt_algorithm '{v}'. Must be one of: {', '.join(valid_algorithms)}"
            )
        return v.upper()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string if provided as string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def validate_allowed_extensions(cls, v: str | list[str]) -> list[str]:
        """Parse allowed extensions from comma-separated string if provided as string."""
        if isinstance(v, str):
            extensions = [ext.strip() for ext in v.split(",") if ext.strip()]
            return [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        return v

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def is_auth0_enabled(self) -> bool:
        """
        Check if Auth0 authentication is fully configured.

        Returns True if all required Auth0 settings are provided:
        - auth0_domain
        - auth0_client_id
        - auth0_client_secret

        When False, the application falls back to local JWT authentication
        using HS256 algorithm with the configured secret_key.
        """
        return all([self.auth0_domain, self.auth0_client_id, self.auth0_client_secret])

    @property
    def max_upload_size_bytes(self) -> int:
        """
        Get maximum upload size in bytes.

        Convenience property that converts max_upload_size_mb to bytes
        for use in file validation logic.
        """
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def direct_upload_threshold_bytes(self) -> int:
        """
        Get direct upload threshold in bytes.

        Convenience property that converts direct_upload_threshold_mb to bytes
        for routing upload requests to direct upload vs presigned URL flow.
        """
        return self.direct_upload_threshold_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def has_openai_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0

    @property
    def has_anthropic_configured(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0

    @property
    def has_google_configured(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None and len(self.google_api_key) > 0

    @property
    def has_youtube_configured(self) -> bool:
        """Check if YouTube Data API key is configured."""
        return self.youtube_api_key is not None and len(self.youtube_api_key) > 0

    @property
    def has_youtube_oauth_configured(self) -> bool:
        """Check if YouTube OAuth credentials are fully configured."""
        return all([
            self.google_oauth_client_id,
            self.google_oauth_client_secret,
        ])

    def get_available_ai_providers(self) -> list[str]:
        """
        Get list of configured AI providers.

        Returns a list of provider names that have API keys configured
        and are ready for use with LangChain.
        """
        providers = []
        if self.has_openai_configured:
            providers.append("openai")
        if self.has_anthropic_configured:
            providers.append("anthropic")
        if self.has_google_configured:
            providers.append("google")
        # Local models are always available as fallback
        providers.append("local")
        return providers


@lru_cache
def get_settings() -> Settings:
    """
    Get the global Settings instance.

    This function provides a singleton pattern for the Settings class using
    lru_cache, ensuring that configuration is loaded only once and reused
    throughout the application lifecycle.

    The @lru_cache decorator ensures that the Settings object is created only
    once on first call, and subsequent calls return the cached instance without
    re-reading environment variables or .env files.

    Returns:
        Settings: The global configuration instance.

    Example:
        ```python
        from app.config import get_settings

        settings = get_settings()
        print(f"Running in {settings.app_env} mode")
        ```
    """
    # Configuration is loaded from environment variables via pydantic-settings
    return Settings()
