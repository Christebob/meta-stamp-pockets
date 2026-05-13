"""
User Pydantic models for META-STAMP V3.

This module defines the User model for authentication and profile management.
Supports both Auth0-based authentication and local JWT fallback strategy.

Per Agent Action Plan section 0.6 transformation mapping for backend/app/models/user.py
and section 0.3 Auth0 with JWT-based authentication requirements.
"""

import re

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# =============================================================================
# CONSTANTS
# =============================================================================

# Username validation pattern (alphanumeric, underscore, hyphen)
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")

# Password validation constants
MIN_PASSWORD_LENGTH = 8

# Supported platforms for AI Touch Value™ calculations
SUPPORTED_PLATFORMS: list[str] = [
    "youtube",
    "vimeo",
    "tiktok",
    "instagram",
    "twitter",
    "twitch",
    "spotify",
    "soundcloud",
    "deviantart",
    "behance",
    "dribbble",
    "other",
]


# =============================================================================
# MODELS
# =============================================================================


class User(BaseModel):
    """
    Pydantic model for user authentication and profile management.

    Stores user authentication data including Auth0 integration (auth0_id),
    email, profile information, account timestamps, and platform metrics
    for AI Touch Value™ calculations.

    Supports both Auth0-based authentication (via auth0_id) and local JWT
    fallback authentication (via hashed_password).

    Attributes:
        id: MongoDB ObjectId as string (aliased from _id)
        email: Validated email address for account identification
        auth0_id: Auth0 user identifier for external authentication
        username: Optional display name (alphanumeric, underscore, hyphen)
        hashed_password: Bcrypt hashed password for local auth fallback
        is_active: Account status flag (default True)
        is_verified: Email verification status (default False)
        full_name: User's full display name
        profile_image_url: Avatar/profile picture URL
        bio: User description/biography
        followers_count: Social media followers for AI Touch Value™
        content_hours: Total content hours created
        total_views: Total views across platforms
        primary_platform: Main creator platform
        created_at: Account registration timestamp (UTC)
        updated_at: Last profile modification timestamp (UTC)
        last_login: Last successful login timestamp

    Example:
        ```python
        user = User(
            email="creator@example.com",
            auth0_id="auth0|123456",
            username="creative_artist",
            full_name="Creative Artist",
            is_active=True,
            is_verified=True,
            followers_count=10000,
            content_hours=150.5,
            total_views=500000,
            primary_platform="youtube"
        )
        ```
    """

    # MongoDB ObjectId field with alias for _id
    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")

    # Identity fields
    email: EmailStr = Field(..., description="Validated email address for account identification")

    auth0_id: str | None = Field(
        default=None,
        max_length=255,
        description="Auth0 user identifier for external authentication",
    )

    username: str | None = Field(
        default=None,
        min_length=3,
        max_length=50,
        description="Display username (alphanumeric, underscore, hyphen)",
    )

    # Authentication fields
    hashed_password: str | None = Field(
        default=None, description="Bcrypt hashed password for local authentication fallback"
    )

    is_active: bool = Field(
        default=True, description="Account status flag (inactive accounts cannot login)"
    )

    is_verified: bool = Field(default=False, description="Email verification status")

    # Profile information
    full_name: str | None = Field(
        default=None, max_length=100, description="User's full display name"
    )

    profile_image_url: str | None = Field(
        default=None, max_length=2048, description="URL to user's avatar/profile picture"
    )

    bio: str | None = Field(default=None, max_length=500, description="User biography/description")

    # Platform metrics for AI Touch Value™ calculations
    followers_count: int = Field(
        default=0, ge=0, description="Total followers/subscribers across platforms"
    )

    content_hours: float = Field(default=0.0, ge=0.0, description="Total hours of content created")

    total_views: int = Field(default=0, ge=0, description="Total content views across platforms")

    primary_platform: str | None = Field(
        default=None, max_length=50, description="Primary creator platform (youtube, vimeo, etc.)"
    )

    # Metadata storage
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional user metadata and settings"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Account registration timestamp (UTC)",
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last profile modification timestamp (UTC)",
    )

    last_login: datetime | None = Field(default=None, description="Last successful login timestamp")

    # MongoDB configuration
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            "example": {
                "email": "creator@example.com",
                "auth0_id": "auth0|123456789",
                "username": "creative_artist",
                "is_active": True,
                "is_verified": True,
                "full_name": "Creative Artist",
                "followers_count": 10000,
                "content_hours": 150.5,
                "total_views": 500000,
                "primary_platform": "youtube",
                "created_at": "2024-01-15T10:30:00Z",
            }
        },
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str | None) -> str | None:
        """
        Validate username format (alphanumeric, underscore, hyphen).

        Username must be 3-50 characters and contain only:
        - Letters (a-z, A-Z)
        - Numbers (0-9)
        - Underscores (_)
        - Hyphens (-)

        Args:
            v: Username string or None

        Returns:
            Validated username or None

        Raises:
            ValueError: If username format is invalid
        """
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        if not USERNAME_PATTERN.match(v):
            raise ValueError(
                "Username must be 3-50 characters and contain only "
                "letters, numbers, underscores, and hyphens"
            )

        return v

    @field_validator("primary_platform")
    @classmethod
    def validate_primary_platform(cls, v: str | None) -> str | None:
        """
        Validate primary platform against supported platforms.

        Args:
            v: Platform name or None

        Returns:
            Lowercase platform name or None

        Raises:
            ValueError: If platform is not supported
        """
        if v is None:
            return v

        v = v.strip().lower()
        if not v:
            return None

        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform '{v}'. "
                f"Supported platforms: {', '.join(SUPPORTED_PLATFORMS)}"
            )

        return v

    @field_validator("profile_image_url")
    @classmethod
    def validate_profile_image_url(cls, v: str | None) -> str | None:
        """
        Validate profile image URL format.

        Args:
            v: URL string or None

        Returns:
            Validated URL or None

        Raises:
            ValueError: If URL is malformed
        """
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        # Basic URL validation
        if not (v.startswith(("http://", "https://"))):
            raise ValueError("Profile image URL must start with http:// or https://")

        return v

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def is_auth0_user(self) -> bool:
        """
        Check if user uses Auth0 authentication.

        Returns:
            True if auth0_id is set
        """
        return self.auth0_id is not None

    def is_local_user(self) -> bool:
        """
        Check if user uses local JWT authentication.

        Returns:
            True if hashed_password is set
        """
        return self.hashed_password is not None

    def can_login(self) -> bool:
        """
        Check if user can login (active and has auth method).

        Returns:
            True if user is active and has authentication configured
        """
        return self.is_active and (self.is_auth0_user() or self.is_local_user())

    def update_last_login(self) -> None:
        """
        Update the last_login timestamp to current UTC time.
        """
        self.last_login = datetime.now(UTC)

    def update_modified(self) -> None:
        """
        Update the updated_at timestamp to current UTC time.
        """
        self.updated_at = datetime.now(UTC)

    def get_display_name(self) -> str:
        """
        Get the best available display name for the user.

        Priority: full_name > username > email username portion

        Returns:
            Display name string
        """
        if self.full_name:
            return self.full_name
        if self.username:
            return self.username
        # Extract username portion from email
        return self.email.split("@")[0]

    def update_platform_metrics(
        self,
        followers_count: int | None = None,
        content_hours: float | None = None,
        total_views: int | None = None,
    ) -> None:
        """
        Update platform metrics for AI Touch Value™ calculations.

        Args:
            followers_count: New follower count (optional)
            content_hours: New content hours (optional)
            total_views: New total views (optional)
        """
        if followers_count is not None:
            self.followers_count = max(0, followers_count)
        if content_hours is not None:
            self.content_hours = max(0.0, content_hours)
        if total_views is not None:
            self.total_views = max(0, total_views)
        self.update_modified()

    def activate(self) -> None:
        """
        Activate the user account.
        """
        self.is_active = True
        self.update_modified()

    def deactivate(self) -> None:
        """
        Deactivate the user account.
        """
        self.is_active = False
        self.update_modified()

    def verify_email(self) -> None:
        """
        Mark email as verified.
        """
        self.is_verified = True
        self.update_modified()


class UserCreate(BaseModel):
    """
    Schema for creating a new user (registration request).

    Used for API request validation during user registration.
    Password is plaintext here and should be hashed before storage.
    """

    email: EmailStr = Field(..., description="User email address")

    password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="Plaintext password (min 8 chars) for local auth",
    )

    username: str | None = Field(
        default=None, min_length=3, max_length=50, description="Display username"
    )

    full_name: str | None = Field(default=None, max_length=100, description="User's full name")

    auth0_id: str | None = Field(default=None, description="Auth0 ID for SSO registration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "newuser@example.com",
                "password": "securepassword123",
                "username": "newuser",
                "full_name": "New User",
            }
        }
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str | None) -> str | None:
        """
        Validate password strength.

        Requirements:
        - Minimum 8 characters
        - At least one letter
        - At least one number

        Args:
            v: Password string or None

        Returns:
            Validated password

        Raises:
            ValueError: If password is too weak
        """
        if v is None:
            return v

        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")

        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")

        return v


class UserLogin(BaseModel):
    """
    Schema for user login request.
    """

    email: EmailStr = Field(..., description="User email address")

    password: str = Field(..., min_length=1, description="User password")

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "user@example.com", "password": "mypassword123"}}
    )


class UserResponse(BaseModel):
    """
    Schema for user API responses (excludes sensitive data).

    Safe for returning to clients - excludes hashed_password and
    other sensitive internal fields.
    """

    id: str | None = Field(None, description="User ID")
    email: EmailStr = Field(..., description="User email")
    username: str | None = Field(None, description="Display username")
    full_name: str | None = Field(None, description="Full name")
    profile_image_url: str | None = Field(None, description="Avatar URL")
    bio: str | None = Field(None, description="User bio")
    followers_count: int = Field(default=0, description="Follower count")
    content_hours: float = Field(default=0.0, description="Content hours")
    total_views: int = Field(default=0, description="Total views")
    primary_platform: str | None = Field(None, description="Primary platform")
    is_active: bool = Field(default=True, description="Account active")
    is_verified: bool = Field(default=False, description="Email verified")
    created_at: datetime = Field(..., description="Registration timestamp")
    last_login: datetime | None = Field(None, description="Last login")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    @classmethod
    def from_user(cls, user: User) -> "UserResponse":
        """
        Create response from User model (excludes sensitive data).

        Args:
            user: User model instance

        Returns:
            UserResponse for API (no password or internal fields)
        """
        return cls(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            profile_image_url=user.profile_image_url,
            bio=user.bio,
            followers_count=user.followers_count,
            content_hours=user.content_hours,
            total_views=user.total_views,
            primary_platform=user.primary_platform,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login=user.last_login,
        )


class UserUpdate(BaseModel):
    """
    Schema for updating user profile.

    All fields are optional - only provided fields will be updated.
    """

    username: str | None = Field(
        default=None, min_length=3, max_length=50, description="New username"
    )

    full_name: str | None = Field(default=None, max_length=100, description="New full name")

    profile_image_url: str | None = Field(
        default=None, max_length=2048, description="New avatar URL"
    )

    bio: str | None = Field(default=None, max_length=500, description="New bio")

    primary_platform: str | None = Field(
        default=None, max_length=50, description="Primary platform"
    )

    followers_count: int | None = Field(default=None, ge=0, description="Updated follower count")

    content_hours: float | None = Field(default=None, ge=0.0, description="Updated content hours")

    total_views: int | None = Field(default=None, ge=0, description="Updated total views")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Updated Name",
                "bio": "New bio description",
                "followers_count": 15000,
            }
        }
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str | None) -> str | None:
        """Validate username format."""
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        if not USERNAME_PATTERN.match(v):
            raise ValueError(
                "Username must be 3-50 characters and contain only "
                "letters, numbers, underscores, and hyphens"
            )

        return v

    @field_validator("primary_platform")
    @classmethod
    def validate_primary_platform(cls, v: str | None) -> str | None:
        """Validate platform against supported list."""
        if v is None:
            return v

        v = v.strip().lower()
        if not v:
            return None

        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform '{v}'. Supported: {', '.join(SUPPORTED_PLATFORMS)}"
            )

        return v


class TokenResponse(BaseModel):
    """
    Schema for JWT token response after authentication.
    """

    access_token: str = Field(..., description="JWT access token")

    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")

    expires_in: int = Field(
        default=86400, description="Token expiration time in seconds"  # 24 hours in seconds
    )

    user: UserResponse = Field(..., description="Authenticated user information")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {"id": "user123", "email": "user@example.com", "username": "creator"},
            }
        }
    )
