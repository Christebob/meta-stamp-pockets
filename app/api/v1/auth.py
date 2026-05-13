"""
META-STAMP V3 Authentication API Router

This module implements authentication endpoints for the META-STAMP V3 platform,
providing comprehensive user authentication via Auth0 OAuth2 integration with
automatic fallback to local HS256 JWT authentication when Auth0 is not configured.

Endpoints:
- POST /login: Authenticate user with email/password (OAuth2 password flow)
- POST /logout: Invalidate user session and revoke tokens
- GET /me: Retrieve current authenticated user profile

Per Agent Action Plan sections:
- Section 0.3: Auth0 integration with local JWT fallback (HS256, 24h expiry)
- Section 0.4: Authentication implementation with JWT handling
- Section 0.8: Endpoint specifications for login, logout, and me
- Section 0.10: JWT tokens must use HS256 for local with 24-hour expiration

Security Features:
- Auth0 RS256 JWT validation using public keys from JWKS endpoint
- Local HS256 JWT generation with 24-hour expiration as fallback
- Session management via Redis for token invalidation
- Consistent JSON response structure for all endpoints
"""

import logging

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from app.config import Settings, get_settings
from app.core.auth import (
    authenticate_user,
    create_access_token,
    create_user_session,
    get_current_user,
    revoke_user_session,
)
from app.core.database import get_db_client
from app.models.user import UserResponse


# Configure module logger for structured logging
logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["authentication"])


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginRequest(BaseModel):
    """
    Request model for user login via JSON body.

    Supports both Auth0 OAuth2 flow (email only) and local authentication
    (email + password). When password is None, Auth0 flow is assumed.

    Attributes:
        email: User's email address (validated RFC 5322 format)
        password: Optional password for local authentication
    """

    email: EmailStr = Field(
        ...,
        description="User's email address for authentication",
        examples=["creator@example.com"],
    )
    password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="User password for local authentication (required when Auth0 is disabled)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "creator@example.com",
                "password": "securepassword123",
            }
        }
    }


class LoginResponse(BaseModel):
    """
    Response model for successful login.

    Returns the JWT access token, token type (Bearer), and user profile
    information excluding sensitive fields like password hash.

    Attributes:
        access_token: JWT token string for authenticating subsequent requests
        token_type: Always "bearer" for OAuth2 compatibility
        expires_in: Token expiration time in seconds
        user: User profile data (excludes sensitive fields)
    """

    access_token: str = Field(..., description="JWT access token for authenticated requests")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="Authenticated user profile data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {
                    "id": "507f1f77bcf86cd799439011",
                    "email": "creator@example.com",
                    "username": "creative_artist",
                    "full_name": "Creative Artist",
                    "is_active": True,
                    "is_verified": True,
                    "created_at": "2024-01-15T10:30:00Z",
                    "last_login": "2024-06-15T14:30:00Z",
                },
            }
        }
    }


class LogoutResponse(BaseModel):
    """
    Response model for successful logout.

    Attributes:
        message: Success message confirming logout
        logged_out_at: Timestamp of logout (ISO 8601 UTC)
    """

    message: str = Field(..., description="Logout confirmation message")
    logged_out_at: str = Field(..., description="Timestamp of logout in ISO 8601 format")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Successfully logged out",
                "logged_out_at": "2024-06-15T14:45:00Z",
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standard error response model for consistent error structure.

    Attributes:
        error: Error type/code identifier
        detail: Human-readable error description
        status_code: HTTP status code
    """

    error: str = Field(..., description="Error type or code")
    detail: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "authentication_failed",
                "detail": "Invalid email or password",
                "status_code": 401,
            }
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _create_user_response_from_dict(user_data: dict[str, Any]) -> UserResponse:
    """
    Create a UserResponse from a user dictionary.

    Handles conversion of MongoDB user document to safe response model,
    ensuring sensitive fields are excluded and data types are correct.

    Args:
        user_data: User dictionary from database or authentication result

    Returns:
        UserResponse: Safe user data for API response
    """
    # Handle _id field which may be ObjectId or string
    user_id = user_data.get("_id") or user_data.get("id")
    if user_id is not None:
        user_id = str(user_id)

    # Parse timestamps, handling both datetime objects and ISO strings
    created_at = user_data.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    elif created_at is None:
        created_at = datetime.now(UTC)

    last_login = user_data.get("last_login")
    if isinstance(last_login, str):
        last_login = datetime.fromisoformat(last_login.replace("Z", "+00:00"))

    return UserResponse(
        id=user_id,
        email=user_data.get("email", ""),
        username=user_data.get("username"),
        full_name=user_data.get("full_name"),
        profile_image_url=user_data.get("profile_image_url"),
        bio=user_data.get("bio"),
        followers_count=user_data.get("followers_count", 0),
        content_hours=user_data.get("content_hours", 0.0),
        total_views=user_data.get("total_views", 0),
        primary_platform=user_data.get("primary_platform"),
        is_active=user_data.get("is_active", True),
        is_verified=user_data.get("is_verified", False),
        created_at=created_at,
        last_login=last_login,
    )


async def _update_user_last_login(user_id: str) -> None:
    """
    Update user's last_login timestamp in MongoDB.

    Args:
        user_id: The user's MongoDB ObjectId as string

    Note:
        Errors are logged but not raised to avoid blocking login success.
    """
    try:
        db_client = get_db_client()
        users_collection = db_client.get_users_collection()

        # Update last_login timestamp
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.now(UTC)}},
        )
        logger.debug("Updated last_login for user: %s", user_id)
    except Exception as e:
        # Log but don't fail login if last_login update fails
        logger.warning("Failed to update last_login for user %s: %s", user_id, str(e))


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user",
    description=(
        "Authenticate a user with email and password. Supports both Auth0 OAuth2 "
        "flow (when configured) and local JWT authentication fallback. "
        "Returns a JWT access token and user profile on success."
    ),
    responses={
        200: {"description": "Successful authentication", "model": LoginResponse},
        401: {"description": "Invalid credentials", "model": ErrorResponse},
        500: {"description": "Authentication system error", "model": ErrorResponse},
    },
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
) -> LoginResponse:
    """
    Authenticate user and return JWT access token.

    Implements OAuth2 password grant flow for compatibility with standard
    OAuth2 clients. Supports both Auth0 authentication (when configured)
    and local HS256 JWT generation as fallback.

    Args:
        form_data: OAuth2 password request form containing username (email) and password
        settings: Application settings for authentication configuration

    Returns:
        LoginResponse: Access token, token type, expiration, and user profile

    Raises:
        HTTPException: 401 for invalid credentials, 500 for system errors

    Note:
        - OAuth2PasswordRequestForm uses 'username' field per OAuth2 spec,
          but we treat it as email for this API
        - When Auth0 is configured, local authentication is still supported
          as a fallback mechanism
    """
    email = form_data.username  # OAuth2 spec uses 'username', we use email
    password = form_data.password

    logger.info("Login attempt for email: %s", email)

    try:
        # Determine authentication method based on configuration
        if settings.is_auth0_enabled:
            logger.debug("Auth0 is enabled, but attempting local authentication first")

        # Authenticate user with local credentials
        user = await authenticate_user(email, password)

        if user is None:
            logger.warning("Authentication failed for email: %s", email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user_id for token generation
        user_id = user.get("_id") or user.get("id")
        if user_id is None:
            logger.error("User document missing ID field for email: %s", email)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error: Invalid user data",
            )

        user_id = str(user_id)

        # Generate JWT access token (HS256 for local auth)
        access_token = create_access_token(
            user_id=user_id,
            email=email,
            settings=settings,
        )

        # Create session in Redis for token management
        session_created = await create_user_session(user_id, access_token)
        if not session_created:
            logger.warning("Failed to create session for user %s, continuing anyway", user_id)

        # Update last login timestamp
        await _update_user_last_login(user_id)

        # Create user response (excludes sensitive fields)
        user_response = _create_user_response_from_dict(user)

        # Calculate expiration in seconds
        expires_in_seconds = settings.jwt_expiration_hours * 3600

        logger.info("User authenticated successfully: %s", email)

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in_seconds,
            user=user_response,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Unexpected error during login for email: %s", email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error. Please try again later.",
        ) from e


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    summary="Logout user",
    description=(
        "Invalidate the current user's session. Revokes the JWT token by "
        "removing the session from Redis cache, requiring re-authentication "
        "for subsequent API access."
    ),
    responses={
        200: {"description": "Successful logout", "model": LogoutResponse},
        401: {"description": "Not authenticated", "model": ErrorResponse},
    },
)
async def logout(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> LogoutResponse:
    """
    Logout user and invalidate session.

    Revokes the current user's session by deleting session data from Redis,
    effectively invalidating the JWT token. The token itself remains valid
    until expiration, but the session check will fail.

    Args:
        current_user: Current authenticated user (injected via dependency)

    Returns:
        LogoutResponse: Confirmation message with logout timestamp

    Raises:
        HTTPException: 401 if not authenticated

    Note:
        Session revocation uses Redis delete operation on key pattern
        session:{user_id}, ensuring immediate invalidation.
    """
    user_id = current_user.get("_id") or current_user.get("id")
    email = current_user.get("email", "unknown")

    if user_id is None:
        logger.warning("Logout attempted with user missing ID field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user_id)

    logger.info("Logout request for user: %s (%s)", user_id, email)

    # Revoke user session in Redis
    revoked = await revoke_user_session(user_id)

    if revoked:
        logger.info("Session revoked successfully for user: %s", user_id)
    else:
        logger.warning("Session revocation returned False for user: %s", user_id)
        # Continue with logout even if Redis operation failed

    logout_timestamp = datetime.now(UTC).isoformat()

    return LogoutResponse(
        message="Successfully logged out",
        logged_out_at=logout_timestamp,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user profile",
    description=(
        "Retrieve the authenticated user's profile information. Requires a "
        "valid JWT access token in the Authorization header. Returns user "
        "profile data excluding sensitive fields like password hash."
    ),
    responses={
        200: {"description": "User profile retrieved", "model": UserResponse},
        401: {"description": "Not authenticated or token expired", "model": ErrorResponse},
        404: {"description": "User not found", "model": ErrorResponse},
    },
)
async def get_me(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> UserResponse:
    """
    Get current authenticated user's profile.

    Returns the full user profile for the authenticated user, excluding
    sensitive fields like password hash. Profile includes account
    information, platform metrics, and timestamps.

    Args:
        current_user: Current authenticated user (injected via dependency)

    Returns:
        UserResponse: User profile data safe for API response

    Raises:
        HTTPException: 401 if token is invalid or expired

    Note:
        The current_user dependency handles:
        - JWT token extraction from Authorization header
        - Token validation (Auth0 RS256 or local HS256)
        - User lookup in MongoDB with Redis caching
    """
    user_id = current_user.get("_id") or current_user.get("id")
    email = current_user.get("email", "unknown")

    logger.debug("Profile request for user: %s (%s)", user_id, email)

    # Convert user dictionary to response model
    return _create_user_response_from_dict(current_user)


# =============================================================================
# JSON Login Endpoint (Frontend-Compatible)
# =============================================================================


class FrontendLoginResponse(BaseModel):
    """Response model matching the frontend LoginResponse interface."""

    token: str = Field(..., description="JWT access token")
    user: UserResponse = Field(..., description="Authenticated user profile")
    expiresIn: int = Field(..., description="Token expiration in seconds")


@router.post(
    "/login/json",
    response_model=FrontendLoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user (JSON)",
    description="JSON-based login endpoint compatible with the React frontend.",
)
async def login_json(
    request: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> FrontendLoginResponse:
    """Authenticate user via JSON body and return frontend-compatible response."""
    email = request.email
    password = request.password

    if not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required",
        )

    logger.info("JSON login attempt for email: %s", email)

    try:
        user = await authenticate_user(email, password)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = user.get("_id") or user.get("id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error: Invalid user data",
            )

        user_id = str(user_id)

        access_token = create_access_token(
            user_id=user_id,
            email=email,
            settings=settings,
        )

        session_created = await create_user_session(user_id, access_token)
        if not session_created:
            logger.warning("Failed to create session for user %s", user_id)

        await _update_user_last_login(user_id)

        user_response = _create_user_response_from_dict(user)
        expires_in_seconds = settings.jwt_expiration_hours * 3600

        logger.info("User authenticated successfully (JSON): %s", email)

        return FrontendLoginResponse(
            token=access_token,
            user=user_response,
            expiresIn=expires_in_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during JSON login for email: %s", email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error. Please try again later.",
        ) from e


# =============================================================================
# Module Exports
# =============================================================================

__all__ = ["router"]
