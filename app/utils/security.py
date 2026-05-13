"""
Security utilities module for META-STAMP V3.

This module provides comprehensive security utilities including:
- JWT token generation and validation (HS256 for local fallback, RS256 for Auth0)
- Password hashing with bcrypt (minimum 12 rounds)
- Secure random string generation
- Token expiration management

Per Agent Action Plan sections 0.3 (Auth0 + local JWT requirements),
0.4 (authentication implementation), 0.6 (security.py specification),
and 0.10 (security constraints).
"""

import logging
import secrets
import string

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError
from passlib.context import CryptContext


# Configure logger for security operations
logger = logging.getLogger(__name__)

# ==============================================================================
# SECURITY CONSTANTS
# ==============================================================================

# Minimum password length for validation
MIN_PASSWORD_LENGTH = 8

# Expected number of parts in Bearer authorization header (scheme + token)
BEARER_TOKEN_PARTS = 2

# ==============================================================================
# PASSWORD HASHING CONFIGURATION
# ==============================================================================

# Password hashing context using bcrypt with minimum 12 rounds
# per Agent Action Plan section 0.10 security requirements
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


# ==============================================================================
# PASSWORD HASHING FUNCTIONS
# ==============================================================================


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt with 12 rounds.

    This function creates a secure hash of the provided password using
    the bcrypt algorithm configured with a minimum of 12 rounds for
    enhanced security per Agent Action Plan section 0.10.

    Args:
        password: The plaintext password to hash.

    Returns:
        The hashed password string suitable for storage.

    Example:
        >>> hashed = hash_password("my_secure_password")
        >>> print(hashed)  # Returns bcrypt hash string
    """
    if not password:
        raise ValueError("Password cannot be empty")

    result: str = pwd_context.hash(password)
    return result


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a hashed password.

    This function securely compares a plaintext password with a previously
    hashed password using constant-time comparison to prevent timing attacks.

    Args:
        plain_password: The plaintext password to verify.
        hashed_password: The hashed password to verify against.

    Returns:
        True if the password matches, False otherwise.

    Example:
        >>> hashed = hash_password("my_password")
        >>> verify_password("my_password", hashed)  # Returns True
        >>> verify_password("wrong_password", hashed)  # Returns False
    """
    if not plain_password or not hashed_password:
        return False

    try:
        result: bool = pwd_context.verify(plain_password, hashed_password)
        return result
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


# ==============================================================================
# JWT TOKEN GENERATION AND VALIDATION
# ==============================================================================


def generate_jwt_token(
    data: dict[str, Any],
    secret_key: str,
    expires_delta: timedelta | None = None,
    algorithm: str = "HS256",
) -> str:
    """
    Generate a JWT token with the provided data and expiration.

    Creates a JWT token containing the provided data with automatic
    expiration time management. Default expiration is 24 hours per
    Agent Action Plan section 0.3 authentication requirements.

    Args:
        data: Dictionary of claims to include in the token payload.
        secret_key: The secret key used for signing the token.
        expires_delta: Optional custom expiration time delta.
                      Defaults to 24 hours if not specified.
        algorithm: The signing algorithm to use. Defaults to "HS256"
                  for local JWT fallback per section 0.3.

    Returns:
        The encoded JWT token string.

    Raises:
        ValueError: If secret_key is empty or data is None.

    Example:
        >>> token = generate_jwt_token(
        ...     {"sub": "user123", "email": "user@example.com"},
        ...     "my_secret_key"
        ... )
    """
    if not secret_key:
        raise ValueError("Secret key cannot be empty")

    if data is None:
        raise ValueError("Token data cannot be None")

    # Create a copy of the data to avoid modifying the original
    payload = data.copy()

    # Calculate expiration time (default 24 hours per section 0.3)
    if expires_delta is not None:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(hours=24)

    # Add standard JWT claims
    payload.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "nbf": datetime.now(UTC),  # Not valid before current time
        }
    )

    # Encode and return the token
    result: str = jwt.encode(payload, secret_key, algorithm=algorithm)
    return result


def validate_jwt_token(
    token: str, secret_key: str, algorithm: str = "HS256"
) -> dict[str, Any] | None:
    """
    Validate and decode a JWT token.

    Attempts to decode and validate a JWT token using the provided
    secret key. Returns the payload if valid, None if invalid.
    Handles expired tokens, invalid signatures, and malformed tokens
    gracefully per Agent Action Plan section 0.8 error handling.

    Args:
        token: The JWT token string to validate.
        secret_key: The secret key used for signature verification.
        algorithm: The signing algorithm used. Defaults to "HS256".

    Returns:
        The decoded payload dictionary if valid, None if invalid.

    Example:
        >>> payload = validate_jwt_token(token, "my_secret_key")
        >>> if payload:
        ...     print(f"User: {payload.get('sub')}")
    """
    if not token or not secret_key:
        logger.warning("Token validation failed: empty token or secret key")
        return None

    try:
        # Decode the token with verification
        result: dict[str, Any] = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
                "require": ["exp", "iat"],
            },
        )
        return result

    except ExpiredSignatureError:
        logger.warning("Token validation failed: token has expired")
        return None

    except JWTClaimsError as e:
        logger.warning(f"Token validation failed: invalid claims - {e}")
        return None

    except JWTError as e:
        logger.warning(f"Token validation failed: {e}")
        return None

    except Exception:
        logger.exception("Unexpected error during token validation")
        return None


# ==============================================================================
# AUTH0 JWT VALIDATION
# ==============================================================================

# Cache configuration
_JWKS_CACHE_TTL = timedelta(hours=1)


class _JWKSCache:
    """Internal cache class for Auth0 JWKS data."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.timestamp: datetime | None = None

    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached JWKS if valid."""
        now = datetime.now(UTC)
        if (
            key in self.cache
            and self.timestamp is not None
            and now - self.timestamp < _JWKS_CACHE_TTL
        ):
            result: dict[str, Any] = self.cache[key]
            return result
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Set cached JWKS value."""
        self.cache[key] = value
        self.timestamp = datetime.now(UTC)


# Singleton cache instance
_jwks_cache = _JWKSCache()


def _get_auth0_jwks(auth0_domain: str) -> dict[str, Any]:
    """
    Fetch and cache Auth0 JSON Web Key Set (JWKS).

    Retrieves the JWKS from Auth0 domain for RS256 token verification.
    Implements caching to reduce network calls with 1-hour TTL.

    Args:
        auth0_domain: The Auth0 domain (e.g., "your-tenant.auth0.com").

    Returns:
        The JWKS dictionary containing the public keys.

    Raises:
        ValueError: If JWKS fetch fails or returns invalid data.
    """
    # Check cache first
    cached = _jwks_cache.get(auth0_domain)
    if cached is not None:
        return cached

    # Construct JWKS URL
    jwks_url = f"https://{auth0_domain}/.well-known/jwks.json"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(jwks_url)
            response.raise_for_status()
            jwks: dict[str, Any] = response.json()

        if "keys" not in jwks or not jwks["keys"]:
            raise ValueError("Invalid JWKS: no keys found")

        # Update cache
        _jwks_cache.set(auth0_domain, jwks)

        return jwks

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        logger.exception("Failed to fetch JWKS from Auth0: HTTP %s", status_code)
        raise ValueError(f"Failed to fetch JWKS: HTTP {status_code}") from e

    except httpx.RequestError as e:
        logger.exception("Network error fetching JWKS from Auth0")
        raise ValueError(f"Network error fetching JWKS: {e}") from e

    except ValueError:
        raise

    except Exception as e:
        logger.exception("Unexpected error fetching JWKS")
        raise ValueError(f"Failed to fetch JWKS: {e}") from e


def _get_rsa_key_from_jwks(jwks: dict[str, Any], token: str) -> dict[str, Any] | None:
    """
    Extract the RSA public key from JWKS matching the token's kid.

    Args:
        jwks: The JSON Web Key Set dictionary.
        token: The JWT token to extract kid from.

    Returns:
        The RSA key dictionary if found, None otherwise.
    """
    try:
        # Get unverified header to extract kid
        unverified_header = jwt.get_unverified_header(token)
        token_kid = unverified_header.get("kid")

        if not token_kid:
            logger.warning("Token does not contain kid in header")
            return None

        # Find matching key in JWKS
        for key in jwks.get("keys", []):
            if key.get("kid") == token_kid:
                return {
                    "kty": key.get("kty"),
                    "kid": key.get("kid"),
                    "use": key.get("use"),
                    "n": key.get("n"),
                    "e": key.get("e"),
                }

        logger.warning(f"No matching key found for kid: {token_kid}")
        return None

    except Exception:
        logger.exception("Error extracting RSA key from JWKS")
        return None


def validate_auth0_token(token: str, auth0_domain: str, api_audience: str) -> dict[str, Any] | None:
    """
    Validate an Auth0 JWT token using RS256 algorithm.

    Verifies the token signature against Auth0's public keys (JWKS),
    validates the issuer and audience claims, and checks expiration.
    Per Agent Action Plan section 0.4 authentication implementation.

    Args:
        token: The JWT token string to validate.
        auth0_domain: The Auth0 domain (e.g., "your-tenant.auth0.com").
        api_audience: The API audience/identifier configured in Auth0.

    Returns:
        The decoded payload dictionary if valid, None if invalid.

    Example:
        >>> payload = validate_auth0_token(
        ...     token,
        ...     "your-tenant.auth0.com",
        ...     "https://api.yourdomain.com"
        ... )
        >>> if payload:
        ...     user_id = payload.get("sub")
    """
    # Validate required parameters
    validation_errors = {
        "empty token": not token,
        "empty domain": not auth0_domain,
        "empty audience": not api_audience,
    }
    for error_msg, condition in validation_errors.items():
        if condition:
            logger.warning("Auth0 token validation failed: %s", error_msg)
            return None

    try:
        # Fetch JWKS and extract RSA key
        jwks = _get_auth0_jwks(auth0_domain)
        rsa_key = _get_rsa_key_from_jwks(jwks, token)

        if not rsa_key:
            logger.warning("Auth0 token validation failed: no matching RSA key")
            return None

        # Decode and validate the token
        issuer = f"https://{auth0_domain}/"
        decoded_payload: dict[str, Any] = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=api_audience,
            issuer=issuer,
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
                "require": ["exp", "iat", "aud", "iss", "sub"],
            },
        )
        return decoded_payload

    except ExpiredSignatureError:
        logger.warning("Auth0 token validation failed: token has expired")
    except JWTClaimsError as e:
        logger.warning("Auth0 token validation failed: invalid claims - %s", e)
    except JWTError as e:
        logger.warning("Auth0 token validation failed: %s", e)
    except ValueError:
        logger.exception("Auth0 token validation failed")
    except Exception:
        logger.exception("Unexpected error during Auth0 token validation")

    return None


# ==============================================================================
# SECURE RANDOM STRING GENERATION
# ==============================================================================


def generate_secure_random(length: int = 32, include_punctuation: bool = False) -> str:
    """
    Generate a cryptographically secure random string.

    Creates a secure random string using the secrets module suitable
    for session IDs, API keys, tokens, and other security-sensitive
    purposes per Agent Action Plan section 0.6.

    Args:
        length: The desired length of the random string. Defaults to 32.
        include_punctuation: If True, includes punctuation characters
                           in the output. Defaults to False for URL-safe strings.

    Returns:
        A cryptographically secure random string.

    Example:
        >>> session_id = generate_secure_random(32)
        >>> api_key = generate_secure_random(64, include_punctuation=False)
    """
    if length <= 0:
        raise ValueError("Length must be a positive integer")

    if include_punctuation:
        # Use custom character set including punctuation
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return "".join(secrets.choice(alphabet) for _ in range(length))
    # Use URL-safe characters for compatibility
    # Note: token_urlsafe returns base64-encoded string which is ~4/3 the byte length
    # We calculate byte length needed to get approximately the requested character length
    byte_length = (length * 3) // 4 + 1
    token = secrets.token_urlsafe(byte_length)
    return token[:length]


# ==============================================================================
# TOKEN EXPIRATION HELPERS
# ==============================================================================


def create_expiration_time(hours: int = 24) -> datetime:
    """
    Create an expiration datetime for token generation.

    Generates a UTC datetime representing when a token should expire.
    Default expiration is 24 hours per Agent Action Plan section 0.3
    authentication requirements.

    Args:
        hours: Number of hours until expiration. Defaults to 24.

    Returns:
        A timezone-aware UTC datetime for token expiration.

    Example:
        >>> exp_time = create_expiration_time(24)  # 24 hours from now
        >>> short_exp = create_expiration_time(1)  # 1 hour from now
    """
    if hours < 0:
        raise ValueError("Hours must be non-negative")

    return datetime.now(UTC) + timedelta(hours=hours)


# ==============================================================================
# ADDITIONAL SECURITY UTILITIES
# ==============================================================================


def extract_token_from_header(authorization_header: str | None) -> str | None:
    """
    Extract the JWT token from an Authorization header.

    Parses the Authorization header and extracts the token,
    supporting the "Bearer" scheme.

    Args:
        authorization_header: The full Authorization header value
                             (e.g., "Bearer eyJ...").

    Returns:
        The extracted token string, or None if invalid format.

    Example:
        >>> token = extract_token_from_header("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    """
    if not authorization_header:
        return None

    parts = authorization_header.split()

    if len(parts) != BEARER_TOKEN_PARTS:
        return None

    scheme, token = parts

    if scheme.lower() != "bearer":
        return None

    return token


def is_token_expired(token: str, secret_key: str, algorithm: str = "HS256") -> bool:
    """
    Check if a JWT token has expired without full validation.

    Performs a quick check on token expiration without validating
    the signature. Useful for determining if a token refresh is needed.

    Args:
        token: The JWT token to check.
        secret_key: The secret key (required for decoding).
        algorithm: The signing algorithm. Defaults to "HS256".

    Returns:
        True if the token is expired, False otherwise.
    """
    try:
        # Decode without verification to check expiration
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            options={"verify_signature": True, "verify_exp": False},  # We'll check manually
        )

        exp = payload.get("exp")
        if exp is None:
            return True  # No expiration means treat as expired

        # Convert to datetime and compare with current time
        exp_datetime = datetime.fromtimestamp(exp, tz=UTC)
        return datetime.now(UTC) > exp_datetime

    except JWTError:
        return True  # If we can't decode, treat as expired

    except Exception:
        return True


def generate_password_reset_token(user_id: str, secret_key: str) -> str:
    """
    Generate a short-lived token for password reset.

    Creates a JWT token with a 1-hour expiration specifically for
    password reset operations.

    Args:
        user_id: The user ID to embed in the token.
        secret_key: The secret key for signing.

    Returns:
        A password reset token valid for 1 hour.
    """
    return generate_jwt_token(
        data={"sub": user_id, "purpose": "password_reset"},
        secret_key=secret_key,
        expires_delta=timedelta(hours=1),
        algorithm="HS256",
    )


def validate_password_strength(password: str) -> tuple[bool, str | None]:
    """
    Validate password meets minimum strength requirements.

    Checks that a password meets security requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Args:
        password: The password to validate.

    Returns:
        A tuple of (is_valid, error_message).
        error_message is None if valid, otherwise describes the issue.
    """
    if not password:
        return False, "Password cannot be empty"

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    special_chars = set(string.punctuation)
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"

    return True, None


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "create_expiration_time",
    # Additional utilities
    "extract_token_from_header",
    # JWT token operations
    "generate_jwt_token",
    "generate_password_reset_token",
    # Utility functions
    "generate_secure_random",
    "hash_password",
    "is_token_expired",
    # Password hashing
    "pwd_context",
    "validate_auth0_token",
    "validate_jwt_token",
    "validate_password_strength",
    "verify_password",
]
