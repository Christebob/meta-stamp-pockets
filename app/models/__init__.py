"""
Models Package for META-STAMP V3.

This package provides Pydantic models for the META-STAMP V3 platform,
including data models for assets, users, wallet/transactions, analytics,
and fingerprinting.

All models are designed for MongoDB compatibility with proper ObjectId
handling via _id alias support and datetime serialization.

Models Overview:
    - Asset: Creative asset metadata and storage references
    - User: User authentication and profile data
    - WalletBalance: User wallet balance and earnings tracking
    - Transaction: Individual financial transaction records
    - AITouchValueCalculation: AI Touch Value™ calculation results
    - Fingerprint: Multi-modal asset fingerprinting data

Example Usage:
    ```python
    from app.models import Asset, User, WalletBalance, Transaction
    from app.models import AITouchValueCalculation, Fingerprint
    from app.models import FileType, UploadStatus, TransactionType

    # Create a new asset
    asset = Asset(
        user_id="user123",
        file_name="artwork.png",
        file_type=FileType.IMAGE,
        file_size=1024000,
        mime_type="image/png",
        s3_key="uploads/artwork.png",
        s3_bucket="meta-stamp"
    )

    # Create a transaction
    transaction = Transaction(
        user_id="user123",
        transaction_type=TransactionType.EARNING,
        amount=Decimal("50.00"),
        description="AI Touch Value™ earning"
    )
    ```

Per Agent Action Plan section 0.6 transformation mapping for
backend/app/models/__init__.py as package marker that exports all models.
"""

__version__ = "1.0.0"

# =============================================================================
# ASSET MODELS
# =============================================================================
# =============================================================================
# ANALYTICS MODELS
# =============================================================================
from app.models.analytics import (
    # Constants
    EQUITY_FACTOR,
    AITouchValueCalculation,
)
from app.models.asset import (
    # Constants
    DANGEROUS_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    Asset,
    AssetCreate,
    AssetResponse,
    FileType,
    ProcessingStatus,
    UploadStatus,
)

# =============================================================================
# FINGERPRINT MODELS
# =============================================================================
from app.models.fingerprint import (
    Fingerprint,
    FingerprintType,
    ProcessingStatus as FingerprintProcessingStatus,
)
from app.models.pocket import (
    Pocket,
    PocketCreateRequest,
    PocketPullResponse,
    PocketResponse,
    PocketStatus,
)

# =============================================================================
# USER MODELS
# =============================================================================
from app.models.user import (
    # Constants
    SUPPORTED_PLATFORMS,
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
)

# =============================================================================
# WALLET MODELS
# =============================================================================
from app.models.wallet import (
    # Constants
    MINIMUM_PAYOUT_THRESHOLD,
    SUPPORTED_CURRENCIES,
    Transaction,
    TransactionStatus,
    TransactionType,
    WalletBalance,
)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "DANGEROUS_EXTENSIONS",
    # Analytics constants
    "EQUITY_FACTOR",
    # Asset constants
    "MAX_FILE_SIZE_BYTES",
    # Wallet constants
    "MINIMUM_PAYOUT_THRESHOLD",
    "SUPPORTED_CURRENCIES",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIME_TYPES",
    # User constants
    "SUPPORTED_PLATFORMS",
    # Analytics models
    "AITouchValueCalculation",
    # Asset models and enums
    "Asset",
    "AssetCreate",
    "AssetResponse",
    "FileType",
    # Fingerprint models and enums
    "Fingerprint",
    "FingerprintProcessingStatus",
    "FingerprintType",
    "Pocket",
    "PocketCreateRequest",
    "PocketPullResponse",
    "PocketResponse",
    "PocketStatus",
    "ProcessingStatus",
    "TokenResponse",
    "Transaction",
    "TransactionStatus",
    "TransactionType",
    "UploadStatus",
    # User models
    "User",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "UserUpdate",
    # Wallet models and enums
    "WalletBalance",
    # Version
    "__version__",
]
