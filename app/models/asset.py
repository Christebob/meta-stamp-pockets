"""
Asset Pydantic models for META-STAMP V3.

This module defines the Asset model for storing creative asset metadata,
storage references, upload status tracking, and fingerprint associations.
Supports all file types: text, images, audio, video, and URLs.

Per Agent Action Plan section 0.6 transformation mapping for backend/app/models/asset.py
and section 0.4 data layer implementation requirements.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum file size: 500MB in bytes
MAX_FILE_SIZE_BYTES: int = 524_288_000

# Supported file extensions per type per Agent Action Plan section 0.3
SUPPORTED_EXTENSIONS: dict[str, list[str]] = {
    "text": [".txt", ".md", ".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".webp"],
    "audio": [".mp3", ".wav", ".aac"],
    "video": [".mp4", ".mov", ".avi"],
    "url": [],  # URLs don't have file extensions
}

# MIME types for supported file formats
SUPPORTED_MIME_TYPES: dict[str, list[str]] = {
    "text": ["text/plain", "text/markdown", "application/pdf"],
    "image": ["image/png", "image/jpeg", "image/webp"],
    "audio": ["audio/mpeg", "audio/wav", "audio/aac", "audio/mp3"],
    "video": ["video/mp4", "video/quicktime", "video/x-msvideo"],
    "url": ["text/html", "application/json"],
}

# Dangerous file extensions that must be rejected
# Per Agent Action Plan section 0.3 Security Constraints
DANGEROUS_EXTENSIONS: list[str] = [
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",  # Archives
    ".exe",
    ".bin",
    ".sh",
    ".app",
    ".msi",
    ".iso",
    ".dmg",  # Executables
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".js",  # Scripts
]


# =============================================================================
# ENUMS
# =============================================================================


class FileType(str, Enum):
    """
    Enumeration of supported file types for asset uploads.

    Per Agent Action Plan section 0.3, supported file types are:
    - Text: .txt, .md, .pdf
    - Images: .png, .jpg, .jpeg, .webp
    - Audio: .mp3, .wav, .aac
    - Video: .mp4, .mov, .avi
    - URLs: YouTube, Vimeo, general webpages
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    URL = "url"


class UploadStatus(str, Enum):
    """
    Enumeration of asset upload status values.

    Status flow: QUEUED → PROCESSING → READY (or FAILED)

    States:
    - QUEUED: Asset upload initiated, waiting for processing
    - PROCESSING: Asset is being uploaded/processed/fingerprinted
    - READY: Asset successfully processed and available
    - FAILED: Asset processing failed with error
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ProcessingStatus(str, Enum):
    """
    Enumeration of fingerprint processing status values.

    Tracks the fingerprinting/analysis stage separate from upload status.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# MODELS
# =============================================================================


class Asset(BaseModel):
    """
    Pydantic model for creative assets in META-STAMP V3.

    Represents uploaded creative content including file metadata,
    storage references (S3/MinIO), upload status tracking, and
    fingerprint associations for AI training detection.

    Attributes:
        id: MongoDB ObjectId as string (aliased from _id)
        user_id: Reference to owning user
        file_name: Original filename with extension
        file_type: Type category (text, image, audio, video, url)
        file_size: Size in bytes (max 500MB)
        mime_type: MIME content type
        s3_key: Object key in S3/MinIO storage
        s3_bucket: Storage bucket name
        upload_status: Current upload state
        processing_status: Fingerprint processing state
        error_message: Error details if failed
        fingerprint_id: Reference to fingerprint document
        url_source: Original URL for URL-type assets
        metadata: File-specific metadata (EXIF, duration, etc.)
        created_at: Asset creation timestamp (UTC)
        updated_at: Last modification timestamp (UTC)

    Example:
        ```python
        asset = Asset(
            user_id="user123",
            file_name="my_artwork.png",
            file_type=FileType.IMAGE,
            file_size=1024000,
            mime_type="image/png",
            s3_key="uploads/user123/my_artwork.png",
            s3_bucket="meta-stamp-assets",
            upload_status=UploadStatus.READY
        )
        ```
    """

    # MongoDB ObjectId field with alias for _id
    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")

    # User association
    user_id: str = Field(
        ..., min_length=1, max_length=100, description="Reference to owning user's ID"
    )

    # File identification
    file_name: str = Field(
        ..., min_length=1, max_length=255, description="Original filename with extension"
    )

    file_type: FileType = Field(..., description="Type category of the asset")

    file_size: int = Field(
        ..., ge=0, le=MAX_FILE_SIZE_BYTES, description="File size in bytes (max 500MB)"
    )

    mime_type: str = Field(
        ..., min_length=1, max_length=100, description="MIME content type (e.g., 'image/png')"
    )

    # Storage references
    s3_key: str = Field(
        ..., min_length=1, max_length=1024, description="Object key in S3/MinIO storage"
    )

    s3_bucket: str = Field(..., min_length=1, max_length=63, description="Storage bucket name")

    # Status tracking
    upload_status: UploadStatus = Field(
        default=UploadStatus.QUEUED, description="Current upload/processing state"
    )

    processing_status: ProcessingStatus | None = Field(
        default=None, description="Fingerprint processing state (None if not started)"
    )

    error_message: str | None = Field(
        default=None, max_length=1000, description="Error details if upload/processing failed"
    )

    # Relationships
    fingerprint_id: str | None = Field(
        default=None, description="Reference to associated fingerprint document"
    )

    url_source: str | None = Field(
        default=None,
        max_length=2048,
        description="Original URL for URL-type assets (YouTube, Vimeo, web)",
    )

    # Metadata storage
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-specific metadata (EXIF, duration, resolution, etc.)",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Asset creation timestamp (UTC)"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last modification timestamp (UTC)"
    )

    # MongoDB configuration
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        },
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "file_name": "my_artwork.png",
                "file_type": "image",
                "file_size": 1024000,
                "mime_type": "image/png",
                "s3_key": "uploads/user123/my_artwork.png",
                "s3_bucket": "meta-stamp-assets",
                "upload_status": "ready",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """
        Validate and sanitize filename for safe storage.

        - Strips leading/trailing whitespace
        - Checks for dangerous file extensions
        - Ensures filename doesn't contain path traversal attempts

        Args:
            v: Input filename string

        Returns:
            Sanitized filename

        Raises:
            ValueError: If filename is invalid or dangerous
        """
        # Strip whitespace
        v = v.strip()

        if not v:
            raise ValueError("File name cannot be empty")

        # Check for path traversal attempts
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("File name cannot contain path separators or '..'")

        # Check for null bytes
        if "\x00" in v:
            raise ValueError("File name cannot contain null bytes")

        # Extract extension and check against dangerous types
        lower_name = v.lower()
        for dangerous_ext in DANGEROUS_EXTENSIONS:
            if lower_name.endswith(dangerous_ext):
                raise ValueError(
                    f"File type '{dangerous_ext}' is not allowed. "
                    f"Dangerous file types are rejected for security."
                )

        return v

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """
        Validate file size against 500MB maximum limit.

        Per Agent Action Plan section 0.3 Security Constraints:
        MAX asset size: 500 MB hard limit

        Args:
            v: File size in bytes

        Returns:
            Validated file size

        Raises:
            ValueError: If file size exceeds 500MB
        """
        if v < 0:
            raise ValueError("File size cannot be negative")

        if v > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = v / (1024 * 1024)
            raise ValueError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed "
                f"size of {max_mb:.0f} MB"
            )

        return v

    @field_validator("file_type", mode="before")
    @classmethod
    def validate_file_type(cls, v: Any) -> FileType:
        """
        Validate file type against allowed types.

        Accepts both string values and FileType enum members.

        Args:
            v: File type (string or FileType enum)

        Returns:
            FileType enum member

        Raises:
            ValueError: If file type is not supported
        """
        if isinstance(v, FileType):
            return v

        if isinstance(v, str):
            v_lower = v.lower()
            try:
                return FileType(v_lower)
            except ValueError:
                allowed_types = [ft.value for ft in FileType]
                raise ValueError(
                    f"Invalid file type '{v}'. Allowed types: {allowed_types}"
                ) from None

        raise ValueError(f"File type must be a string or FileType enum, got {type(v)}")

    @field_validator("upload_status", mode="before")
    @classmethod
    def validate_upload_status(cls, v: Any) -> UploadStatus:
        """
        Validate upload status against allowed values.

        Args:
            v: Upload status (string or UploadStatus enum)

        Returns:
            UploadStatus enum member

        Raises:
            ValueError: If status is invalid
        """
        if isinstance(v, UploadStatus):
            return v

        if isinstance(v, str):
            v_lower = v.lower()
            try:
                return UploadStatus(v_lower)
            except ValueError:
                allowed = [s.value for s in UploadStatus]
                raise ValueError(f"Invalid upload status '{v}'. Allowed: {allowed}") from None

        raise ValueError(f"Upload status must be a string or enum, got {type(v)}")

    @field_validator("url_source")
    @classmethod
    def validate_url_source(cls, v: str | None) -> str | None:
        """
        Validate URL source for URL-type assets.

        Basic validation to ensure URL is well-formed.
        More comprehensive validation happens in url_processor_service.

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
            raise ValueError("URL must start with http:// or https://")

        return v

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def is_ready(self) -> bool:
        """
        Check if asset is ready for use (uploaded and processed).

        Returns:
            True if upload_status is READY
        """
        return self.upload_status == UploadStatus.READY

    def is_failed(self) -> bool:
        """
        Check if asset upload/processing failed.

        Returns:
            True if upload_status is FAILED
        """
        return self.upload_status == UploadStatus.FAILED

    def is_processing(self) -> bool:
        """
        Check if asset is currently being processed.

        Returns:
            True if upload_status is PROCESSING or QUEUED
        """
        return self.upload_status in (UploadStatus.QUEUED, UploadStatus.PROCESSING)

    def has_fingerprint(self) -> bool:
        """
        Check if asset has an associated fingerprint.

        Returns:
            True if fingerprint_id is set
        """
        return self.fingerprint_id is not None

    def get_file_extension(self) -> str | None:
        """
        Extract file extension from filename.

        Returns:
            Lowercase file extension with dot (e.g., '.png') or None
        """
        if "." in self.file_name:
            return "." + self.file_name.rsplit(".", 1)[-1].lower()
        return None

    def get_file_size_mb(self) -> float:
        """
        Get file size in megabytes.

        Returns:
            File size in MB with 2 decimal precision
        """
        return round(self.file_size / (1024 * 1024), 2)

    def is_url_asset(self) -> bool:
        """
        Check if this is a URL-type asset.

        Returns:
            True if file_type is URL
        """
        return self.file_type == FileType.URL

    def mark_as_processing(self) -> None:
        """
        Update status to processing and set updated_at timestamp.
        """
        self.upload_status = UploadStatus.PROCESSING
        self.updated_at = datetime.now(UTC)

    def mark_as_ready(self) -> None:
        """
        Update status to ready and set updated_at timestamp.
        """
        self.upload_status = UploadStatus.READY
        self.updated_at = datetime.now(UTC)

    def mark_as_failed(self, error_message: str) -> None:
        """
        Update status to failed with error message.

        Args:
            error_message: Description of what failed
        """
        self.upload_status = UploadStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now(UTC)

    def set_fingerprint(self, fingerprint_id: str) -> None:
        """
        Associate a fingerprint with this asset.

        Args:
            fingerprint_id: MongoDB ObjectId of fingerprint document
        """
        self.fingerprint_id = fingerprint_id
        self.processing_status = ProcessingStatus.COMPLETED
        self.updated_at = datetime.now(UTC)


class AssetCreate(BaseModel):
    """
    Schema for creating a new asset (request body).

    Used for API request validation when uploading new assets.
    Does not include server-generated fields like id, timestamps, or status.
    """

    user_id: str = Field(
        ..., min_length=1, max_length=100, description="Reference to owning user's ID"
    )

    file_name: str = Field(
        ..., min_length=1, max_length=255, description="Original filename with extension"
    )

    file_type: FileType = Field(..., description="Type category of the asset")

    file_size: int = Field(..., ge=0, le=MAX_FILE_SIZE_BYTES, description="File size in bytes")

    mime_type: str = Field(..., min_length=1, max_length=100, description="MIME content type")

    s3_key: str = Field(
        ..., min_length=1, max_length=1024, description="Object key in S3/MinIO storage"
    )

    s3_bucket: str = Field(..., min_length=1, max_length=63, description="Storage bucket name")

    url_source: str | None = Field(
        default=None, max_length=2048, description="Original URL for URL-type assets"
    )

    metadata: dict[str, Any] = Field(default_factory=dict, description="File-specific metadata")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "file_name": "my_artwork.png",
                "file_type": "image",
                "file_size": 1024000,
                "mime_type": "image/png",
                "s3_key": "uploads/user123/my_artwork.png",
                "s3_bucket": "meta-stamp-assets",
            }
        },
    )


class AssetResponse(BaseModel):
    """
    Schema for asset API responses.

    Includes all asset fields formatted for API consumption.
    Uses string values for enums and ISO format for timestamps.
    """

    id: str | None = Field(None, description="Asset ID")
    user_id: str = Field(..., description="Owner user ID")
    file_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Asset type category")
    file_size: int = Field(..., description="Size in bytes")
    file_size_mb: float = Field(..., description="Size in megabytes")
    mime_type: str = Field(..., description="MIME content type")
    s3_key: str = Field(..., description="Storage object key")
    s3_bucket: str = Field(..., description="Storage bucket")
    upload_status: str = Field(..., description="Upload status")
    processing_status: str | None = Field(None, description="Processing status")
    error_message: str | None = Field(None, description="Error details")
    fingerprint_id: str | None = Field(None, description="Fingerprint reference")
    url_source: str | None = Field(None, description="Source URL")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_ready: bool = Field(..., description="Whether asset is ready")
    has_fingerprint: bool = Field(..., description="Whether fingerprinted")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    @classmethod
    def from_asset(cls, asset: Asset) -> "AssetResponse":
        """
        Create response from Asset model.

        Args:
            asset: Asset model instance

        Returns:
            AssetResponse for API
        """
        return cls(
            id=asset.id,
            user_id=asset.user_id,
            file_name=asset.file_name,
            file_type=(
                asset.file_type.value if isinstance(asset.file_type, FileType) else asset.file_type
            ),
            file_size=asset.file_size,
            file_size_mb=asset.get_file_size_mb(),
            mime_type=asset.mime_type,
            s3_key=asset.s3_key,
            s3_bucket=asset.s3_bucket,
            upload_status=(
                asset.upload_status.value
                if isinstance(asset.upload_status, UploadStatus)
                else asset.upload_status
            ),
            processing_status=(
                asset.processing_status.value
                if isinstance(asset.processing_status, ProcessingStatus)
                else asset.processing_status
            ),
            error_message=asset.error_message,
            fingerprint_id=asset.fingerprint_id,
            url_source=asset.url_source,
            metadata=asset.metadata,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
            is_ready=asset.is_ready(),
            has_fingerprint=asset.has_fingerprint(),
        )
