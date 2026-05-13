"""
META-STAMP V3 Upload Service Module

This module provides the core upload routing service implementing the hybrid upload
architecture as specified in the Agent Action Plan. The service intelligently routes
uploads based on file size:

- Files < 10MB: Direct upload via FastAPI multipart/form-data streaming to S3
- Files >= 10MB: Presigned URL flow with client-to-S3 direct transfer

The service integrates with:
- StorageService: For S3-compatible storage operations and presigned URL generation
- MetadataService: For extracting comprehensive file metadata after upload
- URLProcessorService: For processing URL-based uploads (YouTube, Vimeo, webpages)
- FileValidator: For comprehensive file validation (type, size, dangerous extensions)

Security Constraints (NON-NEGOTIABLE per Agent Action Plan section 0.3):
- REJECT ZIP files entirely: .zip, .rar, .7z, and all archive formats
- REJECT executables: .exe, .bin, .sh, .app, .msi, .iso, .dmg completely forbidden
- Maximum file size: 500 MB hard limit
- URL validation: Reject URLs pointing to dangerous file types
- Presigned URL expiration: 15 minutes

Author: META-STAMP V3 Development Team
"""

import logging
import tempfile
import uuid

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from fastapi import UploadFile

from app.config import Settings, get_settings
from app.core.database import get_db_client
from app.models.asset import Asset, FileType, UploadStatus
from app.services.metadata_service import MetadataService
from app.services.storage_service import StorageService
from app.services.url_processor_service import URLProcessorService
from app.utils.file_validator import (
    validate_file_extension,
    validate_file_size,
    validate_uploaded_file,
)


# Configure module logger
logger = logging.getLogger(__name__)


class UploadServiceError(Exception):
    """Base exception for upload service errors."""


class FileValidationError(UploadServiceError):
    """Exception raised when file validation fails."""


class StorageError(UploadServiceError):
    """Exception raised when storage operations fail."""


class UploadConfirmationError(UploadServiceError):
    """Exception raised when upload confirmation fails."""


class URLProcessingError(UploadServiceError):
    """Exception raised when URL processing fails."""


class UploadService:
    """
    Hybrid upload routing service implementing intelligent file upload handling.

    This service determines the optimal upload strategy based on file size and
    coordinates with storage, metadata extraction, and validation services to
    provide a complete upload workflow.

    Upload Strategies:
    - Direct Upload (<10MB): Files are streamed through FastAPI to a temporary
      location, validated, and then uploaded to S3. This approach minimizes
      complexity for small files while ensuring validation.
    - Presigned URL (>=10MB): For larger files, a presigned PUT URL is generated
      allowing the client to upload directly to S3, bypassing the backend for
      the file transfer. This reduces server load and enables resumable uploads.

    Attributes:
        storage: StorageService instance for S3 operations
        metadata: MetadataService instance for file metadata extraction
        url_processor: URLProcessorService for handling URL-based uploads
        settings: Application settings from config
        logger: Logger instance for operation tracking

    Example:
        ```python
        storage_service = StorageService()
        metadata_service = MetadataService()
        url_processor = URLProcessorService()

        upload_service = UploadService(
            storage_service=storage_service,
            metadata_service=metadata_service,
            url_processor_service=url_processor
        )

        # Direct upload for small files
        result = await upload_service.handle_direct_upload(file, user_id)

        # Presigned URL for large files
        presigned = await upload_service.generate_presigned_upload_url(
            filename="large_video.mp4",
            content_type="video/mp4",
            file_size=100_000_000,
            user_id="user123"
        )
        ```
    """

    def __init__(
        self,
        storage_service: StorageService,
        metadata_service: MetadataService,
        url_processor_service: URLProcessorService,
    ) -> None:
        """
        Initialize the UploadService with required service dependencies.

        Args:
            storage_service: StorageService instance for S3 operations including
                presigned URL generation, file upload/download, and existence checks.
            metadata_service: MetadataService instance for extracting comprehensive
                metadata from uploaded files (EXIF, audio properties, video specs).
            url_processor_service: URLProcessorService for handling URL-based uploads
                including YouTube transcript extraction, Vimeo metadata, and webpage scraping.
        """
        self.storage = storage_service
        self.metadata = metadata_service
        self.url_processor = url_processor_service
        self.settings: Settings = get_settings()
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            "UploadService initialized with direct upload threshold: %d MB, max size: %d MB",
            self.settings.direct_upload_threshold_mb,
            self.settings.max_upload_size_mb,
        )

    def detect_upload_strategy(self, file_size: int) -> str:
        """
        Determine the appropriate upload strategy based on file size.

        Implements the hybrid upload architecture by comparing the file size
        against the configured threshold (default: 10MB). Files below the
        threshold use direct upload through FastAPI, while larger files
        use the presigned URL flow for client-to-S3 direct transfer.

        Args:
            file_size: Size of the file in bytes to be uploaded.

        Returns:
            str: Either "direct" for files < 10MB or "presigned" for files >= 10MB.

        Example:
            ```python
            strategy = upload_service.detect_upload_strategy(5_000_000)  # "direct"
            strategy = upload_service.detect_upload_strategy(15_000_000)  # "presigned"
            ```
        """
        threshold_bytes = self.settings.direct_upload_threshold_bytes

        if file_size < threshold_bytes:
            self.logger.debug(
                "File size %d bytes is below threshold %d bytes, using direct upload",
                file_size,
                threshold_bytes,
            )
            return "direct"
        self.logger.debug(
            "File size %d bytes is at or above threshold %d bytes, using presigned URL",
            file_size,
            threshold_bytes,
        )
        return "presigned"

    def validate_file(
        self,
        filename: str,
        file_size: int,
        content_type: str | None = None,
    ) -> None:
        """
        Validate a file before upload by checking extension and size.

        Performs comprehensive validation according to security requirements:
        - Validates file extension against whitelist of allowed types
        - Rejects dangerous extensions (.zip, .exe, .bin, etc.) - NON-NEGOTIABLE
        - Enforces 500MB maximum file size limit
        - Logs content_type for tracking purposes (actual MIME validation
          requires file content and is done in handle_direct_upload)

        Note:
            Full MIME type validation with magic bytes is performed in
            handle_direct_upload where file content is available.

        Args:
            filename: Name of the file to validate including extension.
            file_size: Size of the file in bytes.
            content_type: Optional MIME type for logging (not validated here).

        Raises:
            FileValidationError: If any validation check fails.

        Example:
            ```python
            try:
                upload_service.validate_file("image.png", 5000000, "image/png")
            except FileValidationError as e:
                print(f"Validation failed: {e}")
            ```
        """
        # Validate file extension (also checks for dangerous extensions)
        is_valid_ext, file_type, error_msg = validate_file_extension(filename)
        if not is_valid_ext:
            self.logger.warning(
                "File validation failed for '%s': %s",
                filename,
                error_msg,
            )
            raise FileValidationError(error_msg or f"Invalid file extension: {filename}")

        # Validate file size (500MB limit per Agent Action Plan)
        is_valid_size, size_error = validate_file_size(file_size)
        if not is_valid_size:
            self.logger.warning(
                "File size validation failed for '%s': %d bytes - %s",
                filename,
                file_size,
                size_error,
            )
            raise FileValidationError(size_error or f"File size {file_size} exceeds maximum limit")

        # Log content_type for tracking (actual MIME validation requires file content)
        if content_type:
            self.logger.debug(
                "File '%s' has declared content-type: %s",
                filename,
                content_type,
            )

        self.logger.debug(
            "File validation passed for '%s': type=%s, size=%d bytes",
            filename,
            file_type,
            file_size,
        )

    def _generate_object_key(self, filename: str, user_id: str) -> str:
        """
        Generate a unique S3 object key combining timestamp and UUID.

        Creates a structured key that ensures no collisions in object storage
        while maintaining human-readable organization by user and date.

        Args:
            filename: Original filename with extension.
            user_id: User identifier for organizing uploads.

        Returns:
            str: Unique S3 object key in format:
                uploads/{user_id}/{YYYY-MM-DD}/{UUID}_{filename}
        """
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
        unique_id = str(uuid.uuid4())
        # Sanitize filename to remove potentially problematic characters
        safe_filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        object_key = f"uploads/{user_id}/{timestamp}/{unique_id}_{safe_filename}"

        self.logger.debug("Generated object key: %s", object_key)
        return object_key

    def _determine_file_type(self, filename: str, _content_type: str | None = None) -> FileType:
        """
        Determine the FileType enum value based on filename extension.

        Maps file extensions to the appropriate FileType enum value for
        database storage and metadata extraction routing.

        Args:
            filename: Filename with extension to analyze.
            content_type: Optional MIME type for additional context.

        Returns:
            FileType: The appropriate FileType enum value.
        """
        extension = Path(filename).suffix.lower()

        # Text files
        if extension in {".txt", ".md", ".pdf"}:
            return FileType.TEXT

        # Image files
        if extension in {".png", ".jpg", ".jpeg", ".webp"}:
            return FileType.IMAGE

        # Audio files
        if extension in {".mp3", ".wav", ".aac"}:
            return FileType.AUDIO

        # Video files
        if extension in {".mp4", ".mov", ".avi"}:
            return FileType.VIDEO

        # Default to TEXT for unknown types (shouldn't happen after validation)
        self.logger.warning(
            "Unknown file extension '%s', defaulting to TEXT file type",
            extension,
        )
        return FileType.TEXT

    async def _extract_metadata_from_s3_file(
        self,
        object_key: str,
        filename: str,
        file_type_str: str,
    ) -> tuple[dict[str, Any], Path | None]:
        """
        Download a file from S3 and extract its metadata.

        Helper method for confirm_presigned_upload that handles the temporary
        file download and metadata extraction process.

        Args:
            object_key: S3 object key where the file is stored.
            filename: Original filename for the temp file.
            file_type_str: File type string for metadata extraction routing.

        Returns:
            Tuple containing:
                - dict: Extracted metadata or error information
                - Path | None: Temporary file path for cleanup
        """
        temp_dir = tempfile.mkdtemp(prefix="metastamp_confirm_")
        temp_file_path = Path(temp_dir) / filename
        extracted_metadata: dict[str, Any] = {}

        try:
            download_success = await self.storage.download_file(
                object_key=object_key,
                file_path=str(temp_file_path),
            )

            if download_success:
                try:
                    extracted_metadata = await self.metadata.extract_metadata(
                        file_path=str(temp_file_path),
                        file_type=file_type_str,
                    )
                except Exception as metadata_error:
                    self.logger.warning(
                        "Metadata extraction failed for '%s': %s",
                        object_key,
                        str(metadata_error),
                    )
                    extracted_metadata = {"extraction_error": str(metadata_error)}
            else:
                self.logger.warning(
                    "Failed to download file for metadata extraction: %s",
                    object_key,
                )
                extracted_metadata = {"download_failed": True}

        except Exception as download_error:
            self.logger.warning(
                "Error downloading file for metadata extraction: %s",
                str(download_error),
            )
            extracted_metadata = {"download_error": str(download_error)}

        return extracted_metadata, temp_file_path

    async def handle_direct_upload(
        self,
        file: UploadFile,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Handle direct upload for files smaller than 10MB.

        Implements the direct upload flow for small files:
        1. Validate file type, size, and MIME type
        2. Generate unique S3 object key with timestamp and UUID
        3. Create temporary file and stream UploadFile content using aiofiles
        4. Extract metadata using MetadataService
        5. Upload file to S3 using StorageService
        6. Clean up temporary file
        7. Create Asset record in MongoDB with status "processing"
        8. Return asset details including asset_id

        Args:
            file: FastAPI UploadFile object containing the uploaded file.
            user_id: User identifier for the asset owner.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - asset_id: Unique identifier for the created asset
                - file_name: Original filename
                - file_type: Detected file type (TEXT, IMAGE, AUDIO, VIDEO)
                - file_size: Size in bytes
                - s3_key: Object key in S3 storage
                - upload_status: Current status ("processing")
                - created_at: ISO timestamp of creation

        Raises:
            FileValidationError: If file validation fails.
            StorageError: If S3 upload fails.
            Exception: For other unexpected errors with proper cleanup.

        Example:
            ```python
            result = await upload_service.handle_direct_upload(
                file=upload_file,
                user_id="user_12345"
            )
            print(f"Asset created: {result['asset_id']}")
            ```
        """
        filename = file.filename or "unknown_file"
        content_type = file.content_type
        temp_file_path: Path | None = None

        self.logger.info(
            "Starting direct upload for file '%s', content_type='%s', user='%s'",
            filename,
            content_type,
            user_id,
        )

        try:
            # Read file content to determine size
            file_content = await file.read()
            file_size = len(file_content)

            self.logger.debug(
                "Read file content: %d bytes for '%s'",
                file_size,
                filename,
            )

            # Validate the file using comprehensive validator
            is_valid, _detected_type, error_msg = await validate_uploaded_file(
                file_content=file_content,
                filename=filename,
                file_size=file_size,
            )

            if not is_valid:
                self.logger.warning(
                    "File validation failed for '%s': %s",
                    filename,
                    error_msg,
                )
                raise FileValidationError(error_msg or f"File validation failed for {filename}")

            # Generate unique object key
            object_key = self._generate_object_key(filename, user_id)
            file_type = self._determine_file_type(filename, content_type)

            # Create temporary file and write content
            temp_dir = tempfile.mkdtemp(prefix="metastamp_upload_")
            temp_file_path = Path(temp_dir) / filename

            async with aiofiles.open(temp_file_path, "wb") as temp_file:
                await temp_file.write(file_content)

            self.logger.debug(
                "Wrote file content to temporary path: %s",
                temp_file_path,
            )

            # Extract metadata from the file
            extracted_metadata = {}
            try:
                extracted_metadata = await self.metadata.extract_metadata(
                    file_path=str(temp_file_path),
                    file_type=file_type.value,
                )
                self.logger.debug(
                    "Extracted metadata for '%s': %s",
                    filename,
                    list(extracted_metadata.keys()),
                )
            except Exception as metadata_error:
                self.logger.warning(
                    "Metadata extraction failed for '%s': %s. Continuing with empty metadata.",
                    filename,
                    str(metadata_error),
                )
                extracted_metadata = {"extraction_error": str(metadata_error)}

            # Upload to S3
            upload_success = await self.storage.upload_file(
                file_path=str(temp_file_path),
                object_key=object_key,
                content_type=content_type or "application/octet-stream",
            )

            if not upload_success:
                self.logger.error(
                    "S3 upload failed for '%s' to key '%s'",
                    filename,
                    object_key,
                )
                raise StorageError(f"Failed to upload file {filename} to storage")

            self.logger.info(
                "Successfully uploaded '%s' to S3 with key '%s'",
                filename,
                object_key,
            )

            # Create Asset record in MongoDB
            asset = Asset(
                _id=str(uuid.uuid4()),
                user_id=user_id,
                file_name=filename,
                file_type=file_type,
                file_size=file_size,
                mime_type=content_type or "application/octet-stream",
                s3_key=object_key,
                s3_bucket=self.settings.s3_bucket_name,
                upload_status=UploadStatus.PROCESSING,
                metadata=extracted_metadata,
                created_at=datetime.now(UTC),
            )

            # Save to MongoDB
            db_client = get_db_client()
            assets_collection = db_client.get_assets_collection()
            asset_dict = asset.model_dump()
            await assets_collection.insert_one(asset_dict)

            self.logger.info(
                "Created asset record '%s' for file '%s'",
                asset.id,
                filename,
            )

            return {
                "asset_id": asset.id,
                "file_name": asset.file_name,
                "file_type": str(asset.file_type),
                "file_size": asset.file_size,
                "mime_type": asset.mime_type,
                "s3_key": asset.s3_key,
                "upload_status": str(asset.upload_status),
                "metadata": extracted_metadata,
                "created_at": asset.created_at.isoformat(),
            }

        except FileValidationError:
            # Re-raise validation errors without modification
            raise
        except StorageError:
            # Re-raise storage errors without modification
            raise
        except Exception as error:
            self.logger.exception(
                "Unexpected error during direct upload of '%s'",
                filename,
            )
            raise UploadServiceError(f"Direct upload failed for {filename}: {error!s}") from error

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    # Also remove the temporary directory
                    temp_file_path.parent.rmdir()
                    self.logger.debug(
                        "Cleaned up temporary file: %s",
                        temp_file_path,
                    )
                except OSError as cleanup_error:
                    self.logger.warning(
                        "Failed to clean up temporary file '%s': %s",
                        temp_file_path,
                        str(cleanup_error),
                    )

    async def generate_presigned_upload_url(
        self,
        filename: str,
        content_type: str,
        file_size: int,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Generate a presigned URL for large file uploads (>=10MB).

        Implements the presigned URL flow for large files:
        1. Validate file type against allowed extensions
        2. Validate file size (max 500MB)
        3. Generate unique S3 object key
        4. Generate presigned PUT URL with 15-minute expiration
        5. Create temporary Asset record in MongoDB with status "pending_upload"
        6. Return presigned URL, object key, asset_id, and expiration time

        The client then uses the presigned URL to upload directly to S3,
        bypassing the backend for the actual file transfer.

        Args:
            filename: Name of the file to be uploaded.
            content_type: MIME type of the file (e.g., "video/mp4").
            file_size: Size of the file in bytes (for validation).
            user_id: User identifier for the asset owner.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - presigned_url: S3 presigned PUT URL for client upload
                - object_key: S3 object key where file will be stored
                - asset_id: Unique identifier for the pending asset
                - expires_in: Expiration time in seconds (900 = 15 minutes)
                - expiration_time: ISO timestamp when URL expires

        Raises:
            FileValidationError: If file validation fails.
            StorageError: If presigned URL generation fails.

        Example:
            ```python
            result = await upload_service.generate_presigned_upload_url(
                filename="large_video.mp4",
                content_type="video/mp4",
                file_size=150_000_000,  # 150MB
                user_id="user_12345"
            )
            # Client uploads directly using result['presigned_url']
            ```
        """
        self.logger.info(
            "Generating presigned URL for file '%s', size=%d bytes, user='%s'",
            filename,
            file_size,
            user_id,
        )

        try:
            # Validate file before generating presigned URL
            self.validate_file(filename, file_size, content_type)

            # Generate unique object key
            object_key = self._generate_object_key(filename, user_id)
            file_type = self._determine_file_type(filename, content_type)

            # Calculate expiration
            expires_in = self.settings.presigned_url_expiration_seconds
            expiration_time = datetime.now(UTC).timestamp() + expires_in
            expiration_datetime = datetime.fromtimestamp(expiration_time, UTC)

            # Generate presigned PUT URL using storage service
            presigned_url = await self.storage.generate_presigned_upload_url(
                object_key=object_key,
                content_type=content_type,
                expiration=expires_in,
            )

            if not presigned_url:
                self.logger.error(
                    "Failed to generate presigned URL for '%s'",
                    filename,
                )
                raise StorageError(f"Failed to generate presigned URL for {filename}")

            self.logger.debug(
                "Generated presigned URL for '%s' with expiration in %d seconds",
                filename,
                expires_in,
            )

            # Create Asset record with queued status (pending presigned upload)
            asset_id = str(uuid.uuid4())
            asset = Asset(
                _id=asset_id,
                user_id=user_id,
                file_name=filename,
                file_type=file_type,
                file_size=file_size,
                mime_type=content_type,
                s3_key=object_key,
                s3_bucket=self.settings.s3_bucket_name,
                upload_status=UploadStatus.QUEUED,
                metadata={
                    "presigned_upload": True,
                    "expires_at": expiration_datetime.isoformat(),
                },
                created_at=datetime.now(UTC),
            )

            # Save to MongoDB
            db_client = get_db_client()
            assets_collection = db_client.get_assets_collection()
            asset_dict = asset.model_dump()
            await assets_collection.insert_one(asset_dict)

            self.logger.info(
                "Created pending asset '%s' for presigned upload of '%s'",
                asset_id,
                filename,
            )

            return {
                "presigned_url": presigned_url,
                "object_key": object_key,
                "asset_id": asset_id,
                "expires_in": expires_in,
                "expiration_time": expiration_datetime.isoformat(),
                "file_name": filename,
                "content_type": content_type,
            }

        except FileValidationError:
            raise
        except StorageError:
            raise
        except Exception as error:
            self.logger.exception(
                "Error generating presigned URL for '%s'",
                filename,
            )
            raise UploadServiceError(
                f"Failed to generate presigned URL for {filename}: {error!s}"
            ) from error

    async def confirm_presigned_upload(
        self,
        asset_id: str,
        object_key: str,
    ) -> dict[str, Any]:
        """
        Confirm that a presigned URL upload completed successfully.

        Called after the client has uploaded a file directly to S3 using a presigned URL.
        This method:
        1. Verifies the S3 object exists using storage service
        2. Gets file metadata from S3 (size, content-type)
        3. Downloads file temporarily for metadata extraction
        4. Extracts metadata using MetadataService
        5. Updates Asset record in MongoDB with status "processing"
        6. Cleans up temporary file
        7. Returns updated asset details

        Args:
            asset_id: Unique identifier of the pending asset record.
            object_key: S3 object key where the file was uploaded.

        Returns:
            Dict[str, Any]: Dictionary containing updated asset information:
                - asset_id: Asset identifier
                - file_name: Original filename
                - file_type: Detected file type
                - file_size: Actual size from S3
                - s3_key: S3 object key
                - upload_status: Updated status ("processing")
                - metadata: Extracted metadata
                - confirmed_at: ISO timestamp of confirmation

        Raises:
            UploadConfirmationError: If the S3 object doesn't exist or confirmation fails.
            StorageError: If S3 operations fail.

        Example:
            ```python
            # After client uploads to presigned URL
            result = await upload_service.confirm_presigned_upload(
                asset_id="asset_12345",
                object_key="uploads/user/2024-01-01/uuid_file.mp4"
            )
            ```
        """
        self.logger.info(
            "Confirming presigned upload for asset '%s', key='%s'",
            asset_id,
            object_key,
        )

        temp_file_path: Path | None = None

        try:
            # Verify S3 object exists
            exists = await self.storage.file_exists(object_key)
            if not exists:
                self.logger.error(
                    "S3 object not found for asset '%s' at key '%s'",
                    asset_id,
                    object_key,
                )
                raise UploadConfirmationError(
                    f"Upload confirmation failed: File not found in storage for asset {asset_id}"
                )

            # Get file metadata from S3
            s3_metadata = await self.storage.get_file_metadata(object_key)
            if not s3_metadata:
                self.logger.warning(
                    "Failed to retrieve S3 metadata for asset '%s'",
                    asset_id,
                )
                s3_metadata = {}

            self.logger.debug(
                "Retrieved S3 metadata for '%s': %s",
                asset_id,
                list(s3_metadata.keys()),
            )

            # Get the asset record from MongoDB
            db_client = get_db_client()
            assets_collection = db_client.get_assets_collection()
            asset_doc = await assets_collection.find_one({"id": asset_id})

            if not asset_doc:
                self.logger.error(
                    "Asset record not found in database for asset '%s'",
                    asset_id,
                )
                raise UploadConfirmationError(f"Asset record not found for asset_id: {asset_id}")

            # Extract actual file size from S3 metadata
            actual_file_size = s3_metadata.get("content_length", asset_doc.get("file_size", 0))
            content_type = s3_metadata.get("content_type", asset_doc.get("mime_type"))
            filename = asset_doc.get("file_name", "unknown")
            file_type_str = asset_doc.get("file_type", "text")

            # Download file temporarily for metadata extraction using helper
            extracted_metadata, temp_file_path = await self._extract_metadata_from_s3_file(
                object_key=object_key,
                filename=filename,
                file_type_str=file_type_str,
            )

            self.logger.debug(
                "Extracted metadata for confirmed upload '%s': %s",
                asset_id,
                list(extracted_metadata.keys()),
            )

            # Update asset record in MongoDB
            confirmation_time = datetime.now(UTC)
            update_data = {
                "upload_status": UploadStatus.PROCESSING.value,
                "file_size": actual_file_size,
                "mime_type": content_type,
                "metadata": extracted_metadata,
                "confirmed_at": confirmation_time,
            }

            await assets_collection.update_one(
                {"id": asset_id},
                {"$set": update_data},
            )

            self.logger.info(
                "Confirmed presigned upload for asset '%s', status updated to PROCESSING",
                asset_id,
            )

            return {
                "asset_id": asset_id,
                "file_name": filename,
                "file_type": file_type_str,
                "file_size": actual_file_size,
                "mime_type": content_type,
                "s3_key": object_key,
                "upload_status": UploadStatus.PROCESSING.value,
                "metadata": extracted_metadata,
                "confirmed_at": confirmation_time.isoformat(),
            }

        except UploadConfirmationError:
            raise
        except StorageError:
            raise
        except Exception as error:
            self.logger.exception(
                "Error confirming presigned upload for asset '%s'",
                asset_id,
            )
            raise UploadConfirmationError(
                f"Failed to confirm upload for asset {asset_id}: {error!s}"
            ) from error

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    temp_file_path.parent.rmdir()
                    self.logger.debug(
                        "Cleaned up temporary file after confirmation: %s",
                        temp_file_path,
                    )
                except OSError as cleanup_error:
                    self.logger.warning(
                        "Failed to clean up temporary file '%s': %s",
                        temp_file_path,
                        str(cleanup_error),
                    )

    async def handle_text_upload(
        self,
        content: str,
        filename: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Handle direct text content upload.

        Allows uploading text content directly as a string rather than as a file.
        This is useful for:
        - Creating assets from text inputs
        - Pasting content directly into the application
        - Programmatic asset creation

        The method:
        1. Validates content size (must be within limits)
        2. Generates unique S3 object key
        3. Uploads text content to S3
        4. Creates Asset record in MongoDB
        5. Returns asset details

        Args:
            content: Text content to upload as the asset.
            filename: Name to assign to the text file.
            user_id: User identifier for the asset owner.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - asset_id: Unique identifier for the created asset
                - file_name: Assigned filename
                - file_type: "text"
                - file_size: Size of content in bytes
                - s3_key: Object key in S3 storage
                - upload_status: Current status ("processing")
                - created_at: ISO timestamp of creation

        Raises:
            FileValidationError: If content exceeds size limits.
            StorageError: If S3 upload fails.

        Example:
            ```python
            result = await upload_service.handle_text_upload(
                content="This is my creative writing...",
                filename="my_story.txt",
                user_id="user_12345"
            )
            ```
        """
        self.logger.info(
            "Handling text upload for '%s', user='%s', content_length=%d",
            filename,
            user_id,
            len(content),
        )

        temp_file_path: Path | None = None

        try:
            # Convert content to bytes and calculate size
            content_bytes = content.encode("utf-8")
            file_size = len(content_bytes)

            # Validate content size
            max_size_bytes = self.settings.max_upload_size_bytes
            if file_size > max_size_bytes:
                self.logger.warning(
                    "Text content exceeds maximum size: %d > %d bytes",
                    file_size,
                    max_size_bytes,
                )
                raise FileValidationError(
                    f"Text content size ({file_size} bytes) exceeds maximum limit "
                    f"({max_size_bytes} bytes)"
                )

            # Ensure filename has .txt extension if not specified
            if not filename.endswith((".txt", ".md")):
                filename = f"{filename}.txt"

            # Generate unique object key
            object_key = self._generate_object_key(filename, user_id)

            # Create temporary file for upload
            temp_dir = tempfile.mkdtemp(prefix="metastamp_text_")
            temp_file_path = Path(temp_dir) / filename

            async with aiofiles.open(temp_file_path, "wb") as temp_file:
                await temp_file.write(content_bytes)

            # Extract text metadata
            extracted_metadata = {}
            try:
                extracted_metadata = await self.metadata.extract_text_metadata(str(temp_file_path))
            except Exception as metadata_error:
                self.logger.warning(
                    "Text metadata extraction failed: %s",
                    str(metadata_error),
                )
                extracted_metadata = {
                    "character_count": len(content),
                    "word_count": len(content.split()),
                    "line_count": content.count("\n") + 1,
                }

            # Upload to S3
            content_type = "text/plain; charset=utf-8"
            upload_success = await self.storage.upload_file(
                file_path=str(temp_file_path),
                object_key=object_key,
                content_type=content_type,
            )

            if not upload_success:
                self.logger.error(
                    "S3 upload failed for text content '%s'",
                    filename,
                )
                raise StorageError(f"Failed to upload text content {filename} to storage")

            self.logger.info(
                "Successfully uploaded text content '%s' to S3",
                filename,
            )

            # Create Asset record in MongoDB
            asset = Asset(
                _id=str(uuid.uuid4()),
                user_id=user_id,
                file_name=filename,
                file_type=FileType.TEXT,
                file_size=file_size,
                mime_type=content_type,
                s3_key=object_key,
                s3_bucket=self.settings.s3_bucket_name,
                upload_status=UploadStatus.PROCESSING,
                metadata=extracted_metadata,
                created_at=datetime.now(UTC),
            )

            # Save to MongoDB
            db_client = get_db_client()
            assets_collection = db_client.get_assets_collection()
            asset_dict = asset.model_dump()
            await assets_collection.insert_one(asset_dict)

            self.logger.info(
                "Created text asset record '%s' for '%s'",
                asset.id,
                filename,
            )

            return {
                "asset_id": asset.id,
                "file_name": asset.file_name,
                "file_type": str(asset.file_type),
                "file_size": asset.file_size,
                "mime_type": asset.mime_type,
                "s3_key": asset.s3_key,
                "upload_status": str(asset.upload_status),
                "metadata": extracted_metadata,
                "created_at": asset.created_at.isoformat(),
            }

        except FileValidationError:
            raise
        except StorageError:
            raise
        except Exception as error:
            self.logger.exception(
                "Error during text upload for '%s'",
                filename,
            )
            raise UploadServiceError(f"Text upload failed for {filename}: {error!s}") from error

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    temp_file_path.parent.rmdir()
                    self.logger.debug(
                        "Cleaned up temporary text file: %s",
                        temp_file_path,
                    )
                except OSError as cleanup_error:
                    self.logger.warning(
                        "Failed to clean up temporary file '%s': %s",
                        temp_file_path,
                        str(cleanup_error),
                    )

    async def handle_url_upload(
        self,
        url: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Handle URL-based content upload.

        Processes URL-based uploads for various platforms:
        - YouTube: Extracts transcript, metadata (title, description, views)
        - Vimeo: Extracts video metadata
        - General webpages: Extracts text content and metadata

        The method:
        1. Validates URL format and safety
        2. Detects platform (YouTube, Vimeo, or general webpage)
        3. Processes URL using URLProcessorService
        4. Stores extracted content/metadata in S3
        5. Creates Asset record with URL reference
        6. Returns asset details

        Args:
            url: URL to process and import as an asset.
            user_id: User identifier for the asset owner.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - asset_id: Unique identifier for the created asset
                - file_name: Generated filename based on URL content
                - file_type: "url"
                - platform: Detected platform (youtube, vimeo, webpage)
                - source_url: Original URL
                - s3_key: Object key where extracted content is stored
                - upload_status: Current status ("processing")
                - metadata: Extracted URL metadata
                - created_at: ISO timestamp of creation

        Raises:
            FileValidationError: If URL validation fails or points to dangerous content.
            URLProcessingError: If URL processing fails.
            StorageError: If S3 upload fails.

        Example:
            ```python
            result = await upload_service.handle_url_upload(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                user_id="user_12345"
            )
            print(f"Platform: {result['platform']}")  # "youtube"
            ```
        """
        self.logger.info(
            "Handling URL upload for '%s', user='%s'",
            url,
            user_id,
        )

        temp_file_path: Path | None = None

        try:
            # Validate URL format and safety
            is_valid_url = self.url_processor.validate_url(url)
            if not is_valid_url:
                self.logger.warning(
                    "URL validation failed for '%s'",
                    url,
                )
                raise FileValidationError(f"Invalid or unsafe URL: {url}")

            # Detect platform
            platform = self.url_processor.detect_platform(url)
            self.logger.info(
                "Detected platform '%s' for URL '%s'",
                platform,
                url,
            )

            # Process URL to extract content
            url_result = await self.url_processor.process_url(url)

            if not url_result or not url_result.get("success"):
                error_msg = url_result.get("error", "Unknown error") if url_result else "No result"
                self.logger.error(
                    "URL processing failed for '%s': %s",
                    url,
                    error_msg,
                )
                raise URLProcessingError(f"Failed to process URL {url}: {error_msg}")

            # Extract content and metadata from result
            extracted_content = url_result.get("content", "")
            extracted_metadata = url_result.get("metadata", {})
            title = extracted_metadata.get("title", "untitled")

            # Generate filename based on content type and title
            safe_title = (
                title[:50].replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
            )
            filename = f"{platform}_{safe_title}.txt"

            # Store extracted content to S3
            content_bytes = extracted_content.encode("utf-8")
            file_size = len(content_bytes)

            # Generate unique object key
            object_key = self._generate_object_key(filename, user_id)

            # Create temporary file for upload
            temp_dir = tempfile.mkdtemp(prefix="metastamp_url_")
            temp_file_path = Path(temp_dir) / filename

            async with aiofiles.open(temp_file_path, "wb") as temp_file:
                await temp_file.write(content_bytes)

            # Upload to S3
            content_type = "text/plain; charset=utf-8"
            upload_success = await self.storage.upload_file(
                file_path=str(temp_file_path),
                object_key=object_key,
                content_type=content_type,
            )

            if not upload_success:
                self.logger.error(
                    "S3 upload failed for URL content from '%s'",
                    url,
                )
                raise StorageError("Failed to upload URL content to storage")

            self.logger.info(
                "Successfully uploaded URL content to S3 with key '%s'",
                object_key,
            )

            # Combine URL metadata with extracted metadata
            combined_metadata = {
                **extracted_metadata,
                "source_url": url,
                "platform": platform,
                "content_length": len(extracted_content),
                "extracted_at": datetime.now(UTC).isoformat(),
            }

            # Create Asset record in MongoDB
            asset = Asset(
                _id=str(uuid.uuid4()),
                user_id=user_id,
                file_name=filename,
                file_type=FileType.URL,
                file_size=file_size,
                mime_type=content_type,
                s3_key=object_key,
                s3_bucket=self.settings.s3_bucket_name,
                upload_status=UploadStatus.PROCESSING,
                metadata=combined_metadata,
                url_source=url,
                created_at=datetime.now(UTC),
            )

            # Save to MongoDB
            db_client = get_db_client()
            assets_collection = db_client.get_assets_collection()
            asset_dict = asset.model_dump()
            await assets_collection.insert_one(asset_dict)

            self.logger.info(
                "Created URL asset record '%s' for '%s'",
                asset.id,
                url,
            )

            return {
                "asset_id": asset.id,
                "file_name": asset.file_name,
                "file_type": str(asset.file_type),
                "file_size": asset.file_size,
                "platform": platform,
                "source_url": url,
                "s3_key": asset.s3_key,
                "upload_status": str(asset.upload_status),
                "metadata": combined_metadata,
                "created_at": asset.created_at.isoformat(),
            }

        except FileValidationError:
            raise
        except URLProcessingError:
            raise
        except StorageError:
            raise
        except Exception as error:
            self.logger.exception(
                "Error during URL upload for '%s'",
                url,
            )
            raise UploadServiceError(f"URL upload failed for {url}: {error!s}") from error

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    temp_file_path.parent.rmdir()
                    self.logger.debug(
                        "Cleaned up temporary URL content file: %s",
                        temp_file_path,
                    )
                except OSError as cleanup_error:
                    self.logger.warning(
                        "Failed to clean up temporary file '%s': %s",
                        temp_file_path,
                        str(cleanup_error),
                    )


# Module exports
__all__ = [
    "FileValidationError",
    "StorageError",
    "URLProcessingError",
    "UploadConfirmationError",
    "UploadService",
    "UploadServiceError",
]
