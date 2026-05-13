"""
META-STAMP V3 Cloud-Agnostic S3-Compatible Storage Client

This module provides a comprehensive storage abstraction layer using boto3 that supports
both MinIO (for development) and AWS S3 (for production) through configurable endpoint URLs.
The implementation follows the Agent Action Plan requirements for cloud-agnostic design,
ensuring no vendor lock-in while providing enterprise-grade storage capabilities.

Key Features:
- Presigned URL generation for hybrid upload architecture (15-minute default expiration)
- Multipart upload support for large files (>10MB) with resumable capability
- Standard file operations (upload, download, delete, exists)
- Comprehensive error handling with structured logging
- Singleton pattern for resource efficiency

All S3 operations are performed through the generic S3 interface, ensuring
compatibility with any S3-compatible storage provider (AWS S3, MinIO, Ceph, etc.)
per Agent Action Plan section 0.3 cloud-agnostic design requirements.
"""

import logging

from typing import Any

import boto3

from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings


# Configure module-level constants to avoid magic numbers
MIN_PRESIGNED_EXPIRATION_SECONDS = 60
MAX_PRESIGNED_EXPIRATION_SECONDS = 3600
MAX_DOWNLOAD_EXPIRATION_SECONDS = 86400
MIN_PART_NUMBER = 1
MAX_PART_NUMBER = 10000

# Configure module-level logger
logger = logging.getLogger(__name__)

# Singleton container for storage client instance
# Using a dict container allows modification without global statement
_singleton_container: dict[str, "StorageClient"] = {}


class StorageClient:
    """
    Cloud-agnostic S3-compatible storage client supporting both MinIO and AWS S3.

    This class provides a unified interface for all storage operations including
    presigned URL generation, multipart uploads, and standard file operations.
    It uses boto3's generic S3 interface to ensure compatibility with any
    S3-compatible storage provider without vendor lock-in.

    The client is initialized with configuration from the Settings class,
    supporting endpoint URL configuration for MinIO (development) or AWS S3
    (production) deployment scenarios.

    Attributes:
        settings: Application settings containing S3 configuration
        s3_client: Initialized boto3 S3 client
        bucket_name: Target bucket for all operations

    Example usage:
        ```python
        from app.core.storage import get_storage_client

        storage = get_storage_client()

        # Generate presigned upload URL
        upload_url = storage.generate_presigned_upload_url(
            key="assets/user123/image.png",
            content_type="image/png"
        )

        # Check if file exists
        exists = storage.file_exists("assets/user123/image.png")
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the S3 storage client with configuration from settings.

        Creates a boto3 S3 client configured for either MinIO (when endpoint_url
        is provided) or AWS S3 (when endpoint_url is None). The client is
        configured with appropriate signature version for presigned URL
        compatibility.

        Args:
            settings: Optional Settings instance. If None, creates a new Settings
                     instance from environment variables.

        Raises:
            ClientError: If S3 client initialization fails due to invalid credentials
                        or network issues.
        """
        self.settings = settings or get_settings()

        # Build client configuration with signature version for presigned URLs
        client_config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},  # Use path-style for MinIO compatibility
            retries={"max_attempts": 3, "mode": "standard"},
        )

        # Initialize boto3 S3 client with endpoint URL support
        # When endpoint_url is None, boto3 defaults to AWS S3
        # When endpoint_url is set (e.g., http://minio:9000), it connects to MinIO
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key_id,
                aws_secret_access_key=self.settings.s3_secret_access_key,
                region_name=self.settings.s3_region,
                config=client_config,
            )
            self.bucket_name = self.settings.s3_bucket_name

            logger.info(
                "S3 storage client initialized successfully",
                extra={
                    "bucket": self.bucket_name,
                    "region": self.settings.s3_region,
                    "endpoint": self.settings.s3_endpoint_url or "AWS S3 (default)",
                },
            )
        except ClientError:
            logger.exception(
                "Failed to initialize S3 storage client",
                extra={"endpoint": self.settings.s3_endpoint_url},
            )
            raise

    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int | None = None,
    ) -> str:
        """
        Generate a presigned PUT URL for direct client-to-S3 upload.

        Creates a presigned URL allowing clients to upload files directly to S3
        without going through the backend server. This is part of the hybrid
        upload architecture for files >10MB per Agent Action Plan section 0.4.

        The presigned URL enforces content-type restrictions to prevent
        unauthorized file type uploads and expires after the configured time
        (default 15 minutes per Agent Action Plan section 0.10).

        Args:
            key: The S3 object key (path) where the file will be stored.
                Example: "assets/user123/document.pdf"
            content_type: MIME type of the file being uploaded.
                Must match the Content-Type header in the upload request.
                Example: "application/pdf"
            expires_in: Optional expiration time in seconds. If None, uses the
                       configured default (presigned_url_expiration_seconds).
                       Must be between 60 and 3600 seconds.

        Returns:
            str: Presigned PUT URL that can be used for direct upload.

        Raises:
            ClientError: If URL generation fails due to invalid bucket or
                        permissions issues.
            ValueError: If expires_in is outside the valid range (60-3600 seconds).

        Example:
            ```python
            url = storage.generate_presigned_upload_url(
                key="assets/user123/image.png",
                content_type="image/png",
                expires_in=900  # 15 minutes
            )
            # Client can now PUT to this URL with Content-Type: image/png
            ```
        """
        # Use default expiration if not specified
        expiration = expires_in or self.settings.presigned_url_expiration_seconds

        # Validate expiration time is within acceptable bounds
        if not MIN_PRESIGNED_EXPIRATION_SECONDS <= expiration <= MAX_PRESIGNED_EXPIRATION_SECONDS:
            raise ValueError(
                f"expires_in must be between {MIN_PRESIGNED_EXPIRATION_SECONDS} and "
                f"{MAX_PRESIGNED_EXPIRATION_SECONDS} seconds, got {expiration}"
            )

        try:
            presigned_url: str = self.s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expiration,
            )

            logger.info(
                "Generated presigned upload URL",
                extra={
                    "key": key,
                    "content_type": content_type,
                    "expires_in": expiration,
                },
            )

            return presigned_url

        except ClientError:
            logger.exception(
                "Failed to generate presigned upload URL",
                extra={"key": key, "content_type": content_type},
            )
            raise

    def generate_presigned_download_url(
        self,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned GET URL for secure file download.

        Creates a presigned URL allowing clients to download files directly from S3
        without exposing bucket credentials. Useful for serving files to authenticated
        users or sharing temporary download links.

        Args:
            key: The S3 object key (path) of the file to download.
                Example: "assets/user123/document.pdf"
            expires_in: Expiration time in seconds. Default is 3600 (1 hour).
                       Must be between 60 and 86400 seconds.

        Returns:
            str: Presigned GET URL that can be used for download.

        Raises:
            ClientError: If URL generation fails due to invalid key or
                        permissions issues.
            ValueError: If expires_in is outside the valid range.

        Example:
            ```python
            url = storage.generate_presigned_download_url(
                key="assets/user123/image.png",
                expires_in=3600  # 1 hour
            )
            # Client can now GET from this URL to download the file
            ```
        """
        # Validate expiration time
        if not MIN_PRESIGNED_EXPIRATION_SECONDS <= expires_in <= MAX_DOWNLOAD_EXPIRATION_SECONDS:
            raise ValueError(
                f"expires_in must be between {MIN_PRESIGNED_EXPIRATION_SECONDS} and "
                f"{MAX_DOWNLOAD_EXPIRATION_SECONDS} seconds, got {expires_in}"
            )

        try:
            presigned_url: str = self.s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": key,
                },
                ExpiresIn=expires_in,
            )

            logger.info(
                "Generated presigned download URL",
                extra={"key": key, "expires_in": expires_in},
            )

            return presigned_url

        except ClientError:
            logger.exception(
                "Failed to generate presigned download URL",
                extra={"key": key},
            )
            raise

    def initiate_multipart_upload(
        self,
        key: str,
        content_type: str,
    ) -> str:
        """
        Initiate a multipart upload for large files.

        Starts a multipart upload session for files >10MB, allowing the upload
        to be split into multiple parts and uploaded in parallel or sequentially.
        This enables resumable uploads where individual parts can be retried
        without re-uploading the entire file per Agent Action Plan section 0.4.

        Args:
            key: The S3 object key where the complete file will be stored.
                Example: "assets/user123/large-video.mp4"
            content_type: MIME type of the file being uploaded.
                Example: "video/mp4"

        Returns:
            str: Upload ID required for uploading parts and completing the upload.
                This ID must be stored and used for all subsequent part uploads.

        Raises:
            ClientError: If multipart upload initiation fails due to bucket
                        permissions or network issues.

        Example:
            ```python
            upload_id = storage.initiate_multipart_upload(
                key="assets/user123/video.mp4",
                content_type="video/mp4"
            )
            # Use upload_id for subsequent part uploads
            ```
        """
        try:
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                ContentType=content_type,
            )
            upload_id: str = response["UploadId"]

            logger.info(
                "Initiated multipart upload",
                extra={"key": key, "upload_id": upload_id, "content_type": content_type},
            )

            return upload_id

        except ClientError:
            logger.exception(
                "Failed to initiate multipart upload",
                extra={"key": key, "content_type": content_type},
            )
            raise

    def generate_presigned_part_url(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        expires_in: int | None = None,
    ) -> str:
        """
        Generate a presigned URL for uploading a single part of a multipart upload.

        Creates a presigned URL allowing clients to upload a specific part
        directly to S3. Each part must be at least 5MB (except the last part)
        and at most 5GB per S3 multipart upload specifications.

        Args:
            key: The S3 object key for the multipart upload.
                Must match the key used in initiate_multipart_upload.
            upload_id: The upload ID returned from initiate_multipart_upload.
            part_number: Part number (1-10000). Parts can be uploaded in any order.
            expires_in: Optional expiration time in seconds. Uses default if None.

        Returns:
            str: Presigned PUT URL for uploading the specified part.
                The response will include an ETag that must be saved for
                completing the multipart upload.

        Raises:
            ClientError: If URL generation fails.
            ValueError: If part_number is outside valid range (1-10000).

        Example:
            ```python
            # Upload part 1
            url = storage.generate_presigned_part_url(
                key="assets/user123/video.mp4",
                upload_id="abc123xyz",
                part_number=1
            )
            # PUT data to URL, save ETag from response
            ```
        """
        # Validate part number
        if not MIN_PART_NUMBER <= part_number <= MAX_PART_NUMBER:
            raise ValueError(
                f"part_number must be between {MIN_PART_NUMBER} and {MAX_PART_NUMBER}, "
                f"got {part_number}"
            )

        expiration = expires_in or self.settings.presigned_url_expiration_seconds

        try:
            presigned_url: str = self.s3_client.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=expiration,
            )

            logger.debug(
                "Generated presigned part upload URL",
                extra={
                    "key": key,
                    "upload_id": upload_id,
                    "part_number": part_number,
                    "expires_in": expiration,
                },
            )

            return presigned_url

        except ClientError:
            logger.exception(
                "Failed to generate presigned part upload URL",
                extra={
                    "key": key,
                    "upload_id": upload_id,
                    "part_number": part_number,
                },
            )
            raise

    def complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> bool:
        """
        Complete a multipart upload by assembling all uploaded parts.

        Finalizes a multipart upload by combining all uploaded parts into
        the final object. The parts list must include the PartNumber and
        ETag for each uploaded part, in order by part number.

        Args:
            key: The S3 object key for the multipart upload.
            upload_id: The upload ID returned from initiate_multipart_upload.
            parts: List of dictionaries containing 'PartNumber' (int) and
                  'ETag' (str) for each uploaded part. Parts must be in
                  ascending order by PartNumber.
                  Example: [
                      {"PartNumber": 1, "ETag": "\"abc123\""},
                      {"PartNumber": 2, "ETag": "\"def456\""}
                  ]

        Returns:
            bool: True if the multipart upload completed successfully.

        Raises:
            ClientError: If completion fails due to missing parts,
                        invalid ETags, or network issues.
            ValueError: If parts list is empty or malformed.

        Example:
            ```python
            parts = [
                {"PartNumber": 1, "ETag": etag1},
                {"PartNumber": 2, "ETag": etag2},
            ]
            success = storage.complete_multipart_upload(
                key="assets/user123/video.mp4",
                upload_id="abc123xyz",
                parts=parts
            )
            ```
        """
        if not parts:
            raise ValueError("parts list cannot be empty")

        # Validate parts structure
        for part in parts:
            if "PartNumber" not in part or "ETag" not in part:
                raise ValueError("Each part must contain 'PartNumber' and 'ETag' keys")

        try:
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(
                "Completed multipart upload",
                extra={
                    "key": key,
                    "upload_id": upload_id,
                    "total_parts": len(parts),
                },
            )

            return True

        except ClientError:
            logger.exception(
                "Failed to complete multipart upload",
                extra={
                    "key": key,
                    "upload_id": upload_id,
                    "parts_count": len(parts),
                },
            )
            raise

    def abort_multipart_upload(
        self,
        key: str,
        upload_id: str,
    ) -> bool:
        """
        Abort an in-progress multipart upload and clean up uploaded parts.

        Cancels a multipart upload and deletes any parts that have already
        been uploaded. Use this to clean up after upload failures or
        cancellations to avoid storage charges for orphaned parts.

        Args:
            key: The S3 object key for the multipart upload.
            upload_id: The upload ID returned from initiate_multipart_upload.

        Returns:
            bool: True if the abort operation completed successfully.

        Raises:
            ClientError: If abort fails due to invalid upload ID or
                        network issues.

        Example:
            ```python
            # User cancels upload
            storage.abort_multipart_upload(
                key="assets/user123/video.mp4",
                upload_id="abc123xyz"
            )
            ```
        """
        try:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
            )

            logger.info(
                "Aborted multipart upload",
                extra={"key": key, "upload_id": upload_id},
            )

            return True

        except ClientError:
            logger.exception(
                "Failed to abort multipart upload",
                extra={"key": key, "upload_id": upload_id},
            )
            raise

    def upload_file(
        self,
        file_path: str,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """
        Upload a file from the local filesystem to S3.

        Performs a direct server-side upload of a file to S3. This is used
        for small files (<10MB) that go through the direct upload flow
        per the hybrid upload architecture in Agent Action Plan section 0.4.

        Args:
            file_path: Path to the local file to upload.
                Example: "/tmp/uploads/image.png"
            key: The S3 object key where the file will be stored.
                Example: "assets/user123/image.png"
            metadata: Optional dictionary of custom metadata to attach to
                     the object. Keys and values must be strings.
                     Example: {"user-id": "123", "original-name": "photo.png"}

        Returns:
            bool: True if the upload completed successfully.

        Raises:
            ClientError: If upload fails due to permissions, network, or
                        storage issues.
            FileNotFoundError: If the local file does not exist.

        Example:
            ```python
            success = storage.upload_file(
                file_path="/tmp/uploads/temp123.png",
                key="assets/user123/image.png",
                metadata={"user-id": "123"}
            )
            ```
        """
        extra_args: dict[str, Any] = {}
        if metadata:
            extra_args["Metadata"] = metadata

        try:
            self.s3_client.upload_file(
                Filename=file_path,
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs=extra_args if extra_args else None,
            )

            logger.info(
                "Uploaded file to S3",
                extra={"file_path": file_path, "key": key, "metadata": metadata},
            )

            return True

        except ClientError:
            logger.exception(
                "Failed to upload file to S3",
                extra={"file_path": file_path, "key": key},
            )
            raise

    def download_file(
        self,
        key: str,
        file_path: str,
    ) -> bool:
        """
        Download a file from S3 to the local filesystem.

        Retrieves an object from S3 and saves it to the specified local path.
        Useful for server-side processing of uploaded assets, such as
        fingerprint generation.

        Args:
            key: The S3 object key of the file to download.
                Example: "assets/user123/image.png"
            file_path: Local path where the file should be saved.
                Example: "/tmp/processing/image.png"

        Returns:
            bool: True if the download completed successfully.

        Raises:
            ClientError: If download fails due to missing key, permissions,
                        or network issues.
            IOError: If the local file cannot be written.

        Example:
            ```python
            success = storage.download_file(
                key="assets/user123/image.png",
                file_path="/tmp/processing/temp123.png"
            )
            ```
        """
        try:
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=key,
                Filename=file_path,
            )

            logger.info(
                "Downloaded file from S3",
                extra={"key": key, "file_path": file_path},
            )

            return True

        except ClientError:
            logger.exception(
                "Failed to download file from S3",
                extra={"key": key, "file_path": file_path},
            )
            raise

    def delete_file(
        self,
        key: str,
    ) -> bool:
        """
        Delete a file from S3.

        Removes an object from the S3 bucket. This operation is idempotent -
        deleting a non-existent key does not raise an error.

        Args:
            key: The S3 object key of the file to delete.
                Example: "assets/user123/image.png"

        Returns:
            bool: True if the delete operation completed (even if key didn't exist).

        Raises:
            ClientError: If delete fails due to permissions or network issues.

        Example:
            ```python
            success = storage.delete_file("assets/user123/image.png")
            ```
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key,
            )

            logger.info(
                "Deleted file from S3",
                extra={"key": key},
            )

            return True

        except ClientError:
            logger.exception(
                "Failed to delete file from S3",
                extra={"key": key},
            )
            raise

    def file_exists(
        self,
        key: str,
    ) -> bool:
        """
        Check if a file exists in S3.

        Uses HEAD request to efficiently check object existence without
        downloading the file content. This is used by the upload confirmation
        endpoint to verify S3 upload success before creating MongoDB records
        per Agent Action Plan section 0.4.

        Args:
            key: The S3 object key to check.
                Example: "assets/user123/image.png"

        Returns:
            bool: True if the object exists, False otherwise.

        Example:
            ```python
            if storage.file_exists("assets/user123/image.png"):
                # File was uploaded successfully
                create_asset_record(...)
            ```
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key,
            )

            logger.debug(
                "File exists in S3",
                extra={"key": key},
            )

            return True

        except ClientError as e:
            # 404 means file doesn't exist - this is expected behavior
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey"}:
                logger.debug(
                    "File does not exist in S3",
                    extra={"key": key},
                )
                return False

            # Other errors should be logged and re-raised
            logger.exception(
                "Failed to check file existence in S3",
                extra={"key": key},
            )
            raise


def get_storage_client() -> StorageClient:
    """
    Get the singleton StorageClient instance.

    Returns a shared StorageClient instance, creating it if it doesn't exist.
    This function implements the singleton pattern to ensure efficient
    resource usage by reusing the same S3 client connection across the
    application.

    The singleton pattern is appropriate here because:
    - The S3 client is thread-safe and supports concurrent operations
    - Creating multiple clients wastes resources and connections
    - Configuration is loaded once from Settings

    Returns:
        StorageClient: The shared storage client instance.

    Example:
        ```python
        from app.core.storage import get_storage_client

        storage = get_storage_client()
        url = storage.generate_presigned_upload_url(
            key="assets/image.png",
            content_type="image/png"
        )
        ```
    """
    if "instance" not in _singleton_container:
        _singleton_container["instance"] = StorageClient()
        logger.info("Created new StorageClient singleton instance")

    return _singleton_container["instance"]
