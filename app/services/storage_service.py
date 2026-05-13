"""
S3-compatible storage service for META-STAMP V3.

This module provides a cloud-agnostic storage service wrapping boto3 operations
for file storage, presigned URL generation, and multipart upload support.
Compatible with both MinIO (development) and AWS S3 (production) environments.

Key Features:
- Presigned URL generation with 15-minute expiration for secure uploads
- Multipart upload support for resumable large file transfers
- Standard file operations (upload, download, delete, exists, metadata)
- Async-wrapped operations for non-blocking I/O

Per Agent Action Plan:
- Section 0.3: Must use S3-compatible API via boto3
- Section 0.3: Must generate presigned URLs with 15-minute expiration
- Section 0.3: Must support multipart upload for resumable transfers
- Section 0.4: Must work with MinIO for development and AWS S3 for production
- Section 0.8: Storage operations must be async-wrapped
"""

import asyncio
import logging

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, BinaryIO, TypeVar

import boto3

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


# Set up module-level logger for tracking S3 operations
logger = logging.getLogger(__name__)

# Type variable for generic async wrapper
T = TypeVar("T")

# Default presigned URL expiration time (15 minutes = 900 seconds)
DEFAULT_PRESIGNED_URL_EXPIRATION = 900

# Default multipart upload part size (5 MB minimum for S3)
DEFAULT_PART_SIZE = 5 * 1024 * 1024  # 5 MB


def async_wrap(func: Callable[..., T]) -> Callable[..., "Coroutine[Any, Any, T]"]:
    """
    Decorator to wrap synchronous boto3 operations for async execution.

    Uses asyncio.to_thread to run blocking boto3 operations in a separate
    thread pool, preventing event loop blocking during S3 operations.

    Args:
        func: The synchronous function to wrap

    Returns:
        An async function that executes the original in a thread pool
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


class StorageServiceError(Exception):
    """Base exception for storage service errors."""


class StorageConnectionError(StorageServiceError):
    """Raised when connection to storage backend fails."""


class StorageCredentialsError(StorageServiceError):
    """Raised when storage credentials are missing or invalid."""


class StorageOperationError(StorageServiceError):
    """Raised when a storage operation fails."""


class StorageNotFoundError(StorageServiceError):
    """Raised when a requested object does not exist."""


class StorageService:
    """
    S3-compatible storage service for cloud-agnostic file operations.

    This service wraps boto3 operations and provides async methods for all
    storage operations. It is designed to work with both MinIO (development)
    and AWS S3 (production) through configurable endpoint URLs.

    Attributes:
        bucket_name: The default S3 bucket name for operations
        endpoint_url: The S3-compatible endpoint URL (MinIO or AWS S3)
        region_name: AWS region name (optional, defaults to us-east-1)

    Example:
        >>> service = StorageService(
        ...     bucket_name="meta-stamp-assets",
        ...     endpoint_url="http://localhost:9000",  # MinIO
        ...     access_key="your-minio-access-key",
        ...     secret_key="your-minio-secret-key"
        ... )
        >>> url = await service.generate_presigned_upload_url("uploads/test.jpg")
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region_name: str = "us-east-1",
        use_ssl: bool = True,
    ) -> None:
        """
        Initialize the S3-compatible storage service.

        Args:
            bucket_name: Default bucket name for all operations
            endpoint_url: S3-compatible endpoint URL (None for AWS S3 default)
                         Use http://localhost:9000 for MinIO development
            access_key: AWS access key ID or MinIO access key
            secret_key: AWS secret access key or MinIO secret key
            region_name: AWS region (default: us-east-1)
            use_ssl: Whether to use SSL for connections (default: True)

        Raises:
            StorageCredentialsError: If credentials are missing or invalid
            StorageConnectionError: If connection to storage backend fails
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name

        logger.info(
            f"Initializing StorageService with bucket={bucket_name}, "
            f"endpoint={endpoint_url or 'AWS S3 default'}"
        )

        try:
            # Build S3 client configuration
            client_config: dict[str, Any] = {
                "service_name": "s3",
                "region_name": region_name,
                "use_ssl": use_ssl,
            }

            # Add endpoint URL for MinIO or custom S3-compatible storage
            if endpoint_url:
                client_config["endpoint_url"] = endpoint_url

            # Add credentials if provided (otherwise use environment/IAM role)
            if access_key and secret_key:
                client_config["aws_access_key_id"] = access_key
                client_config["aws_secret_access_key"] = secret_key

            # Create the boto3 S3 client
            self._client = boto3.client(**client_config)

            logger.info("StorageService S3 client initialized successfully")

        except NoCredentialsError as e:
            error_msg = (
                "S3 credentials not found. Please configure AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables or provide credentials directly."
            )
            logger.exception(f"Credential configuration error: {error_msg}")
            raise StorageCredentialsError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Failed to initialize S3 client: {e!s}"
            logger.exception(error_msg)
            raise StorageConnectionError(error_msg) from e

    async def generate_presigned_upload_url(
        self,
        object_key: str,
        content_type: str | None = None,
        expiration: int = DEFAULT_PRESIGNED_URL_EXPIRATION,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a presigned URL for uploading files directly to S3.

        Creates a presigned PUT URL that allows clients to upload files
        directly to S3/MinIO without routing through the backend server.
        Default expiration is 15 minutes per Agent Action Plan requirements.

        Args:
            object_key: The S3 object key (path) for the uploaded file
            content_type: Optional MIME type for Content-Type header restriction
            expiration: URL expiration time in seconds (default: 900 = 15 minutes)
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - url: The presigned URL for PUT upload
                - object_key: The S3 object key
                - bucket: The bucket name
                - expiration_seconds: Time until URL expires
                - method: HTTP method (PUT)

        Raises:
            StorageOperationError: If URL generation fails

        Example:
            >>> result = await service.generate_presigned_upload_url(
            ...     "uploads/2024/01/image.jpg",
            ...     content_type="image/jpeg"
            ... )
            >>> print(result["url"])  # Use this URL for direct S3 upload
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(
            f"Generating presigned upload URL for object_key={object_key}, "
            f"bucket={target_bucket}, expiration={expiration}s"
        )

        try:
            # Build presigned URL parameters
            params: dict[str, Any] = {
                "Bucket": target_bucket,
                "Key": object_key,
            }

            # Add content type condition if specified
            if content_type:
                params["ContentType"] = content_type

            # Generate the presigned URL asynchronously
            @async_wrap
            def _generate() -> str:
                url: str = self._client.generate_presigned_url(
                    ClientMethod="put_object",
                    Params=params,
                    ExpiresIn=expiration,
                    HttpMethod="PUT",
                )
                return url

            presigned_url = await _generate()

            logger.info(f"Successfully generated presigned upload URL for {object_key}")

            result = {
                "url": presigned_url,
                "object_key": object_key,
                "bucket": target_bucket,
                "expiration_seconds": expiration,
                "method": "PUT",
            }

        except ClientError as e:
            error_msg = f"Failed to generate presigned upload URL: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during presigned URL generation: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def generate_presigned_download_url(
        self,
        object_key: str,
        expiration: int = DEFAULT_PRESIGNED_URL_EXPIRATION,
        bucket_name: str | None = None,
        response_content_type: str | None = None,
        response_content_disposition: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a presigned URL for downloading files from S3.

        Creates a presigned GET URL that allows clients to download files
        directly from S3/MinIO without routing through the backend server.

        Args:
            object_key: The S3 object key (path) of the file to download
            expiration: URL expiration time in seconds (default: 900 = 15 minutes)
            bucket_name: Optional bucket override (defaults to service bucket)
            response_content_type: Override Content-Type in response headers
            response_content_disposition: Override Content-Disposition for downloads

        Returns:
            Dictionary containing:
                - url: The presigned URL for GET download
                - object_key: The S3 object key
                - bucket: The bucket name
                - expiration_seconds: Time until URL expires
                - method: HTTP method (GET)

        Raises:
            StorageOperationError: If URL generation fails

        Example:
            >>> result = await service.generate_presigned_download_url(
            ...     "uploads/2024/01/image.jpg",
            ...     response_content_disposition="attachment; filename=download.jpg"
            ... )
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(
            f"Generating presigned download URL for object_key={object_key}, "
            f"bucket={target_bucket}, expiration={expiration}s"
        )

        try:
            # Build presigned URL parameters
            params: dict[str, Any] = {
                "Bucket": target_bucket,
                "Key": object_key,
            }

            # Add response header overrides if specified
            if response_content_type:
                params["ResponseContentType"] = response_content_type
            if response_content_disposition:
                params["ResponseContentDisposition"] = response_content_disposition

            # Generate the presigned URL asynchronously
            @async_wrap
            def _generate() -> str:
                url: str = self._client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params=params,
                    ExpiresIn=expiration,
                    HttpMethod="GET",
                )
                return url

            presigned_url = await _generate()

            logger.info(f"Successfully generated presigned download URL for {object_key}")

            result = {
                "url": presigned_url,
                "object_key": object_key,
                "bucket": target_bucket,
                "expiration_seconds": expiration,
                "method": "GET",
            }

        except ClientError as e:
            error_msg = (
                f"Failed to generate presigned download URL: {e.response['Error']['Message']}"
            )
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during presigned URL generation: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def initiate_multipart_upload(
        self,
        object_key: str,
        content_type: str | None = None,
        bucket_name: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Initiate a multipart upload for large files.

        Multipart uploads allow resumable transfers for files larger than 10MB.
        The upload ID returned must be used for all subsequent upload_part and
        complete_multipart_upload calls.

        Args:
            object_key: The S3 object key (path) for the file
            content_type: MIME type of the file being uploaded
            bucket_name: Optional bucket override (defaults to service bucket)
            metadata: Optional metadata key-value pairs to attach to object

        Returns:
            Dictionary containing:
                - upload_id: Unique identifier for this multipart upload
                - object_key: The S3 object key
                - bucket: The bucket name

        Raises:
            StorageOperationError: If multipart upload initiation fails

        Example:
            >>> result = await service.initiate_multipart_upload(
            ...     "uploads/large_video.mp4",
            ...     content_type="video/mp4"
            ... )
            >>> upload_id = result["upload_id"]
            >>> # Use upload_id for upload_part and complete_multipart_upload
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(
            f"Initiating multipart upload for object_key={object_key}, bucket={target_bucket}"
        )

        try:
            # Build multipart upload request parameters
            params: dict[str, Any] = {
                "Bucket": target_bucket,
                "Key": object_key,
            }

            if content_type:
                params["ContentType"] = content_type

            if metadata:
                params["Metadata"] = metadata

            # Execute multipart upload initiation asynchronously
            @async_wrap
            def _initiate() -> dict[str, Any]:
                response: dict[str, Any] = self._client.create_multipart_upload(**params)
                return response

            response = await _initiate()
            upload_id = response["UploadId"]

            logger.info(
                f"Successfully initiated multipart upload for {object_key}, "
                f"upload_id={upload_id}"
            )

            result = {
                "upload_id": upload_id,
                "object_key": object_key,
                "bucket": target_bucket,
            }

        except ClientError as e:
            error_msg = f"Failed to initiate multipart upload: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during multipart initiation: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def upload_part(
        self,
        object_key: str,
        upload_id: str,
        part_number: int,
        body: bytes | BinaryIO,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload a single part of a multipart upload.

        Each part must be at least 5MB (except the last part). Part numbers
        must be sequential starting from 1. The returned ETag must be stored
        for use in complete_multipart_upload.

        Args:
            object_key: The S3 object key (path) for the file
            upload_id: The upload ID from initiate_multipart_upload
            part_number: Sequential part number (1-10000)
            body: The part data as bytes or file-like object
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - etag: ETag identifier for this part (required for completion)
                - part_number: The part number uploaded

        Raises:
            StorageOperationError: If part upload fails

        Example:
            >>> # After initiating multipart upload
            >>> part_data = read_chunk_from_file(file, chunk_size=5*1024*1024)
            >>> result = await service.upload_part(
            ...     "uploads/large_video.mp4",
            ...     upload_id="abc123",
            ...     part_number=1,
            ...     body=part_data
            ... )
            >>> parts.append({"PartNumber": 1, "ETag": result["etag"]})
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(
            f"Uploading part {part_number} for object_key={object_key}, upload_id={upload_id}"
        )

        try:
            # Execute part upload asynchronously
            @async_wrap
            def _upload() -> dict[str, Any]:
                response: dict[str, Any] = self._client.upload_part(
                    Bucket=target_bucket,
                    Key=object_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=body,
                )
                return response

            response = await _upload()
            etag = response["ETag"]

            logger.info(f"Successfully uploaded part {part_number} for {object_key}, ETag={etag}")

            result = {
                "etag": etag,
                "part_number": part_number,
            }

        except ClientError as e:
            error_msg = f"Failed to upload part {part_number}: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during part upload: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def complete_multipart_upload(
        self,
        object_key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete a multipart upload by assembling all uploaded parts.

        All parts must have been uploaded successfully before calling this method.
        Parts are assembled in order by part number to create the final object.

        Args:
            object_key: The S3 object key (path) for the file
            upload_id: The upload ID from initiate_multipart_upload
            parts: List of dictionaries with "PartNumber" and "ETag" for each part
                  Format: [{"PartNumber": 1, "ETag": "abc..."}, ...]
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - location: URL of the completed object
                - etag: ETag of the completed object
                - object_key: The S3 object key
                - bucket: The bucket name

        Raises:
            StorageOperationError: If completion fails

        Example:
            >>> # After uploading all parts
            >>> parts = [
            ...     {"PartNumber": 1, "ETag": "etag1"},
            ...     {"PartNumber": 2, "ETag": "etag2"},
            ... ]
            >>> result = await service.complete_multipart_upload(
            ...     "uploads/large_video.mp4",
            ...     upload_id="abc123",
            ...     parts=parts
            ... )
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(
            f"Completing multipart upload for object_key={object_key}, "
            f"upload_id={upload_id}, parts_count={len(parts)}"
        )

        try:
            # Sort parts by part number to ensure correct assembly
            sorted_parts = sorted(parts, key=lambda x: x.get("PartNumber", 0))

            # Execute multipart completion asynchronously
            @async_wrap
            def _complete() -> dict[str, Any]:
                response: dict[str, Any] = self._client.complete_multipart_upload(
                    Bucket=target_bucket,
                    Key=object_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": sorted_parts},
                )
                return response

            response = await _complete()

            logger.info(
                f"Successfully completed multipart upload for {object_key}, "
                f"ETag={response.get('ETag')}"
            )

            return {
                "location": response.get("Location", ""),
                "etag": response.get("ETag", ""),
                "object_key": object_key,
                "bucket": target_bucket,
            }

        except ClientError as e:
            error_msg = f"Failed to complete multipart upload: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during multipart completion: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

    async def abort_multipart_upload(
        self,
        object_key: str,
        upload_id: str,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Abort a multipart upload and clean up uploaded parts.

        Cancels the multipart upload and deletes all previously uploaded parts.
        Should be called if upload fails or is cancelled by the user.

        Args:
            object_key: The S3 object key (path) for the file
            upload_id: The upload ID from initiate_multipart_upload
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - success: Boolean indicating successful abort
                - object_key: The S3 object key
                - upload_id: The aborted upload ID

        Raises:
            StorageOperationError: If abort operation fails

        Example:
            >>> # If upload fails or is cancelled
            >>> result = await service.abort_multipart_upload(
            ...     "uploads/large_video.mp4",
            ...     upload_id="abc123"
            ... )
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(f"Aborting multipart upload for object_key={object_key}, upload_id={upload_id}")

        try:
            # Execute abort operation asynchronously
            @async_wrap
            def _abort() -> None:
                self._client.abort_multipart_upload(
                    Bucket=target_bucket,
                    Key=object_key,
                    UploadId=upload_id,
                )

            await _abort()

            logger.info(
                f"Successfully aborted multipart upload for {object_key}, upload_id={upload_id}"
            )

            result = {
                "success": True,
                "object_key": object_key,
                "upload_id": upload_id,
            }

        except ClientError as e:
            error_msg = f"Failed to abort multipart upload: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during multipart abort: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def upload_file(
        self,
        object_key: str,
        file_path: str | None = None,
        file_data: bytes | BinaryIO | None = None,
        content_type: str | None = None,
        bucket_name: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Upload a file to S3 storage.

        Supports uploading from either a local file path or raw data (bytes/file object).
        Use this for direct uploads of files smaller than 10MB per Agent Action Plan.

        Args:
            object_key: The S3 object key (path) for the uploaded file
            file_path: Local file system path to upload
            file_data: Raw bytes or file-like object to upload
            content_type: MIME type of the file
            bucket_name: Optional bucket override (defaults to service bucket)
            metadata: Optional metadata key-value pairs to attach to object

        Returns:
            Dictionary containing:
                - success: Boolean indicating successful upload
                - object_key: The S3 object key
                - bucket: The bucket name
                - etag: ETag of the uploaded object

        Raises:
            StorageOperationError: If upload fails
            ValueError: If neither file_path nor file_data is provided

        Example:
            >>> # Upload from file path
            >>> result = await service.upload_file(
            ...     "uploads/image.jpg",
            ...     file_path="/tmp/image.jpg",
            ...     content_type="image/jpeg"
            ... )
            >>>
            >>> # Or upload from bytes
            >>> result = await service.upload_file(
            ...     "uploads/text.txt",
            ...     file_data=b"Hello, World!",
            ...     content_type="text/plain"
            ... )
        """
        if not file_path and file_data is None:
            raise ValueError("Either file_path or file_data must be provided")

        target_bucket = bucket_name or self.bucket_name

        logger.info(f"Uploading file to object_key={object_key}, bucket={target_bucket}")

        try:
            # Build extra arguments for upload
            extra_args: dict[str, Any] = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = metadata

            if file_path:
                # Upload from local file path
                @async_wrap
                def _upload_file() -> None:
                    self._client.upload_file(
                        file_path,
                        target_bucket,
                        object_key,
                        ExtraArgs=extra_args if extra_args else None,
                    )

                await _upload_file()

            else:
                # Upload from bytes or file-like object
                @async_wrap
                def _put_object() -> dict[str, Any]:
                    params: dict[str, Any] = {
                        "Bucket": target_bucket,
                        "Key": object_key,
                        "Body": file_data,
                    }
                    params.update(extra_args)
                    response: dict[str, Any] = self._client.put_object(**params)
                    return response

                await _put_object()

            # Get the ETag of uploaded object
            metadata_result = await self.get_file_metadata(object_key, bucket_name)

            logger.info(f"Successfully uploaded file to {object_key}")

            return {
                "success": True,
                "object_key": object_key,
                "bucket": target_bucket,
                "etag": metadata_result.get("etag", ""),
            }

        except ClientError as e:
            error_msg = f"Failed to upload file: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during file upload: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except OSError as e:
            error_msg = f"File system error during upload: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

    async def download_file(
        self,
        object_key: str,
        file_path: str | None = None,
        bucket_name: str | None = None,
    ) -> dict[str, Any] | bytes:
        """
        Download a file from S3 storage.

        Can either download to a local file path or return the file content as bytes.

        Args:
            object_key: The S3 object key (path) of the file to download
            file_path: Optional local file system path to save the file
                      If None, file content is returned as bytes
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            If file_path provided: Dictionary with success status and path
            If file_path is None: Raw bytes of the file content

        Raises:
            StorageNotFoundError: If the object does not exist
            StorageOperationError: If download fails

        Example:
            >>> # Download to file
            >>> result = await service.download_file(
            ...     "uploads/image.jpg",
            ...     file_path="/tmp/downloaded.jpg"
            ... )
            >>>
            >>> # Or get bytes directly
            >>> content = await service.download_file("uploads/text.txt")
            >>> print(content.decode("utf-8"))
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(f"Downloading file from object_key={object_key}, bucket={target_bucket}")

        try:
            if file_path:
                # Download to local file path
                @async_wrap
                def _download_file() -> None:
                    self._client.download_file(
                        target_bucket,
                        object_key,
                        file_path,
                    )

                await _download_file()

                logger.info(f"Successfully downloaded {object_key} to {file_path}")

                result: dict[str, Any] | bytes = {
                    "success": True,
                    "object_key": object_key,
                    "file_path": file_path,
                }
            else:
                # Download and return as bytes
                @async_wrap
                def _get_object() -> dict[str, Any]:
                    response: dict[str, Any] = self._client.get_object(
                        Bucket=target_bucket,
                        Key=object_key,
                    )
                    return response

                response = await _get_object()
                content = response["Body"].read()

                logger.info(f"Successfully downloaded {object_key} as bytes")

                result = content

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                error_msg = f"Object not found: {object_key}"
                logger.warning(error_msg)
                raise StorageNotFoundError(error_msg) from e
            error_msg = f"Failed to download file: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during file download: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except OSError as e:
            error_msg = f"File system error during download: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def delete_file(
        self,
        object_key: str,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete a file from S3 storage.

        Permanently removes the object from the specified bucket.
        Note: S3 delete operations succeed even if the object doesn't exist.

        Args:
            object_key: The S3 object key (path) of the file to delete
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - success: Boolean indicating operation completed
                - object_key: The S3 object key
                - bucket: The bucket name

        Raises:
            StorageOperationError: If delete operation fails

        Example:
            >>> result = await service.delete_file("uploads/old_image.jpg")
            >>> print(result["success"])  # True
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(f"Deleting file at object_key={object_key}, bucket={target_bucket}")

        try:

            @async_wrap
            def _delete() -> dict[str, Any]:
                response: dict[str, Any] = self._client.delete_object(
                    Bucket=target_bucket,
                    Key=object_key,
                )
                return response

            await _delete()

            logger.info(f"Successfully deleted {object_key}")

            result = {
                "success": True,
                "object_key": object_key,
                "bucket": target_bucket,
            }

        except ClientError as e:
            error_msg = f"Failed to delete file: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during file deletion: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return result

    async def file_exists(
        self,
        object_key: str,
        bucket_name: str | None = None,
    ) -> bool:
        """
        Check if a file exists in S3 storage.

        Uses head_object to efficiently check existence without downloading content.
        This is used to verify successful presigned URL uploads before creating
        asset records in MongoDB.

        Args:
            object_key: The S3 object key (path) to check
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            True if object exists, False otherwise

        Raises:
            StorageOperationError: If the check operation fails (not including 404)

        Example:
            >>> exists = await service.file_exists("uploads/image.jpg")
            >>> if exists:
            ...     print("File was uploaded successfully")
        """
        target_bucket = bucket_name or self.bucket_name

        logger.debug(f"Checking if file exists at object_key={object_key}, bucket={target_bucket}")

        try:

            @async_wrap
            def _head() -> dict[str, Any]:
                response: dict[str, Any] = self._client.head_object(
                    Bucket=target_bucket,
                    Key=object_key,
                )
                return response

            await _head()

            logger.debug(f"File exists at {object_key}")
            exists = True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                logger.debug(f"File does not exist at {object_key}")
                exists = False
            else:
                error_msg = f"Failed to check file existence: {e.response['Error']['Message']}"
                logger.exception(error_msg)
                raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during existence check: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        return exists

    async def get_file_metadata(
        self,
        object_key: str,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get metadata for a file in S3 storage.

        Retrieves object metadata including size, content type, ETag, and
        custom metadata without downloading the file content.

        Args:
            object_key: The S3 object key (path) to get metadata for
            bucket_name: Optional bucket override (defaults to service bucket)

        Returns:
            Dictionary containing:
                - object_key: The S3 object key
                - bucket: The bucket name
                - content_type: MIME type of the object
                - content_length: Size in bytes
                - etag: ETag identifier
                - last_modified: Last modification timestamp (ISO format)
                - metadata: Custom metadata dictionary

        Raises:
            StorageNotFoundError: If the object does not exist
            StorageOperationError: If metadata retrieval fails

        Example:
            >>> metadata = await service.get_file_metadata("uploads/image.jpg")
            >>> print(f"File size: {metadata['content_length']} bytes")
            >>> print(f"Content type: {metadata['content_type']}")
        """
        target_bucket = bucket_name or self.bucket_name

        logger.info(f"Getting metadata for object_key={object_key}, bucket={target_bucket}")

        try:

            @async_wrap
            def _head() -> dict[str, Any]:
                response: dict[str, Any] = self._client.head_object(
                    Bucket=target_bucket,
                    Key=object_key,
                )
                return response

            response = await _head()

            # Extract and format metadata
            last_modified = response.get("LastModified")
            last_modified_iso = last_modified.isoformat() if last_modified else None

            file_metadata = {
                "object_key": object_key,
                "bucket": target_bucket,
                "content_type": response.get("ContentType", "application/octet-stream"),
                "content_length": response.get("ContentLength", 0),
                "etag": response.get("ETag", "").strip('"'),
                "last_modified": last_modified_iso,
                "metadata": response.get("Metadata", {}),
            }

            logger.info(f"Successfully retrieved metadata for {object_key}")

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                error_msg = f"Object not found: {object_key}"
                logger.warning(error_msg)
                raise StorageNotFoundError(error_msg) from e
            error_msg = f"Failed to get file metadata: {e.response['Error']['Message']}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        except BotoCoreError as e:
            error_msg = f"Storage operation error during metadata retrieval: {e!s}"
            logger.exception(error_msg)
            raise StorageOperationError(error_msg) from e

        else:
            return file_metadata


# Export custom exceptions for use by other modules
__all__ = [
    "StorageConnectionError",
    "StorageCredentialsError",
    "StorageNotFoundError",
    "StorageOperationError",
    "StorageService",
    "StorageServiceError",
]
