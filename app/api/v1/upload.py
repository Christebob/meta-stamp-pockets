"""
FastAPI Upload Router for META-STAMP V3

This module implements the hybrid upload architecture with 7 endpoints:
- POST /text - Direct text content upload (<10MB)
- POST /image - Direct image upload (<10MB)
- POST /audio - Direct audio upload (<10MB)
- POST /video - Direct video upload (<10MB or presigned URL flow redirect)
- POST /url - URL-based content import (YouTube, Vimeo, general web)
- GET /presigned-url - Generate S3 presigned PUT URL for files >=10MB
- POST /confirmation - Validate and register S3 uploads after presigned upload

Per Agent Action Plan:
- Section 0.3: Hybrid upload architecture (direct <10MB, presigned >=10MB)
- Section 0.3: File size limit 500MB max, reject ZIP/executables
- Section 0.4: Upload endpoint implementation details
- Section 0.6: upload.py transformation requirements
- Section 0.8: Endpoint specifications with error handling
- Section 0.10: Presigned URLs expire within 15 minutes

Author: META-STAMP V3 Platform
License: Proprietary
"""

import logging

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.models.asset import UploadStatus
from app.models.user import User
from app.services.fingerprinting_service import FingerprintingService
from app.services.storage_service import StorageService
from app.services.upload_service import UploadService
from app.utils.file_validator import validate_url


# Configure module logger
logger = logging.getLogger(__name__)

# Constants for upload thresholds and limits
DIRECT_UPLOAD_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10MB threshold for direct upload
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB maximum file size
PRESIGNED_URL_EXPIRATION_SECONDS = 900  # 15 minutes

# Allowed file extensions by type
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".pdf"}


# ============================================================================
# Request/Response Pydantic Models
# ============================================================================


class PresignedUrlRequest(BaseModel):
    """
    Request model for generating S3 presigned upload URL.

    Used by GET /presigned-url endpoint to generate a signed URL
    for direct client-to-S3 upload of files >= 10MB.
    """

    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename with extension",
        examples=["my_video.mp4", "presentation.pdf"],
    )
    content_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="MIME type of the file",
        examples=["video/mp4", "image/png"],
    )
    file_size: int = Field(
        ...,
        ge=0,
        le=MAX_FILE_SIZE_BYTES,
        description="File size in bytes (max 500MB)",
    )


class PresignedUrlResponse(BaseModel):
    """
    Response model for presigned URL generation.

    Contains the signed S3 PUT URL, associated asset ID for later
    confirmation, and URL expiration timestamp.
    """

    upload_url: str = Field(
        ...,
        description="S3 presigned PUT URL for direct upload",
    )
    asset_id: str = Field(
        ...,
        description="Asset ID for use in confirmation endpoint",
    )
    s3_key: str = Field(
        ...,
        description="S3 object key where file will be stored",
    )
    expiration: str = Field(
        ...,
        description="ISO 8601 timestamp when URL expires",
    )
    expires_in_seconds: int = Field(
        ...,
        description="Seconds until URL expiration",
    )


class UploadConfirmationRequest(BaseModel):
    """
    Request model for confirming S3 presigned upload completion.

    After uploading file to S3 via presigned URL, client calls
    POST /confirmation with the asset_id to finalize the upload.
    """

    asset_id: str = Field(
        ...,
        min_length=1,
        description="Asset ID from presigned URL response",
    )
    s3_key: str = Field(
        ...,
        min_length=1,
        description="S3 object key from presigned URL response",
    )


class URLUploadRequest(BaseModel):
    """
    Request model for URL-based content import.

    Supports YouTube, Vimeo, and general webpage URLs for
    content extraction and fingerprinting.
    """

    url: HttpUrl = Field(
        ...,
        description="URL to import (YouTube, Vimeo, or webpage)",
        examples=["https://youtube.com/watch?v=xxx", "https://vimeo.com/123456"],
    )
    asset_type: str | None = Field(
        default=None,
        description="Optional asset type hint (video, webpage)",
        examples=["video", "webpage"],
    )


class UploadResponse(BaseModel):
    """
    Standard response model for successful uploads.

    Returned by all direct upload endpoints after successful
    file processing and asset record creation.
    """

    asset_id: str = Field(
        ...,
        description="Unique identifier for the uploaded asset",
    )
    status: str = Field(
        ...,
        description="Upload status (queued, processing, ready)",
    )
    message: str = Field(
        ...,
        description="Human-readable status message",
    )
    file_name: str | None = Field(
        default=None,
        description="Original filename if applicable",
    )
    file_type: str | None = Field(
        default=None,
        description="Detected file type (image, audio, video, text)",
    )
    file_size: int | None = Field(
        default=None,
        description="File size in bytes",
    )
    s3_key: str | None = Field(
        default=None,
        description="S3 storage key for the file",
    )


class PresignedUploadRequiredResponse(BaseModel):
    """
    Response indicating presigned URL upload is required.

    Returned when file exceeds direct upload threshold (10MB)
    and client should use GET /presigned-url endpoint instead.
    """

    message: str = Field(
        ...,
        description="Instructions for using presigned URL upload",
    )
    file_size: int = Field(
        ...,
        description="Size of file that triggered redirect",
    )
    threshold: int = Field(
        ...,
        description="Direct upload size threshold in bytes",
    )
    presigned_url_endpoint: str = Field(
        ...,
        description="Endpoint to call for presigned URL",
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Provides structured error information for client consumption.
    """

    error: str = Field(
        ...,
        description="Error type/code",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


# ============================================================================
# Router Definition
# ============================================================================

router = APIRouter(
    tags=["upload"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Invalid or missing token"},
        413: {"model": ErrorResponse, "description": "File too large (>500MB)"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# ============================================================================
# Dependency Injection Functions
# ============================================================================


def get_upload_service() -> UploadService:
    """
    Dependency injection for UploadService.

    Creates and returns UploadService instance with required
    dependencies (StorageService, database client).

    Returns:
        UploadService: Configured upload service instance.
    """
    storage_service = StorageService()
    db_client = get_db_client()
    return UploadService(storage_service=storage_service, db_client=db_client)


def get_storage_service() -> StorageService:
    """
    Dependency injection for StorageService.

    Returns:
        StorageService: Configured storage service instance.
    """
    return StorageService()


def get_fingerprinting_service() -> FingerprintingService:
    """
    Dependency injection for FingerprintingService.

    Returns:
        FingerprintingService: Configured fingerprinting service instance.
    """
    storage_service = StorageService()
    return FingerprintingService(storage_service=storage_service)


# ============================================================================
# Helper Functions
# ============================================================================


def validate_file_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """
    Validate file extension against allowed list.

    Args:
        filename: Original filename to check.
        allowed_extensions: Set of allowed lowercase extensions with dots.

    Returns:
        bool: True if extension is allowed, False otherwise.
    """
    if not filename:
        return False

    # Extract extension and normalize to lowercase
    extension = ""
    if "." in filename:
        extension = "." + filename.rsplit(".", 1)[-1].lower()

    return extension in allowed_extensions


def get_file_extension(filename: str) -> str:
    """
    Extract and normalize file extension from filename.

    Args:
        filename: Original filename.

    Returns:
        str: Lowercase extension with dot (e.g., ".jpg") or empty string.
    """
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


async def trigger_fingerprint_generation(
    background_tasks: BackgroundTasks,
    fingerprinting_service: FingerprintingService,
    asset_id: str,
    s3_key: str,
    file_type: str,
    user_id: str,
) -> None:
    """
    Queue fingerprint generation as background task.

    Adds fingerprint generation to FastAPI BackgroundTasks for
    asynchronous processing after response is sent to client.

    Args:
        background_tasks: FastAPI background tasks handler.
        fingerprinting_service: FingerprintingService instance.
        asset_id: Asset ID to generate fingerprint for.
        s3_key: S3 object key for the asset file.
        file_type: Type of file (image, audio, video, text).
        user_id: User ID who owns the asset.
    """
    logger.info(f"Queueing fingerprint generation for asset_id={asset_id}, file_type={file_type}")

    background_tasks.add_task(
        fingerprinting_service.generate_fingerprint,
        asset_id=asset_id,
        object_key=s3_key,
        file_type=file_type,
        user_id=user_id,
    )


# ============================================================================
# Upload Endpoints
# ============================================================================


@router.post(
    "/text",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload text content",
    description="Direct upload of text content (txt, md, pdf) files under 10MB.",
    responses={
        201: {"model": UploadResponse, "description": "Text uploaded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or format"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_text(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Text file to upload"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse:
    """
    Upload text content directly to the platform.

    Accepts text files (txt, md, pdf) under 10MB for direct upload.
    Files are validated, stored in S3, registered in MongoDB, and
    queued for fingerprint generation.

    Args:
        background_tasks: FastAPI background tasks for async processing.
        file: Text file uploaded via multipart form.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Upload confirmation with asset ID and status.

    Raises:
        HTTPException: 400 if invalid file type, 413 if file too large.
    """
    logger.info(f"Text upload request from user {current_user.id}: {file.filename}")

    # Validate file extension
    if not validate_file_extension(file.filename or "", ALLOWED_TEXT_EXTENSIONS):
        logger.warning(f"Invalid text file extension: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"Invalid file type. Allowed text extensions: {', '.join(ALLOWED_TEXT_EXTENSIONS)}",
                "filename": file.filename,
            },
        )

    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    # Check file size limits
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"Text file too large: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                "file_size": file_size,
                "max_size": MAX_FILE_SIZE_BYTES,
            },
        )

    # Check if exceeds direct upload threshold
    if file_size >= DIRECT_UPLOAD_THRESHOLD_BYTES:
        logger.info(f"Text file exceeds direct upload threshold: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "use_presigned_upload",
                "message": "File exceeds 10MB threshold. Use GET /presigned-url for large files.",
                "file_size": file_size,
                "threshold": DIRECT_UPLOAD_THRESHOLD_BYTES,
                "presigned_url_endpoint": "/api/v1/upload/presigned-url",
            },
        )

    try:
        # Decode bytes to string for text upload service
        text_content = content.decode("utf-8", errors="replace")

        # Handle direct upload via upload service
        result = await upload_service.handle_text_upload(
            content=text_content,
            filename=file.filename or "upload.txt",
            user_id=str(current_user.id),
        )

        # Queue fingerprint generation in background
        await trigger_fingerprint_generation(
            background_tasks=background_tasks,
            fingerprinting_service=fingerprinting_service,
            asset_id=result["asset_id"],
            s3_key=result["s3_key"],
            file_type="text",
            user_id=str(current_user.id),
        )

        logger.info(f"Text upload successful: asset_id={result['asset_id']}")

        return UploadResponse(
            asset_id=result["asset_id"],
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message="Text file uploaded successfully. Fingerprint generation queued.",
            file_name=result.get("file_name", file.filename),
            file_type=result.get("file_type", "text"),
            file_size=result.get("file_size", file_size),
            s3_key=result.get("s3_key"),
        )

    except Exception as e:
        logger.exception("Text upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "upload_failed",
                "message": f"Failed to upload text file: {e!s}",
            },
        ) from None


@router.post(
    "/image",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload image",
    description="Direct upload of image files (png, jpg, jpeg, webp) under 10MB.",
    responses={
        201: {"model": UploadResponse, "description": "Image uploaded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or format"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to upload"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse:
    """
    Upload image directly to the platform.

    Accepts image files (png, jpg, jpeg, webp) under 10MB for direct upload.
    Files are validated, stored in S3, registered in MongoDB, and
    queued for perceptual hash fingerprint generation (pHash, aHash, dHash).

    Args:
        background_tasks: FastAPI background tasks for async processing.
        file: Image file uploaded via multipart form.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Upload confirmation with asset ID and status.

    Raises:
        HTTPException: 400 if invalid file type, 413 if file too large.
    """
    logger.info(f"Image upload request from user {current_user.id}: {file.filename}")

    # Validate file extension
    if not validate_file_extension(file.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        logger.warning(f"Invalid image file extension: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "unsupported_media_type",
                "message": f"Invalid image type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
                "filename": file.filename,
            },
        )

    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    # Check file size limits
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"Image file too large: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                "file_size": file_size,
                "max_size": MAX_FILE_SIZE_BYTES,
            },
        )

    # Check if exceeds direct upload threshold - redirect to presigned URL
    if file_size >= DIRECT_UPLOAD_THRESHOLD_BYTES:
        logger.info(f"Image file exceeds direct upload threshold: {file_size} bytes")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "File exceeds 10MB threshold. Use presigned URL upload.",
                "file_size": file_size,
                "threshold": DIRECT_UPLOAD_THRESHOLD_BYTES,
                "presigned_url_endpoint": "/api/v1/upload/presigned-url",
                "instructions": "Call GET /presigned-url with filename, content_type, and file_size",
            },
        )

    try:
        # Reset file position for upload service
        await file.seek(0)

        # Handle direct upload (upload service reads file content internally)
        result = await upload_service.handle_direct_upload(
            file=file,
            user_id=str(current_user.id),
        )

        # Queue fingerprint generation in background
        await trigger_fingerprint_generation(
            background_tasks=background_tasks,
            fingerprinting_service=fingerprinting_service,
            asset_id=result["asset_id"],
            s3_key=result["s3_key"],
            file_type="image",
            user_id=str(current_user.id),
        )

        logger.info(f"Image upload successful: asset_id={result['asset_id']}")

        return UploadResponse(
            asset_id=result["asset_id"],
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message="Image uploaded successfully. Fingerprint generation queued.",
            file_name=result.get("file_name", file.filename),
            file_type=result.get("file_type", "image"),
            file_size=result.get("file_size", file_size),
            s3_key=result.get("s3_key"),
        )

    except Exception as e:
        logger.exception("Image upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "upload_failed",
                "message": f"Failed to upload image: {e!s}",
            },
        ) from None


@router.post(
    "/audio",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio",
    description="Direct upload of audio files (mp3, wav, aac) under 10MB.",
    responses={
        201: {"model": UploadResponse, "description": "Audio uploaded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or format"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to upload"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse:
    """
    Upload audio directly to the platform.

    Accepts audio files (mp3, wav, aac) under 10MB for direct upload.
    Files are validated, stored in S3, registered in MongoDB, and
    queued for spectral fingerprint generation (mel-spectrogram, chromagram).

    Args:
        background_tasks: FastAPI background tasks for async processing.
        file: Audio file uploaded via multipart form.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Upload confirmation with asset ID and status.

    Raises:
        HTTPException: 400 if invalid file type, 413 if file too large.
    """
    logger.info(f"Audio upload request from user {current_user.id}: {file.filename}")

    # Validate file extension
    if not validate_file_extension(file.filename or "", ALLOWED_AUDIO_EXTENSIONS):
        logger.warning(f"Invalid audio file extension: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "unsupported_media_type",
                "message": f"Invalid audio type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
                "filename": file.filename,
            },
        )

    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    # Check file size limits
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"Audio file too large: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                "file_size": file_size,
                "max_size": MAX_FILE_SIZE_BYTES,
            },
        )

    # Check if exceeds direct upload threshold
    if file_size >= DIRECT_UPLOAD_THRESHOLD_BYTES:
        logger.info(f"Audio file exceeds direct upload threshold: {file_size} bytes")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "File exceeds 10MB threshold. Use presigned URL upload.",
                "file_size": file_size,
                "threshold": DIRECT_UPLOAD_THRESHOLD_BYTES,
                "presigned_url_endpoint": "/api/v1/upload/presigned-url",
                "instructions": "Call GET /presigned-url with filename, content_type, and file_size",
            },
        )

    try:
        # Reset file position
        await file.seek(0)

        # Handle direct upload (upload service reads file content internally)
        result = await upload_service.handle_direct_upload(
            file=file,
            user_id=str(current_user.id),
        )

        # Queue fingerprint generation
        await trigger_fingerprint_generation(
            background_tasks=background_tasks,
            fingerprinting_service=fingerprinting_service,
            asset_id=result["asset_id"],
            s3_key=result["s3_key"],
            file_type="audio",
            user_id=str(current_user.id),
        )

        logger.info(f"Audio upload successful: asset_id={result['asset_id']}")

        return UploadResponse(
            asset_id=result["asset_id"],
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message="Audio uploaded successfully. Fingerprint generation queued.",
            file_name=result.get("file_name", file.filename),
            file_type=result.get("file_type", "audio"),
            file_size=result.get("file_size", file_size),
            s3_key=result.get("s3_key"),
        )

    except Exception as e:
        logger.exception("Audio upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "upload_failed",
                "message": f"Failed to upload audio: {e!s}",
            },
        ) from None


@router.post(
    "/video",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload video",
    description="Upload video files (mp4, mov, avi). Files over 10MB use presigned URL.",
    responses={
        201: {"model": UploadResponse, "description": "Video uploaded successfully"},
        200: {"model": PresignedUploadRequiredResponse, "description": "Presigned URL required"},
        400: {"model": ErrorResponse, "description": "Invalid file type or format"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to upload"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse | JSONResponse:
    """
    Upload video to the platform.

    Accepts video files (mp4, mov, avi). Most videos exceed 10MB,
    so this endpoint often redirects to presigned URL flow.
    Small videos are directly uploaded and queued for frame-based fingerprinting.

    Args:
        background_tasks: FastAPI background tasks for async processing.
        file: Video file uploaded via multipart form.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Upload confirmation if direct upload.
        JSONResponse: Presigned URL instructions if file > 10MB.

    Raises:
        HTTPException: 415 if invalid file type, 413 if file > 500MB.
    """
    logger.info(f"Video upload request from user {current_user.id}: {file.filename}")

    # Validate file extension
    if not validate_file_extension(file.filename or "", ALLOWED_VIDEO_EXTENSIONS):
        logger.warning(f"Invalid video file extension: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "unsupported_media_type",
                "message": f"Invalid video type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
                "filename": file.filename,
            },
        )

    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    # Check absolute file size limit
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"Video file too large: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                "file_size": file_size,
                "max_size": MAX_FILE_SIZE_BYTES,
            },
        )

    # Videos typically exceed direct upload threshold - redirect to presigned URL
    if file_size >= DIRECT_UPLOAD_THRESHOLD_BYTES:
        logger.info(
            f"Video exceeds direct upload threshold ({file_size} bytes). "
            "Redirecting to presigned URL flow."
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Video exceeds 10MB threshold. Use presigned URL upload for better reliability.",
                "file_size": file_size,
                "threshold": DIRECT_UPLOAD_THRESHOLD_BYTES,
                "presigned_url_endpoint": "/api/v1/upload/presigned-url",
                "instructions": (
                    "1. Call GET /presigned-url with filename, content_type, file_size\n"
                    "2. Upload file directly to returned upload_url\n"
                    "3. Call POST /confirmation with asset_id and s3_key"
                ),
            },
        )

    try:
        # Reset file position for small video direct upload
        await file.seek(0)

        # Handle direct upload (upload service reads file content internally)
        result = await upload_service.handle_direct_upload(
            file=file,
            user_id=str(current_user.id),
        )

        # Queue fingerprint generation
        await trigger_fingerprint_generation(
            background_tasks=background_tasks,
            fingerprinting_service=fingerprinting_service,
            asset_id=result["asset_id"],
            s3_key=result["s3_key"],
            file_type="video",
            user_id=str(current_user.id),
        )

        logger.info(f"Video upload successful: asset_id={result['asset_id']}")

        return UploadResponse(
            asset_id=result["asset_id"],
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message="Video uploaded successfully. Fingerprint generation queued.",
            file_name=result.get("file_name", file.filename),
            file_type=result.get("file_type", "video"),
            file_size=result.get("file_size", file_size),
            s3_key=result.get("s3_key"),
        )

    except Exception as e:
        logger.exception("Video upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "upload_failed",
                "message": f"Failed to upload video: {e!s}",
            },
        ) from None


@router.post(
    "/url",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import content from URL",
    description="Import content from YouTube, Vimeo, or general webpage URLs.",
    responses={
        201: {"model": UploadResponse, "description": "URL content imported successfully"},
        400: {"model": ErrorResponse, "description": "Invalid or dangerous URL"},
    },
)
async def upload_url(
    background_tasks: BackgroundTasks,
    request: URLUploadRequest,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse:
    """
    Import and process content from a URL.

    Supports:
    - YouTube: Extracts transcript and metadata
    - Vimeo: Extracts video metadata
    - General webpages: Extracts text content

    URLs are validated for safety (no dangerous file types),
    content is extracted, stored, and queued for fingerprinting.

    Args:
        background_tasks: FastAPI background tasks for async processing.
        request: URL upload request with URL and optional type hint.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Import confirmation with asset ID and status.

    Raises:
        HTTPException: 400 if URL is invalid or dangerous.
    """
    url_str = str(request.url)
    logger.info(f"URL import request from user {current_user.id}: {url_str}")

    # Validate URL safety using file_validator
    is_valid, _url_type, error_message = validate_url(url_str)

    if not is_valid:
        logger.warning(f"Invalid URL rejected: {url_str} - {error_message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_url",
                "message": error_message or "URL validation failed",
                "url": url_str,
            },
        )

    try:
        # Handle URL upload via upload service (internally detects platform and processes)
        result = await upload_service.handle_url_upload(
            url=url_str,
            user_id=str(current_user.id),
        )

        # Determine file type for fingerprinting based on platform from result
        platform = result.get("platform", "webpage")
        file_type = "text"  # Default for webpage content
        if platform in ("youtube", "vimeo"):
            file_type = "text"  # URL content is stored as text (transcript/metadata)

        # Queue fingerprint generation if S3 key exists
        if result.get("s3_key"):
            await trigger_fingerprint_generation(
                background_tasks=background_tasks,
                fingerprinting_service=fingerprinting_service,
                asset_id=result["asset_id"],
                s3_key=result["s3_key"],
                file_type=file_type,
                user_id=str(current_user.id),
            )

        logger.info(f"URL import successful: asset_id={result['asset_id']}, platform={platform}")

        return UploadResponse(
            asset_id=result["asset_id"],
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message=f"URL content from {platform} imported successfully. Processing queued.",
            file_name=result.get("file_name", url_str),
            file_type=result.get("file_type", "url"),
            s3_key=result.get("s3_key"),
        )

    except Exception as e:
        logger.exception("URL import failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "import_failed",
                "message": f"Failed to import URL content: {e!s}",
                "url": url_str,
            },
        ) from None


@router.get(
    "/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Generate presigned upload URL",
    description="Generate S3 presigned PUT URL for files >= 10MB (max 500MB).",
    responses={
        200: {"model": PresignedUrlResponse, "description": "Presigned URL generated"},
        400: {"model": ErrorResponse, "description": "Invalid file type or parameters"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def get_presigned_url(
    filename: str = Query(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename with extension",
    ),
    content_type: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="MIME type of the file",
    ),
    file_size: int = Query(
        ...,
        ge=0,
        le=MAX_FILE_SIZE_BYTES,
        description="File size in bytes",
    ),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
) -> PresignedUrlResponse:
    """
    Generate S3 presigned PUT URL for large file upload.

    Creates a time-limited (15 minutes) signed URL for direct
    client-to-S3 upload. Also creates a placeholder asset record
    in MongoDB with "pending" status.

    Workflow:
    1. Client calls this endpoint with file metadata
    2. Client uploads file directly to returned upload_url
    3. Client calls POST /confirmation to finalize upload

    Args:
        filename: Original filename with extension for validation.
        content_type: MIME type for Content-Type header.
        file_size: File size for validation (max 500MB).
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.

    Returns:
        PresignedUrlResponse: Signed URL and asset ID for confirmation.

    Raises:
        HTTPException: 400 if invalid file type, 413 if too large.
    """
    logger.info(
        f"Presigned URL request from user {current_user.id}: "
        f"filename={filename}, size={file_size}"
    )

    # Validate file extension
    extension = get_file_extension(filename)
    all_allowed = (
        ALLOWED_IMAGE_EXTENSIONS
        | ALLOWED_AUDIO_EXTENSIONS
        | ALLOWED_VIDEO_EXTENSIONS
        | ALLOWED_TEXT_EXTENSIONS
    )

    if extension not in all_allowed:
        logger.warning(f"Invalid file extension for presigned URL: {extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"File type '{extension}' not allowed. Allowed types: {', '.join(sorted(all_allowed))}",
                "filename": filename,
            },
        )

    # Reject dangerous file types (additional security check)
    dangerous_extensions = {
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".exe",
        ".bin",
        ".sh",
        ".bat",
        ".cmd",
        ".msi",
        ".app",
        ".dmg",
        ".iso",
    }
    if extension in dangerous_extensions:
        logger.warning(f"Dangerous file extension rejected: {extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "dangerous_file_type",
                "message": f"File type '{extension}' is not allowed for security reasons",
                "filename": filename,
            },
        )

    # Validate file size
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File too large for presigned URL: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                "file_size": file_size,
                "max_size": MAX_FILE_SIZE_BYTES,
            },
        )

    try:
        # Generate presigned URL via upload service
        result = await upload_service.generate_presigned_upload_url(
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            user_id=str(current_user.id),
        )

        # Use expiration time from result or calculate based on settings
        expiration_time_str = result.get("expiration_time")
        if expiration_time_str:
            # Parse the ISO timestamp from the service
            expiration = expiration_time_str
        else:
            # Calculate expiration timestamp if not provided
            expiration_dt = datetime.now(UTC) + timedelta(seconds=PRESIGNED_URL_EXPIRATION_SECONDS)
            expiration = expiration_dt.isoformat()

        expires_in = result.get("expires_in", PRESIGNED_URL_EXPIRATION_SECONDS)

        logger.info(f"Presigned URL generated: asset_id={result['asset_id']}, expires={expiration}")

        return PresignedUrlResponse(
            upload_url=result["presigned_url"],  # Service returns 'presigned_url'
            asset_id=result["asset_id"],
            s3_key=result["object_key"],  # Service returns 'object_key'
            expiration=expiration,
            expires_in_seconds=expires_in,
        )

    except Exception as e:
        logger.exception("Presigned URL generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "presigned_url_failed",
                "message": f"Failed to generate presigned URL: {e!s}",
            },
        ) from None


@router.post(
    "/confirmation",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Confirm presigned upload",
    description="Confirm S3 upload completion and finalize asset registration.",
    responses={
        200: {"model": UploadResponse, "description": "Upload confirmed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid asset ID or S3 key"},
        404: {"model": ErrorResponse, "description": "Asset or S3 object not found"},
    },
)
async def confirm_upload(
    background_tasks: BackgroundTasks,
    request: UploadConfirmationRequest,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service),
    storage_service: StorageService = Depends(get_storage_service),
    fingerprinting_service: FingerprintingService = Depends(get_fingerprinting_service),
) -> UploadResponse:
    """
    Confirm and finalize presigned URL upload.

    After client uploads file to S3 via presigned URL, this endpoint:
    1. Verifies the S3 object exists via HEAD request
    2. Updates asset record status from "pending" to "uploaded"
    3. Queues fingerprint generation for the uploaded file

    Args:
        background_tasks: FastAPI background tasks for async processing.
        request: Confirmation request with asset_id and s3_key.
        current_user: Authenticated user from JWT token.
        upload_service: Injected upload service.
        storage_service: Injected storage service for S3 verification.
        fingerprinting_service: Injected fingerprinting service.

    Returns:
        UploadResponse: Confirmation with updated asset status.

    Raises:
        HTTPException: 404 if asset or S3 object not found.
    """
    logger.info(
        f"Upload confirmation from user {current_user.id}: "
        f"asset_id={request.asset_id}, s3_key={request.s3_key}"
    )

    try:
        # Verify S3 object exists
        s3_exists = await storage_service.file_exists(request.s3_key)

        if not s3_exists:
            logger.warning(f"S3 object not found: {request.s3_key}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "s3_object_not_found",
                    "message": "File not found in S3. Upload may have failed or URL may have expired.",
                    "s3_key": request.s3_key,
                },
            )

        # Confirm upload via upload service (validates ownership internally)
        result = await upload_service.confirm_presigned_upload(
            asset_id=request.asset_id,
            object_key=request.s3_key,
        )

        # Determine file type from result or S3 key extension
        file_type_result = result.get("file_type")
        if file_type_result:
            # Use the file type from result (may be an enum value)
            file_type = str(file_type_result).lower()
        else:
            # Fallback to extension-based detection
            extension = get_file_extension(request.s3_key)
            if extension in ALLOWED_IMAGE_EXTENSIONS:
                file_type = "image"
            elif extension in ALLOWED_AUDIO_EXTENSIONS:
                file_type = "audio"
            elif extension in ALLOWED_VIDEO_EXTENSIONS:
                file_type = "video"
            elif extension in ALLOWED_TEXT_EXTENSIONS:
                file_type = "text"
            else:
                file_type = "image"  # Default

        # Queue fingerprint generation
        await trigger_fingerprint_generation(
            background_tasks=background_tasks,
            fingerprinting_service=fingerprinting_service,
            asset_id=request.asset_id,
            s3_key=request.s3_key,
            file_type=file_type,
            user_id=str(current_user.id),
        )

        logger.info(f"Upload confirmed: asset_id={request.asset_id}, file_type={file_type}")

        return UploadResponse(
            asset_id=request.asset_id,
            status=result.get("upload_status", UploadStatus.QUEUED.value),
            message="Upload confirmed successfully. Fingerprint generation queued.",
            file_name=result.get("file_name"),
            file_type=file_type,
            file_size=result.get("file_size"),
            s3_key=request.s3_key,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Upload confirmation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "confirmation_failed",
                "message": f"Failed to confirm upload: {e!s}",
                "asset_id": request.asset_id,
            },
        ) from None
