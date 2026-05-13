"""
META-STAMP V3 Fingerprinting API Router Module

This module implements the fingerprinting REST API endpoints for generating and retrieving
multi-modal fingerprints for creative assets. The fingerprinting system uniquely identifies
content using:

- Perceptual hashing (pHash, aHash, dHash) for images using discrete cosine transform
- Spectral analysis (mel-spectrogram, chromagram) for audio using librosa
- Frame-based hashing at 1-second intervals for video using OpenCV
- Semantic embeddings via LangChain OpenAI for multi-modal similarity detection

Endpoints:
    POST /  - Initiate fingerprint generation for an asset (async background processing)
    GET /{fingerprint_id} - Retrieve complete fingerprint data including hashes and embeddings

Per Agent Action Plan:
- Section 0.4: Multi-modal fingerprinting implementation
- Section 0.6: fingerprint.py API route transformation
- Section 0.8: Endpoint specifications with BackgroundTasks for async processing
- Section 0.10: Phase 2 preparation with TODO markers for AI training detection

Author: META-STAMP V3 Platform
License: Proprietary
"""

import logging

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.models.asset import ProcessingStatus, UploadStatus
from app.models.fingerprint import ProcessingStatus as FingerprintProcessingStatus
from app.services.fingerprinting_service import (
    FingerprintGenerationError,
    FingerprintingService,
    UnsupportedFileTypeError,
)
from app.services.metadata_service import MetadataService
from app.services.storage_service import StorageService


# Configure module logger
logger = logging.getLogger(__name__)

# Constants for validation
MONGO_OBJECTID_LENGTH = 24  # MongoDB ObjectIds are 24 character hex strings


# =============================================================================
# API ROUTER CONFIGURATION
# =============================================================================

router = APIRouter(
    tags=["fingerprinting"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Access forbidden - not authorized"},
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
    },
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class FingerprintRequest(BaseModel):
    """
    Request model for initiating fingerprint generation.

    Attributes:
        asset_id: MongoDB ObjectId string reference to the asset to fingerprint.
                 The asset must exist, belong to the authenticated user, and
                 not already have an associated fingerprint.
    """

    asset_id: str = Field(
        ...,
        min_length=24,
        max_length=24,
        description="MongoDB ObjectId of the asset to fingerprint",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )


class FingerprintStatusResponse(BaseModel):
    """
    Response model for fingerprint generation initiation.

    Returns the fingerprint ID for status tracking and the initial
    processing status. The fingerprint will be generated asynchronously
    in the background.

    Attributes:
        fingerprint_id: MongoDB ObjectId of the created fingerprint record
        status: Current processing status (pending, processing, completed, failed)
        message: Human-readable status message
        created_at: Timestamp when fingerprint generation was initiated
    """

    fingerprint_id: str = Field(
        ...,
        description="MongoDB ObjectId of the fingerprint record",
        json_schema_extra={"example": "507f1f77bcf86cd799439012"},
    )
    status: str = Field(
        ...,
        description="Processing status (pending, processing, completed, failed)",
        json_schema_extra={"example": "processing"},
    )
    message: str = Field(
        ...,
        max_length=500,
        description="Human-readable status message",
        json_schema_extra={"example": "Fingerprint generation has been queued for processing"},
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when fingerprint generation was initiated (UTC)",
    )


class PerceptualHashesResponse(BaseModel):
    """Response model for perceptual hash data (images)."""

    phash: str | None = Field(default=None, description="Perceptual hash using DCT")
    ahash: str | None = Field(default=None, description="Average hash")
    dhash: str | None = Field(default=None, description="Difference hash")
    hash_size: int | None = Field(default=None, description="Hash bit length")


class SpectralDataResponse(BaseModel):
    """Response model for audio spectral analysis data."""

    mel_spectrogram_mean: float | None = Field(
        default=None, description="Mean of mel-spectrogram features"
    )
    chromagram_mean: float | None = Field(default=None, description="Mean of chromagram features")
    spectral_centroid: float | None = Field(
        default=None, description="Spectral centroid (brightness measure)"
    )
    duration: float | None = Field(default=None, description="Audio duration in seconds")
    sample_rate: int | None = Field(default=None, description="Audio sample rate")


class VideoHashesResponse(BaseModel):
    """Response model for video frame hash data."""

    frame_hashes: list[str] | None = Field(
        default=None, description="List of representative frame hashes"
    )
    average_hash: str | None = Field(default=None, description="Overall video hash")
    sampling_interval: float | None = Field(
        default=None, description="Frame extraction interval in seconds"
    )
    total_frames_analyzed: int | None = Field(
        default=None, description="Number of frames processed"
    )


class EmbeddingsResponse(BaseModel):
    """Response model for multi-modal embeddings data."""

    openai_embedding: list[float] | None = Field(
        default=None, description="1536-dimensional OpenAI embedding vector"
    )
    clip_embedding: list[float] | None = Field(
        default=None, description="512-dimensional CLIP embedding vector"
    )
    embedding_model: str | None = Field(default=None, description="Model identifier")
    embedding_version: str | None = Field(default=None, description="Version tracking")


class FingerprintDetailResponse(BaseModel):
    """
    Comprehensive response model for fingerprint retrieval.

    Contains all fingerprint data including perceptual hashes for images,
    spectral analysis for audio, frame hashes for video, and multi-modal
    embeddings for semantic similarity detection.

    Phase 2 fields (training_detected, dataset_matches, similarity_scores,
    legal_status) are included as placeholders for future AI training detection.
    """

    fingerprint_id: str = Field(
        ...,
        description="MongoDB ObjectId of the fingerprint",
        json_schema_extra={"example": "507f1f77bcf86cd799439012"},
    )
    asset_id: str = Field(
        ...,
        description="MongoDB ObjectId of the associated asset",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )
    fingerprint_type: str = Field(
        ...,
        description="Type of fingerprint (image, audio, video, text)",
        json_schema_extra={"example": "image"},
    )
    processing_status: str = Field(
        ...,
        description="Processing status (pending, processing, completed, failed)",
        json_schema_extra={"example": "completed"},
    )

    # Image fingerprint data
    perceptual_hashes: dict[str, Any] | None = Field(
        default=None,
        description="Perceptual hash data for images (pHash, aHash, dHash)",
    )

    # Audio fingerprint data
    spectral_data: dict[str, Any] | None = Field(
        default=None,
        description="Audio spectral analysis data (mel-spectrogram, chromagram)",
    )

    # Video fingerprint data
    video_hashes: dict[str, Any] | None = Field(
        default=None,
        description="Video frame hash data with sampling metadata",
    )

    # Multi-modal embeddings
    embeddings: dict[str, Any] | None = Field(
        default=None,
        description="Multi-modal embedding vectors (OpenAI, CLIP)",
    )

    # Metadata
    image_metadata: dict[str, Any] | None = Field(
        default=None, description="Image-specific metadata (EXIF, dimensions)"
    )
    audio_metadata: dict[str, Any] | None = Field(
        default=None, description="Audio-specific metadata (codec, bitrate)"
    )
    video_metadata: dict[str, Any] | None = Field(
        default=None, description="Video-specific metadata (resolution, fps)"
    )
    text_hash: str | None = Field(default=None, description="SHA-256 hash of text content")
    text_length: int | None = Field(
        default=None, description="Length of text content in characters"
    )

    # Processing metadata
    processing_duration: float | None = Field(
        default=None, description="Time taken to generate fingerprint in seconds"
    )
    error_message: str | None = Field(
        default=None, description="Error details if processing failed"
    )
    created_at: datetime = Field(..., description="Timestamp when fingerprint was created (UTC)")
    updated_at: datetime | None = Field(
        default=None, description="Timestamp when fingerprint was last modified (UTC)"
    )

    # Phase 2 AI Training Detection Fields
    # TODO Phase 2: Implement AI training detection engine
    training_detected: bool | None = Field(
        default=None,
        description="Phase 2: Whether asset was detected in AI training data",
    )
    # TODO Phase 2: Compare embeddings against known datasets
    dataset_matches: list[str] | None = Field(
        default=None,
        description="Phase 2: List of matched AI training datasets",
    )
    # TODO Phase 2: Calculate embedding drift scores
    similarity_scores: dict[str, float] | None = Field(
        default=None,
        description="Phase 2: Similarity scores against matched datasets",
    )
    # TODO Phase 2: Apply similarity-law thresholds
    legal_status: str | None = Field(
        default=None,
        description="Phase 2: Legal status determination based on analysis",
    )


# =============================================================================
# BACKGROUND TASK FUNCTION
# =============================================================================


async def process_fingerprint_background(
    asset_id: str,
    fingerprint_id: str,
    object_key: str,
    file_type: str,
    user_id: str,
) -> None:
    """
    Background task to generate fingerprint for an asset.

    This function is executed asynchronously after the API response is sent.
    It downloads the asset from S3/MinIO, generates the appropriate fingerprint
    based on file type, and updates both the fingerprint record and asset status.

    The processing flow:
    1. Update fingerprint status to 'processing'
    2. Initialize fingerprinting service with storage and metadata services
    3. Call generate_fingerprint() which handles:
       - File download from S3/MinIO
       - Media type detection and routing
       - Hash/embedding generation
       - Result storage in MongoDB
    4. Update asset status to 'ready' and link fingerprint_id
    5. Handle errors by updating status to 'failed' with error message

    Args:
        asset_id: MongoDB ObjectId of the asset being fingerprinted
        fingerprint_id: MongoDB ObjectId of the fingerprint record to update
        object_key: S3 object key (path) for the file in storage
        file_type: Type of file (image, audio, video, text)
        user_id: MongoDB ObjectId of the user who owns the asset

    Note:
        This function does not raise exceptions - all errors are logged and
        stored in the fingerprint record for later retrieval.

    TODO Phase 2: After fingerprint generation, trigger AI training detection
    TODO Phase 2: Compare embeddings against known datasets
    TODO Phase 2: Calculate embedding drift scores
    TODO Phase 2: Apply similarity-law thresholds for legal determination
    """
    logger.info(
        f"Starting background fingerprint generation - "
        f"asset_id={asset_id}, fingerprint_id={fingerprint_id}, file_type={file_type}"
    )

    try:
        # Get database client for updates
        db_client = get_db_client()
        fingerprints_collection = db_client.get_fingerprints_collection()
        assets_collection = db_client.get_assets_collection()

        # Update fingerprint status to processing
        await fingerprints_collection.update_one(
            {"_id": fingerprint_id},
            {
                "$set": {
                    "processing_status": FingerprintProcessingStatus.PROCESSING.value,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        logger.debug(f"Updated fingerprint {fingerprint_id} status to processing")

        # Initialize services for fingerprint generation
        settings = get_settings()

        # Create storage service with settings
        storage_service = StorageService(
            bucket_name=settings.s3_bucket_name,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )

        # Create metadata service
        metadata_service = MetadataService()

        # Create fingerprinting service with optional OpenAI API key for embeddings
        fingerprinting_service = FingerprintingService(
            storage_service=storage_service,
            metadata_service=metadata_service,
            openai_api_key=settings.openai_api_key,
        )

        # Generate fingerprint using the service
        # This will download the file, generate hashes/embeddings, and store in MongoDB
        result = await fingerprinting_service.generate_fingerprint(
            asset_id=asset_id,
            object_key=object_key,
            file_type=file_type,
            user_id=user_id,
        )

        logger.info(
            f"Fingerprint generation completed successfully - "
            f"fingerprint_id={result.get('fingerprint_id', fingerprint_id)}, "
            f"processing_duration={result.get('processing_duration', 0)}s"
        )

        # The fingerprinting service creates its own record, but we need to ensure
        # our placeholder record is either updated or removed
        # First, check if the service created a different record
        service_fingerprint_id = result.get("fingerprint_id")
        if service_fingerprint_id and service_fingerprint_id != fingerprint_id:
            # Service created its own record, delete our placeholder
            await fingerprints_collection.delete_one({"_id": fingerprint_id})
            fingerprint_id = service_fingerprint_id
            logger.debug(f"Using service-created fingerprint ID: {fingerprint_id}")

        # Update asset with fingerprint_id and set status to ready
        await assets_collection.update_one(
            {"_id": ObjectId(asset_id)},
            {
                "$set": {
                    "fingerprint_id": fingerprint_id,
                    "upload_status": UploadStatus.READY.value,
                    "processing_status": ProcessingStatus.COMPLETED.value,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        logger.info(f"Asset {asset_id} updated with fingerprint_id={fingerprint_id}, status=ready")

        # TODO Phase 2: Implement AI training detection engine
        # After successful fingerprint generation, initiate AI training detection
        # await detect_ai_training_usage(fingerprint_id)

        # TODO Phase 2: Compare embeddings against known datasets
        # if result.get("embeddings"):
        #     await compare_against_training_datasets(result["embeddings"])

        # TODO Phase 2: Calculate embedding drift scores
        # await calculate_embedding_drift(fingerprint_id)

        # TODO Phase 2: Apply similarity-law thresholds for legal determination
        # await apply_legal_thresholds(fingerprint_id)

    except UnsupportedFileTypeError as e:
        logger.exception("Unsupported file type for fingerprinting")
        await _handle_fingerprint_error(
            fingerprint_id=fingerprint_id,
            asset_id=asset_id,
            error_message=f"Unsupported file type: {e}",
        )

    except FingerprintGenerationError as e:
        logger.exception("Fingerprint generation failed")
        await _handle_fingerprint_error(
            fingerprint_id=fingerprint_id,
            asset_id=asset_id,
            error_message=f"Fingerprint generation failed: {e}",
        )

    except Exception as e:
        logger.exception("Unexpected error during fingerprint generation")
        await _handle_fingerprint_error(
            fingerprint_id=fingerprint_id,
            asset_id=asset_id,
            error_message=f"Internal error during fingerprint generation: {e}",
        )


async def _handle_fingerprint_error(
    fingerprint_id: str,
    asset_id: str,
    error_message: str,
) -> None:
    """
    Handle fingerprint generation errors by updating status records.

    Updates both the fingerprint record and asset record to reflect the
    failed processing state with appropriate error messages.

    Args:
        fingerprint_id: MongoDB ObjectId of the fingerprint record
        asset_id: MongoDB ObjectId of the asset
        error_message: Human-readable error description
    """
    try:
        db_client = get_db_client()
        fingerprints_collection = db_client.get_fingerprints_collection()
        assets_collection = db_client.get_assets_collection()

        # Update fingerprint status to failed
        await fingerprints_collection.update_one(
            {"_id": fingerprint_id},
            {
                "$set": {
                    "processing_status": FingerprintProcessingStatus.FAILED.value,
                    "error_message": error_message[:1000],  # Limit error message length
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        # Update asset status to failed
        await assets_collection.update_one(
            {"_id": ObjectId(asset_id)},
            {
                "$set": {
                    "upload_status": UploadStatus.FAILED.value,
                    "processing_status": ProcessingStatus.FAILED.value,
                    "error_message": error_message[:1000],
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        logger.warning(
            f"Updated fingerprint {fingerprint_id} and asset {asset_id} to failed status"
        )

    except Exception:
        logger.exception("Failed to update error status")


# =============================================================================
# API ENDPOINTS
# =============================================================================


@router.post(
    "/",
    response_model=FingerprintStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate fingerprint for asset",
    description="""
    Initiates multi-modal fingerprint generation for a creative asset.

    The fingerprint generation is performed asynchronously in the background.
    Returns immediately with a fingerprint_id that can be used to check
    processing status via GET /{fingerprint_id}.

    Fingerprinting includes:
    - **Images**: Perceptual hashes (pHash, aHash, dHash) resistant to modifications
    - **Audio**: Spectral analysis (mel-spectrogram, chromagram, spectral centroid)
    - **Video**: Frame-based hashing at 1-second intervals
    - **Text**: SHA-256 content hash
    - **All types**: OpenAI embeddings for semantic similarity (if configured)

    Requirements:
    - Asset must exist and belong to the authenticated user
    - Asset must not already have an associated fingerprint
    - Asset upload status must be 'ready' or 'queued'
    """,
    responses={
        202: {
            "description": "Fingerprint generation queued successfully",
            "model": FingerprintStatusResponse,
        },
        400: {"description": "Invalid asset_id format"},
        403: {"description": "Not authorized to access this asset"},
        404: {"description": "Asset not found"},
        409: {"description": "Asset already has a fingerprint"},
    },
)
async def create_fingerprint(
    request: FingerprintRequest,
    background_tasks: BackgroundTasks,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> FingerprintStatusResponse:
    """
    Create a fingerprint for the specified asset.

    Validates the asset exists and belongs to the authenticated user,
    then queues fingerprint generation for background processing.

    Args:
        request: FingerprintRequest containing the asset_id
        background_tasks: FastAPI BackgroundTasks for async processing
        current_user: Authenticated user from JWT token (injected via Depends)

    Returns:
        FingerprintStatusResponse with fingerprint_id and processing status

    Raises:
        HTTPException(400): If asset_id format is invalid
        HTTPException(403): If user doesn't own the asset
        HTTPException(404): If asset not found
        HTTPException(409): If asset already has a fingerprint
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("Authenticated user missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    # Validate asset_id is a valid ObjectId format
    if not ObjectId.is_valid(request.asset_id):
        logger.warning(f"Invalid asset_id format: {request.asset_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid asset_id format. Must be a valid MongoDB ObjectId.",
        )

    # Get database client
    db_client = get_db_client()
    assets_collection = db_client.get_assets_collection()
    fingerprints_collection = db_client.get_fingerprints_collection()

    # Retrieve asset from database
    asset = await assets_collection.find_one({"_id": ObjectId(request.asset_id)})

    if asset is None:
        logger.warning(f"Asset not found: {request.asset_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset with ID {request.asset_id} not found",
        )

    # Verify user owns the asset
    asset_user_id = asset.get("user_id")
    if asset_user_id != user_id:
        logger.warning(
            f"User {user_id} attempted to access asset {request.asset_id} "
            f"owned by {asset_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this asset",
        )

    # Check if asset already has a fingerprint
    if asset.get("fingerprint_id"):
        existing_fingerprint_id = asset.get("fingerprint_id")
        logger.info(f"Asset {request.asset_id} already has fingerprint {existing_fingerprint_id}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Asset already has a fingerprint (ID: {existing_fingerprint_id}). "
            "Delete the existing fingerprint before generating a new one.",
        )

    # Check if a fingerprint record already exists for this asset
    existing_fingerprint = await fingerprints_collection.find_one({"asset_id": request.asset_id})
    if existing_fingerprint:
        existing_id = str(existing_fingerprint.get("_id", "unknown"))
        logger.info(f"Fingerprint already exists for asset {request.asset_id}: {existing_id}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A fingerprint already exists for this asset (ID: {existing_id})",
        )

    # Validate asset has required fields for fingerprinting
    s3_key = asset.get("s3_key")
    file_type = asset.get("file_type")

    if not s3_key:
        logger.error(f"Asset {request.asset_id} missing s3_key")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asset is missing storage reference (s3_key). "
            "Ensure the asset upload is complete.",
        )

    if not file_type:
        logger.error(f"Asset {request.asset_id} missing file_type")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asset is missing file type information. "
            "Ensure the asset was uploaded with a valid file type.",
        )

    # Generate fingerprint ID for the new record
    fingerprint_id = str(ObjectId())
    created_at = datetime.now(UTC)

    # Create placeholder fingerprint record with pending status
    fingerprint_doc = {
        "_id": fingerprint_id,
        "asset_id": request.asset_id,
        "user_id": user_id,
        "fingerprint_type": file_type,
        "processing_status": FingerprintProcessingStatus.PENDING.value,
        "perceptual_hashes": None,
        "spectral_data": None,
        "video_hashes": None,
        "embeddings": None,
        "image_metadata": None,
        "audio_metadata": None,
        "video_metadata": None,
        "text_hash": None,
        "text_length": None,
        "error_message": None,
        "processing_duration": None,
        "created_at": created_at,
        "updated_at": created_at,
        # Phase 2 placeholder fields
        # TODO Phase 2: Implement AI training detection engine
        "training_detected": None,
        # TODO Phase 2: Compare embeddings against known datasets
        "dataset_matches": None,
        # TODO Phase 2: Calculate embedding drift scores
        "similarity_scores": None,
        # TODO Phase 2: Apply similarity-law thresholds
        "legal_status": None,
    }

    # Insert placeholder record
    await fingerprints_collection.insert_one(fingerprint_doc)

    logger.info(f"Created fingerprint placeholder {fingerprint_id} for asset {request.asset_id}")

    # Update asset processing status
    await assets_collection.update_one(
        {"_id": ObjectId(request.asset_id)},
        {
            "$set": {
                "processing_status": ProcessingStatus.PROCESSING.value,
                "updated_at": datetime.now(UTC),
            }
        },
    )

    # Queue background task for fingerprint generation
    background_tasks.add_task(
        process_fingerprint_background,
        asset_id=request.asset_id,
        fingerprint_id=fingerprint_id,
        object_key=s3_key,
        file_type=file_type,
        user_id=user_id,
    )

    logger.info(
        f"Queued fingerprint generation for asset {request.asset_id} "
        f"(fingerprint_id={fingerprint_id})"
    )

    return FingerprintStatusResponse(
        fingerprint_id=fingerprint_id,
        status=FingerprintProcessingStatus.PENDING.value,
        message="Fingerprint generation has been queued for processing. "
        "Use GET /{fingerprint_id} to check status.",
        created_at=created_at,
    )


@router.get(
    "/{fingerprint_id}",
    response_model=FingerprintDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve fingerprint data",
    description="""
    Retrieves complete fingerprint data for the specified fingerprint ID.

    Returns all fingerprint data based on asset type:
    - **Images**: Perceptual hashes (pHash, aHash, dHash), image metadata (EXIF)
    - **Audio**: Spectral data (mel-spectrogram, chromagram), audio metadata
    - **Video**: Frame hashes, video metadata (resolution, fps, duration)
    - **Text**: Content hash, text statistics
    - **All types**: OpenAI embeddings if available, processing status

    The authenticated user must own the asset associated with the fingerprint.

    Phase 2 fields (training_detected, dataset_matches, similarity_scores,
    legal_status) are included for future AI training detection capabilities.
    """,
    responses={
        200: {
            "description": "Fingerprint data retrieved successfully",
            "model": FingerprintDetailResponse,
        },
        400: {"description": "Invalid fingerprint_id format"},
        403: {"description": "Not authorized to access this fingerprint"},
        404: {"description": "Fingerprint not found"},
    },
)
async def get_fingerprint(
    fingerprint_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> FingerprintDetailResponse:
    """
    Retrieve complete fingerprint data by ID.

    Fetches the fingerprint record from MongoDB, verifies the authenticated
    user owns the associated asset, and returns the complete fingerprint data.

    Args:
        fingerprint_id: MongoDB ObjectId of the fingerprint to retrieve
        current_user: Authenticated user from JWT token (injected via Depends)

    Returns:
        FingerprintDetailResponse with all fingerprint data

    Raises:
        HTTPException(400): If fingerprint_id format is invalid
        HTTPException(403): If user doesn't own the associated asset
        HTTPException(404): If fingerprint not found
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("Authenticated user missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    # Validate fingerprint_id format
    # MongoDB ObjectIds are 24 character hex strings, but we also accept
    # the generated string IDs from our placeholder records
    if not fingerprint_id or len(fingerprint_id) < MONGO_OBJECTID_LENGTH:
        logger.warning(f"Invalid fingerprint_id format: {fingerprint_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid fingerprint_id format. Must be a valid MongoDB ObjectId.",
        )

    # Get database client
    db_client = get_db_client()
    fingerprints_collection = db_client.get_fingerprints_collection()
    assets_collection = db_client.get_assets_collection()

    # Try to find fingerprint - first as string ID, then as ObjectId
    fingerprint = await fingerprints_collection.find_one({"_id": fingerprint_id})

    if fingerprint is None and ObjectId.is_valid(fingerprint_id):
        fingerprint = await fingerprints_collection.find_one({"_id": ObjectId(fingerprint_id)})

    if fingerprint is None:
        logger.warning(f"Fingerprint not found: {fingerprint_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fingerprint with ID {fingerprint_id} not found",
        )

    # Get the associated asset to verify ownership
    asset_id = fingerprint.get("asset_id")
    if not asset_id:
        logger.error(f"Fingerprint {fingerprint_id} missing asset_id reference")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fingerprint data is corrupt: missing asset reference",
        )

    # Check fingerprint user_id first (if available)
    fingerprint_user_id = fingerprint.get("user_id")
    if fingerprint_user_id and fingerprint_user_id != user_id:
        logger.warning(
            f"User {user_id} attempted to access fingerprint {fingerprint_id} "
            f"owned by {fingerprint_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this fingerprint",
        )

    # If fingerprint doesn't have user_id, verify via asset ownership
    if not fingerprint_user_id:
        # Retrieve asset to verify ownership
        asset = None
        if ObjectId.is_valid(asset_id):
            asset = await assets_collection.find_one({"_id": ObjectId(asset_id)})
        if asset is None:
            asset = await assets_collection.find_one({"_id": asset_id})

        if asset is None:
            logger.warning(f"Asset {asset_id} not found for fingerprint {fingerprint_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Associated asset not found",
            )

        # Verify user owns the asset
        asset_user_id = asset.get("user_id")
        if asset_user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to access fingerprint {fingerprint_id} "
                f"for asset owned by {asset_user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this fingerprint",
            )

    # Convert _id to string if it's an ObjectId
    fp_id = fingerprint.get("_id")
    if isinstance(fp_id, ObjectId):
        fp_id = str(fp_id)

    # Build response from fingerprint data
    logger.info(f"Returning fingerprint data for {fingerprint_id}")

    return FingerprintDetailResponse(
        fingerprint_id=fp_id or fingerprint_id,
        asset_id=str(asset_id),
        fingerprint_type=fingerprint.get("fingerprint_type", "unknown"),
        processing_status=fingerprint.get(
            "processing_status", FingerprintProcessingStatus.PENDING.value
        ),
        perceptual_hashes=fingerprint.get("perceptual_hashes"),
        spectral_data=fingerprint.get("spectral_data"),
        video_hashes=fingerprint.get("video_hashes"),
        embeddings=fingerprint.get("embeddings"),
        image_metadata=fingerprint.get("image_metadata"),
        audio_metadata=fingerprint.get("audio_metadata"),
        video_metadata=fingerprint.get("video_metadata"),
        text_hash=fingerprint.get("text_hash"),
        text_length=fingerprint.get("text_length"),
        processing_duration=fingerprint.get("processing_duration"),
        error_message=fingerprint.get("error_message"),
        created_at=fingerprint.get("created_at", datetime.now(UTC)),
        updated_at=fingerprint.get("updated_at"),
        # Phase 2 fields
        # TODO Phase 2: Implement AI training detection engine
        training_detected=fingerprint.get("training_detected"),
        # TODO Phase 2: Compare embeddings against known datasets
        dataset_matches=fingerprint.get("dataset_matches"),
        # TODO Phase 2: Calculate embedding drift scores
        similarity_scores=fingerprint.get("similarity_scores"),
        # TODO Phase 2: Apply similarity-law thresholds
        legal_status=fingerprint.get("legal_status"),
    )
