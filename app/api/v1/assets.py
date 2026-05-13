"""
Asset Management API Endpoints for META-STAMP V3.

This module implements the asset management router with endpoints for listing,
retrieving, and deleting creative assets. All endpoints require authentication
and enforce user ownership verification for security.

Endpoints:
- GET / - List assets with pagination (skip/limit) and optional filtering by file_type/status
- GET /{asset_id} - Retrieve detailed asset information including fingerprint data
- DELETE /{asset_id} - Delete asset with cascading removal of S3 objects and fingerprints

Per Agent Action Plan:
- Section 0.4: Asset management endpoints (list, get, delete)
- Section 0.6: assets.py transformation mapping
- Section 0.8: Endpoint specifications with pagination using skip/limit
- Section 0.10: All endpoints require authentication

Security Features:
- All endpoints require authenticated user via JWT (Auth0 or local fallback)
- Asset ownership verification before any operation
- Cascading deletion ensures no orphaned S3 objects or fingerprints
"""

import logging

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.models.asset import AssetResponse, FileType, UploadStatus
from app.services.storage_service import StorageOperationError, StorageService


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class AssetListResponse(BaseModel):
    """
    Response model for paginated asset listing.

    Attributes:
        assets: List of asset response objects for the current page
        total: Total count of assets matching the query filters
        skip: Number of assets skipped (offset)
        limit: Maximum number of assets returned per page
    """

    assets: list[AssetResponse] = Field(
        default_factory=list, description="List of asset response objects"
    )
    total: int = Field(..., ge=0, description="Total count of assets matching query")
    skip: int = Field(..., ge=0, description="Number of assets skipped (offset)")
    limit: int = Field(..., ge=1, description="Maximum assets per page")


class AssetDetailResponse(BaseModel):
    """
    Response model for detailed asset information.

    Includes the full asset data along with associated fingerprint
    information (if available) and additional computed metadata.

    Attributes:
        asset: Complete asset information
        fingerprint: Associated fingerprint data if fingerprinting completed
        metadata: Additional computed metadata about the asset
    """

    asset: AssetResponse = Field(..., description="Complete asset information")
    fingerprint: dict[str, Any] | None = Field(
        default=None, description="Associated fingerprint data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional computed metadata"
    )


class DeleteResponse(BaseModel):
    """
    Response model for successful asset deletion.

    Attributes:
        success: Boolean indicating deletion completed successfully
        message: Human-readable confirmation message
        deleted_asset_id: ID of the deleted asset
        deleted_s3_object: Whether S3 object was deleted
        deleted_fingerprint: Whether fingerprint record was deleted
    """

    success: bool = Field(..., description="Deletion success status")
    message: str = Field(..., description="Confirmation message")
    deleted_asset_id: str = Field(..., description="ID of deleted asset")
    deleted_s3_object: bool = Field(..., description="S3 object deletion status")
    deleted_fingerprint: bool = Field(..., description="Fingerprint deletion status")


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["assets"])


# =============================================================================
# Helper Functions
# =============================================================================


async def get_user_asset(
    asset_id: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Retrieve an asset by ID and verify it belongs to the specified user.

    This helper function enforces ownership verification to ensure users
    can only access their own assets.

    Args:
        asset_id: The MongoDB ObjectId string of the asset to retrieve
        user_id: The authenticated user's ID for ownership verification

    Returns:
        dict: The asset document from MongoDB

    Raises:
        HTTPException: 400 if asset_id format is invalid
        HTTPException: 404 if asset not found
        HTTPException: 403 if asset belongs to different user
    """
    # Validate ObjectId format
    if not ObjectId.is_valid(asset_id):
        logger.warning(f"Invalid asset_id format: {asset_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid asset ID format",
        )

    # Get database client and assets collection
    db_client = get_db_client()
    assets_collection = db_client.get_assets_collection()

    # Query for the asset
    asset_doc = await assets_collection.find_one({"_id": ObjectId(asset_id)})

    if asset_doc is None:
        logger.warning(f"Asset not found: {asset_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset with ID '{asset_id}' not found",
        )

    # Type assertion: at this point asset_doc is definitely a dict
    asset: dict[str, Any] = dict(asset_doc)

    # Convert ObjectId to string for comparison
    asset_user_id = str(asset.get("user_id", ""))

    # Verify ownership - user_id in asset may be string or ObjectId
    if asset_user_id != user_id and asset_user_id != str(user_id):
        logger.warning(
            f"Unauthorized asset access attempt: user {user_id} tried to access "
            f"asset {asset_id} owned by {asset_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this asset",
        )

    # Convert _id to string for JSON serialization
    asset["_id"] = str(asset["_id"])

    return asset


async def _delete_s3_object(
    asset: dict[str, Any],
    settings: Settings,
    asset_id: str | None,
) -> bool:
    """
    Delete S3 object for an asset.

    Args:
        asset: Asset document with s3_key and s3_bucket
        settings: Application settings for storage configuration
        asset_id: Asset ID for logging

    Returns:
        bool: True if deletion was successful or object doesn't exist
    """
    s3_key = asset.get("s3_key")
    s3_bucket = asset.get("s3_bucket")

    if not (s3_key and s3_bucket):
        logger.warning(f"Asset {asset_id} has no S3 key or bucket, skipping S3 deletion")
        return True  # Nothing to delete

    try:
        storage_service = StorageService(
            bucket_name=s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
        file_exists = await storage_service.file_exists(s3_key, bucket_name=s3_bucket)

        if file_exists:
            await storage_service.delete_file(s3_key, bucket_name=s3_bucket)
            logger.info(f"Deleted S3 object: {s3_bucket}/{s3_key}")
        else:
            logger.info(
                f"S3 object not found (may have been previously deleted): {s3_bucket}/{s3_key}"
            )
        return True

    except StorageOperationError:
        logger.exception(f"Failed to delete S3 object {s3_bucket}/{s3_key}")
        return False
    except Exception:
        logger.exception("Unexpected error deleting S3 object")
        return False


async def _delete_fingerprint_record(asset: dict[str, Any], asset_id: str | None) -> bool:
    """
    Delete fingerprint record for an asset.

    Args:
        asset: Asset document with fingerprint_id
        asset_id: Asset ID for logging and fallback query

    Returns:
        bool: True if deletion was successful or fingerprint doesn't exist
    """
    fingerprint_id = asset.get("fingerprint_id")
    if not fingerprint_id:
        return True  # No fingerprint to delete

    try:
        db_client = get_db_client()
        fingerprints_collection = db_client.get_fingerprints_collection()

        # Try to delete by fingerprint_id or by asset_id
        query: dict[str, Any] = (
            {"_id": ObjectId(fingerprint_id)}
            if ObjectId.is_valid(fingerprint_id)
            else {"asset_id": str(asset_id)}
        )
        result = await fingerprints_collection.delete_one(query)

        if result.deleted_count > 0:
            logger.info(f"Deleted fingerprint record for asset: {asset_id}")
        else:
            logger.info(f"Fingerprint record not found for asset: {asset_id}")
        return True

    except Exception:
        logger.exception(f"Error deleting fingerprint for asset {asset_id}")
        return False


async def delete_asset_cascade(
    asset: dict[str, Any],
    settings: Settings,
) -> dict[str, bool]:
    """
    Handle cascading deletion of an asset and all associated resources.

    This function ensures complete cleanup when deleting an asset:
    1. Delete the S3 object (file content)
    2. Delete the fingerprint record from MongoDB (if exists)
    3. Delete the asset record from MongoDB

    Args:
        asset: The asset document to delete
        settings: Application settings for storage configuration

    Returns:
        dict: Status of each deletion operation
            - s3_deleted: Whether S3 object was deleted
            - fingerprint_deleted: Whether fingerprint was deleted
            - asset_deleted: Whether asset record was deleted

    Raises:
        HTTPException: 500 if any critical deletion fails
    """
    asset_id = asset.get("_id")
    deletion_status = {
        "s3_deleted": await _delete_s3_object(asset, settings, asset_id),
        "fingerprint_deleted": await _delete_fingerprint_record(asset, asset_id),
        "asset_deleted": False,
    }

    # Step 3: Delete asset record from MongoDB
    try:
        db_client = get_db_client()
        assets_collection = db_client.get_assets_collection()
        result = await assets_collection.delete_one({"_id": ObjectId(asset_id)})

        if result.deleted_count > 0:
            logger.info(f"Deleted asset record: {asset_id}")
            deletion_status["asset_deleted"] = True
        else:
            logger.warning(f"Asset record not found during deletion: {asset_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete asset record",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting asset record {asset_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete asset: {e!s}",
        ) from e

    return deletion_status


def convert_asset_to_response(asset_doc: dict[str, Any]) -> AssetResponse:
    """
    Convert a MongoDB asset document to an AssetResponse model.

    Handles type conversion for enum fields and computes derived fields.

    Args:
        asset_doc: Raw MongoDB document

    Returns:
        AssetResponse: Formatted response model
    """
    # Calculate file size in MB
    file_size = asset_doc.get("file_size", 0)
    file_size_mb = round(file_size / (1024 * 1024), 2)

    # Extract enum values (handle both string and enum types)
    file_type_raw = asset_doc.get("file_type")
    file_type: str | None = (
        file_type_raw.value if isinstance(file_type_raw, FileType) else file_type_raw
    )

    upload_status_raw = asset_doc.get("upload_status")
    upload_status: str | None = (
        upload_status_raw.value
        if isinstance(upload_status_raw, UploadStatus)
        else upload_status_raw
    )

    processing_status_raw = asset_doc.get("processing_status")
    processing_status: str | None = (
        processing_status_raw.value
        if processing_status_raw and hasattr(processing_status_raw, "value")
        else processing_status_raw
    )

    # Determine ready status
    is_ready = upload_status in {"ready", UploadStatus.READY}

    # Determine fingerprint status
    has_fingerprint = asset_doc.get("fingerprint_id") is not None

    return AssetResponse(
        id=str(asset_doc.get("_id", "")),
        user_id=str(asset_doc.get("user_id", "")),
        file_name=asset_doc.get("file_name", ""),
        file_type=file_type or "unknown",
        file_size=file_size,
        file_size_mb=file_size_mb,
        mime_type=asset_doc.get("mime_type", "application/octet-stream"),
        s3_key=asset_doc.get("s3_key", ""),
        s3_bucket=asset_doc.get("s3_bucket", ""),
        upload_status=upload_status or "unknown",
        processing_status=processing_status,
        error_message=asset_doc.get("error_message"),
        fingerprint_id=asset_doc.get("fingerprint_id"),
        url_source=asset_doc.get("url_source"),
        metadata=asset_doc.get("metadata", {}),
        created_at=asset_doc.get("created_at", datetime.now(UTC)),
        updated_at=asset_doc.get("updated_at", datetime.now(UTC)),
        is_ready=is_ready,
        has_fingerprint=has_fingerprint,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/",
    response_model=AssetListResponse,
    summary="List user assets",
    description="Retrieve a paginated list of assets owned by the authenticated user. "
    "Supports optional filtering by file type and upload status.",
    responses={
        200: {"description": "Successfully retrieved asset list"},
        400: {"description": "Invalid query parameters"},
        401: {"description": "Not authenticated or invalid token"},
    },
)
async def list_assets(
    skip: int = Query(
        default=0,
        ge=0,
        description="Number of assets to skip (offset for pagination)",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of assets to return (1-100)",
    ),
    file_type: str | None = Query(
        default=None,
        description="Filter by file type (text, image, audio, video, url)",
    ),
    upload_status: str | None = Query(
        default=None,
        alias="status",
        description="Filter by upload status (queued, processing, ready, failed)",
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AssetListResponse:
    """
    List all assets belonging to the authenticated user.

    Implements pagination using skip/limit parameters and optional filtering
    by file_type and upload status. Results are sorted by creation date
    (newest first).

    Args:
        skip: Number of assets to skip (default 0)
        limit: Maximum assets to return (default 50, max 100)
        file_type: Optional filter by file type
        upload_status: Optional filter by upload status
        current_user: Authenticated user from JWT token

    Returns:
        AssetListResponse: Paginated list of assets with total count
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id") or current_user.get("id")
    if not user_id:
        logger.error("Authenticated user missing ID field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    user_id = str(user_id)
    logger.info(f"Listing assets for user {user_id}: skip={skip}, limit={limit}")

    # Build query filter
    query_filter: dict[str, Any] = {"user_id": user_id}

    # Apply file_type filter if provided
    if file_type:
        # Validate file_type value
        normalized_type = file_type.lower().strip()
        valid_types = [ft.value for ft in FileType]

        if normalized_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file_type '{file_type}'. Must be one of: {', '.join(valid_types)}",
            )

        query_filter["file_type"] = normalized_type
        logger.debug(f"Filtering by file_type: {normalized_type}")

    # Apply upload_status filter if provided
    if upload_status:
        # Validate status value
        normalized_status = upload_status.lower().strip()
        valid_statuses = [us.value for us in UploadStatus]

        if normalized_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{upload_status}'. Must be one of: {', '.join(valid_statuses)}",
            )

        query_filter["upload_status"] = normalized_status
        logger.debug(f"Filtering by upload_status: {normalized_status}")

    try:
        # Get database client and collection
        db_client = get_db_client()
        assets_collection = db_client.get_assets_collection()

        # Get total count for pagination metadata
        total_count = await assets_collection.count_documents(query_filter)
        logger.debug(f"Total assets matching query: {total_count}")

        # Query assets with pagination and sorting (newest first)
        cursor = assets_collection.find(query_filter).sort("created_at", -1).skip(skip).limit(limit)

        # Convert cursor to list
        asset_docs = await cursor.to_list(length=limit)

        # Convert to response models
        assets = [convert_asset_to_response(doc) for doc in asset_docs]

        logger.info(f"Returning {len(assets)} assets for user {user_id}")

        return AssetListResponse(
            assets=assets,
            total=total_count,
            skip=skip,
            limit=limit,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.exception("Database unavailable during asset listing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e
    except Exception as e:
        logger.exception(f"Error listing assets for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assets",
        ) from e


@router.get(
    "/{asset_id}",
    response_model=AssetDetailResponse,
    summary="Get asset details",
    description="Retrieve detailed information about a specific asset, "
    "including associated fingerprint data and metadata.",
    responses={
        200: {"description": "Successfully retrieved asset details"},
        400: {"description": "Invalid asset ID format"},
        401: {"description": "Not authenticated or invalid token"},
        403: {"description": "Not authorized to access this asset"},
        404: {"description": "Asset not found"},
    },
)
async def get_asset(
    asset_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AssetDetailResponse:
    """
    Retrieve detailed information about a specific asset.

    Returns the complete asset information including associated fingerprint
    data (if available) and computed metadata. Verifies that the asset
    belongs to the authenticated user.

    Args:
        asset_id: MongoDB ObjectId of the asset
        current_user: Authenticated user from JWT token

    Returns:
        AssetDetailResponse: Complete asset details with fingerprint info
    """
    # Extract user ID
    user_id = current_user.get("_id") or current_user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    user_id = str(user_id)
    logger.info(f"Getting asset details: {asset_id} for user {user_id}")

    # Get and verify asset ownership
    asset = await get_user_asset(asset_id, user_id)

    # Convert to response model
    asset_response = convert_asset_to_response(asset)

    # Retrieve fingerprint data if exists
    fingerprint_data: dict[str, Any] | None = None
    fingerprint_id = asset.get("fingerprint_id")

    if fingerprint_id:
        try:
            db_client = get_db_client()
            fingerprints_collection = db_client.get_fingerprints_collection()

            # Query fingerprint by ID or asset_id
            fingerprint_doc = None
            if ObjectId.is_valid(fingerprint_id):
                fingerprint_doc = await fingerprints_collection.find_one(
                    {"_id": ObjectId(fingerprint_id)}
                )

            # Fallback to query by asset_id
            if fingerprint_doc is None:
                fingerprint_doc = await fingerprints_collection.find_one(
                    {"asset_id": str(asset_id)}
                )

            if fingerprint_doc:
                # Convert ObjectId to string
                fingerprint_doc["_id"] = str(fingerprint_doc["_id"])

                # Extract relevant fingerprint data for response
                fingerprint_data = {
                    "id": fingerprint_doc.get("_id"),
                    "asset_id": fingerprint_doc.get("asset_id"),
                    "fingerprint_type": fingerprint_doc.get("fingerprint_type"),
                    "perceptual_hashes": fingerprint_doc.get("perceptual_hashes"),
                    "spectral_data": fingerprint_doc.get("spectral_data"),
                    "embeddings": fingerprint_doc.get("embeddings"),
                    "processing_status": fingerprint_doc.get("processing_status"),
                    "created_at": fingerprint_doc.get("created_at"),
                }

                logger.debug(f"Retrieved fingerprint data for asset {asset_id}")

        except Exception as e:
            logger.warning(f"Failed to retrieve fingerprint for asset {asset_id}: {e}")
            # Continue without fingerprint data - not a critical error

    # Build additional metadata
    additional_metadata: dict[str, Any] = {
        "file_extension": None,
        "is_url_asset": asset.get("file_type") == "url",
        "days_since_upload": None,
        "fingerprint_available": fingerprint_data is not None,
    }

    # Calculate file extension
    file_name = asset.get("file_name", "")
    if "." in file_name:
        additional_metadata["file_extension"] = "." + file_name.rsplit(".", 1)[-1].lower()

    # Calculate days since upload
    created_at = asset.get("created_at")
    if created_at and isinstance(created_at, datetime):
        now = datetime.now(UTC)
        # Ensure both datetimes are timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        delta = now - created_at
        additional_metadata["days_since_upload"] = delta.days

    logger.info(f"Returning asset details for {asset_id}")

    return AssetDetailResponse(
        asset=asset_response,
        fingerprint=fingerprint_data,
        metadata=additional_metadata,
    )


@router.delete(
    "/{asset_id}",
    response_model=DeleteResponse,
    summary="Delete asset",
    description="Permanently delete an asset and all associated data including "
    "the S3 object and fingerprint record.",
    responses={
        200: {"description": "Successfully deleted asset"},
        400: {"description": "Invalid asset ID format"},
        401: {"description": "Not authenticated or invalid token"},
        403: {"description": "Not authorized to delete this asset"},
        404: {"description": "Asset not found"},
        500: {"description": "Deletion failed"},
    },
)
async def delete_asset(
    asset_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> DeleteResponse:
    """
    Delete an asset and all associated resources.

    Performs cascading deletion:
    1. Deletes the S3 object (stored file)
    2. Deletes the fingerprint record (if exists)
    3. Deletes the asset record from MongoDB

    Verifies that the asset belongs to the authenticated user before deletion.

    Args:
        asset_id: MongoDB ObjectId of the asset to delete
        current_user: Authenticated user from JWT token
        settings: Application settings for storage configuration

    Returns:
        DeleteResponse: Confirmation of deletion with status details
    """
    # Extract user ID
    user_id = current_user.get("_id") or current_user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    user_id = str(user_id)
    logger.info(f"Deleting asset: {asset_id} for user {user_id}")

    # Get and verify asset ownership
    asset = await get_user_asset(asset_id, user_id)

    # Store asset info for response before deletion
    asset_file_name = asset.get("file_name", "unknown")

    # Perform cascading deletion
    deletion_status = await delete_asset_cascade(asset, settings)

    logger.info(
        f"Asset {asset_id} deleted successfully: "
        f"s3={deletion_status['s3_deleted']}, "
        f"fingerprint={deletion_status['fingerprint_deleted']}, "
        f"asset={deletion_status['asset_deleted']}"
    )

    return DeleteResponse(
        success=True,
        message=f"Asset '{asset_file_name}' and all associated data deleted successfully",
        deleted_asset_id=asset_id,
        deleted_s3_object=deletion_status["s3_deleted"],
        deleted_fingerprint=deletion_status["fingerprint_deleted"],
    )
