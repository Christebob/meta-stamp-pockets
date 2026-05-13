"""
Bulk YouTube channel ingestion API endpoints for META-STAMP V3.

Endpoints:
- POST /ingestion/channel: Start a bulk channel ingestion job
- GET /ingestion/jobs/{job_id}: Get job status
- GET /ingestion/jobs: List all ingestion jobs for creator
"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.core.auth import get_current_user
from app.models.ingestion import (
    IngestionJobCreateRequest,
    IngestionJobResponse,
)
from app.services.youtube_ingestion_service import (
    InvalidChannelURLError,
    YouTubeIngestionError,
    YouTubeIngestionService,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


def get_ingestion_service() -> YouTubeIngestionService:
    """Dependency provider for YouTubeIngestionService."""
    try:
        return YouTubeIngestionService()
    except YouTubeIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _resolve_creator_id(current_user: dict[str, Any]) -> str:
    """Resolve authenticated creator ID from current user context."""
    creator_id = current_user.get("_id") or current_user.get("id")
    if not isinstance(creator_id, str) or not creator_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user context",
        )
    return creator_id.strip()


@router.post(
    "/channel",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start channel ingestion",
    description=(
        "Start a bulk ingestion job for a YouTube channel. "
        "Returns immediately with job_id for tracking. "
        "Processing happens in the background."
    ),
)
async def start_channel_ingestion(
    request: IngestionJobCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict[str, Any] = Depends(get_current_user),
    ingestion_service: YouTubeIngestionService = Depends(get_ingestion_service),
) -> JSONResponse:
    """
    Start a bulk YouTube channel ingestion job.

    Creates an IngestionJob record and schedules background processing.
    Each video in the channel will be created as a Pocket with the
    specified price_per_pull (or default if not provided).
    """
    creator_id = _resolve_creator_id(current_user)

    try:
        # Create the job record
        job = await ingestion_service.ingest_channel(
            channel_url=request.channel_url,
            creator_id=creator_id,
            price_per_pull=request.price_per_pull,
        )

        # Schedule background processing
        background_tasks.add_task(
            ingestion_service.process_ingestion_job,
            job_id=job.id,
        )

        response = IngestionJobResponse.from_job(job)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response.model_dump(mode="json"),
        )

    except InvalidChannelURLError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except YouTubeIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError:
        logger.exception("Database unavailable during ingestion job creation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error starting ingestion job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while starting ingestion job",
        ) from None


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job status",
    description="Get the current status and progress of an ingestion job.",
)
async def get_job_status(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    ingestion_service: YouTubeIngestionService = Depends(get_ingestion_service),
) -> JSONResponse:
    """Get the status of a specific ingestion job."""
    creator_id = _resolve_creator_id(current_user)

    try:
        job = await ingestion_service.get_job_status(job_id)

        # Verify the job belongs to the authenticated creator
        if job.creator_id != creator_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ingestion job not found",
            )

        response = IngestionJobResponse.from_job(job)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response.model_dump(mode="json"),
        )

    except YouTubeIngestionError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError:
        logger.exception("Database unavailable while fetching job status")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error fetching job status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while fetching job status",
        ) from None


@router.get(
    "/jobs",
    response_model=list[IngestionJobResponse],
    status_code=status.HTTP_200_OK,
    summary="List ingestion jobs",
    description="List all ingestion jobs for the authenticated creator.",
)
async def list_ingestion_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: dict[str, Any] = Depends(get_current_user),
    ingestion_service: YouTubeIngestionService = Depends(get_ingestion_service),
) -> JSONResponse:
    """List creator-owned ingestion jobs in reverse chronological order."""
    creator_id = _resolve_creator_id(current_user)

    try:
        jobs = await ingestion_service.list_jobs(creator_id=creator_id, limit=limit)
        response_payload = [
            IngestionJobResponse.from_job(job).model_dump(mode="json") for job in jobs
        ]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_payload,
        )

    except RuntimeError:
        logger.exception("Database unavailable while listing jobs")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error listing jobs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while listing jobs",
        ) from None
