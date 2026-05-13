"""
Ingestion job models for META-STAMP V3 bulk YouTube channel processing.

This module defines data models for tracking bulk YouTube channel ingestion jobs,
which process entire YouTube channels by creating a Pocket for each video.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


class IngestionJobStatus(StrEnum):
    """Status values for bulk ingestion job lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    """
    Core IngestionJob document model stored in MongoDB.

    Attributes:
        id: MongoDB document ID (alias of _id)
        creator_id: Authenticated creator/user identifier
        channel_url: YouTube channel URL provided by creator
        channel_name: Resolved channel name from YouTube API
        channel_id: YouTube channel ID extracted from URL
        total_videos: Total number of videos to process in channel
        completed: Number of videos successfully processed into Pockets
        failed: Number of videos that failed during processing
        status: Job processing lifecycle state
        price_per_pull: Custom price per pull for this ingestion (USD)
        error_message: Job-level error details when status is failed
        created_at: Job creation timestamp (UTC)
        updated_at: Job last update timestamp (UTC)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda value: value.isoformat()},
        validate_default=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    creator_id: str = Field(..., min_length=1, max_length=100, description="Creator ID")
    channel_url: str = Field(
        ..., min_length=1, max_length=2048, description="YouTube channel URL"
    )
    channel_name: str | None = Field(
        default=None, max_length=200, description="Resolved channel name from YouTube"
    )
    channel_id: str | None = Field(
        default=None, max_length=100, description="YouTube channel ID"
    )
    total_videos: int = Field(default=0, ge=0, description="Total videos in channel")
    completed: int = Field(default=0, ge=0, description="Successfully processed videos")
    failed: int = Field(default=0, ge=0, description="Failed video processing count")
    status: IngestionJobStatus = Field(
        default=IngestionJobStatus.PENDING, description="Job status"
    )
    price_per_pull: float = Field(
        default=0.01, ge=0.0, description="Custom price per pull for this job (USD)"
    )
    error_message: str | None = Field(
        default=None, max_length=2000, description="Job-level error details"
    )
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp (UTC)")
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp (UTC)",
    )


class IngestionJobCreateRequest(BaseModel):
    """Request schema for creating a new bulk ingestion job."""

    channel_url: str = Field(..., min_length=1, description="YouTube channel URL to ingest")
    price_per_pull: float | None = Field(
        default=None, ge=0.0, description="Custom price per pull (optional, defaults to config)"
    )


class IngestionJobResponse(BaseModel):
    """Ingestion job API response schema."""

    model_config = ConfigDict(
        json_encoders={datetime: lambda value: value.isoformat()},
    )

    id: str = Field(..., description="Job identifier")
    creator_id: str = Field(..., description="Creator identifier")
    channel_url: str = Field(..., description="YouTube channel URL")
    channel_name: str | None = Field(default=None, description="Channel name")
    channel_id: str | None = Field(default=None, description="YouTube channel ID")
    total_videos: int = Field(..., ge=0, description="Total videos to process")
    completed: int = Field(..., ge=0, description="Completed video count")
    failed: int = Field(..., ge=0, description="Failed video count")
    status: str = Field(..., description="Job status")
    price_per_pull: float = Field(..., ge=0.0, description="Price per pull (USD)")
    error_message: str | None = Field(default=None, description="Error details if failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    @classmethod
    def from_job(cls, job: IngestionJob) -> "IngestionJobResponse":
        """Create IngestionJobResponse from an IngestionJob model instance."""
        if job.id is None:
            raise ValueError("IngestionJob ID is required for response serialization")

        return cls(
            id=job.id,
            creator_id=job.creator_id,
            channel_url=job.channel_url,
            channel_name=job.channel_name,
            channel_id=job.channel_id,
            total_videos=job.total_videos,
            completed=job.completed,
            failed=job.failed,
            status=(
                job.status.value if isinstance(job.status, IngestionJobStatus) else job.status
            ),
            price_per_pull=job.price_per_pull,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
