"""
Pocket models for META-STAMP V3 creator content snapshots.

This module defines data models for the Pockets feature, which lets creators
pre-index URL content into retrievable snapshots with pull-based compensation.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def _utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


class PocketStatus(StrEnum):
    """Lifecycle status values for a creator pocket."""

    INDEXING = "indexing"
    ACTIVE = "active"
    FAILED = "failed"


class Pocket(BaseModel):
    """
    Core Pocket document model stored in MongoDB.

    Attributes:
        id: MongoDB document ID (alias of _id)
        creator_id: Authenticated creator/user identifier
        content_url: Source URL submitted by the creator
        content_type: Resolved source type (youtube, vimeo, webpage, unknown)
        status: Pocket indexing lifecycle state
        pull_count: Number of simulated AI retrieval pulls
        compensation_earned: Total compensation accumulated from pulls
        snapshot_text: Pre-indexed text snapshot extracted from source URL
        error_message: Indexing error details when status is failed
        source_metadata: Small metadata payload from URL processing
        created_at: Pocket creation timestamp (UTC)
        updated_at: Pocket last update timestamp (UTC)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda value: value.isoformat()},
        validate_default=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    creator_id: str = Field(..., min_length=1, max_length=100, description="Creator ID")
    content_url: str = Field(..., min_length=1, max_length=2048, description="Source content URL")
    content_type: str = Field(
        default="unknown", min_length=1, max_length=50, description="Resolved content source type"
    )
    status: PocketStatus = Field(default=PocketStatus.INDEXING, description="Pocket status")
    pull_count: int = Field(default=0, ge=0, description="Number of content pulls")
    compensation_earned: float = Field(
        default=0.0, ge=0.0, description="Total compensation earned in USD"
    )
    price_per_pull: float = Field(
        default=0.0025, ge=0.0, description="USD price charged per AI pull"
    )
    snapshot_text: str | None = Field(
        default=None, description="Pre-indexed snapshot content for retrieval"
    )
    error_message: str | None = Field(
        default=None, max_length=1000, description="Error details if indexing failed"
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata captured during URL processing"
    )
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp (UTC)")
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp (UTC)",
    )

    @field_validator("content_url")
    @classmethod
    def validate_content_url(cls, value: str) -> str:
        """Ensure the pocket source URL is HTTP(S)."""
        normalized = value.strip()
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("content_url must start with http:// or https://")
        return normalized

    @field_validator("compensation_earned")
    @classmethod
    def validate_compensation_earned(cls, value: float) -> float:
        """Normalize floating compensation values to avoid precision noise."""
        rounded = round(float(value), 6)
        if rounded < 0:
            raise ValueError("compensation_earned cannot be negative")
        return rounded


class PocketCreateRequest(BaseModel):
    """Request schema for creating a new pocket from a URL."""

    content_url: HttpUrl = Field(..., description="Creator content URL to pre-index")


class PocketResponse(BaseModel):
    """Pocket API response schema."""

    model_config = ConfigDict(
        json_encoders={datetime: lambda value: value.isoformat()},
    )

    id: str = Field(..., description="Pocket identifier")
    creator_id: str = Field(..., description="Creator identifier")
    content_url: str = Field(..., description="Source content URL")
    content_type: str = Field(..., description="Source type")
    status: str = Field(..., description="Pocket status")
    pull_count: int = Field(..., ge=0, description="Current pull count")
    compensation_earned: float = Field(..., ge=0.0, description="Total compensation earned")
    price_per_pull: float = Field(default=0.0025, ge=0.0, description="USD price per AI pull")
    snapshot_text: str | None = Field(default=None, description="Pre-indexed snapshot content")
    error_message: str | None = Field(default=None, description="Indexing error details")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    @classmethod
    def from_pocket(cls, pocket: Pocket) -> "PocketResponse":
        """Create PocketResponse from a Pocket model instance."""
        if pocket.id is None:
            raise ValueError("Pocket ID is required for response serialization")

        return cls(
            id=pocket.id,
            creator_id=pocket.creator_id,
            content_url=pocket.content_url,
            content_type=pocket.content_type,
            status=(
                pocket.status.value
                if isinstance(pocket.status, PocketStatus)
                else pocket.status
            ),
            pull_count=pocket.pull_count,
            compensation_earned=pocket.compensation_earned,
            price_per_pull=pocket.price_per_pull,
            snapshot_text=pocket.snapshot_text,
            error_message=pocket.error_message,
            created_at=pocket.created_at,
            updated_at=pocket.updated_at,
        )


class PocketPullResponse(BaseModel):
    """Response schema for a simulated pocket pull operation."""

    pocket: PocketResponse = Field(..., description="Updated pocket after pull")
    retrieved_content: str = Field(..., description="Snapshot text returned to the AI agent")
    compensation_increment: float = Field(
        ..., ge=0.0, description="Compensation added by this pull operation"
    )
