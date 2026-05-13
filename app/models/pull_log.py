"""
Pull log models for META-STAMP V3 Pockets audit trail.

Every content pull by an AI agent is logged for:
- Creator compensation calculation
- Usage analytics and reporting
- Legal audit trail
- Rate limiting enforcement
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PullLog(BaseModel):
    """
    Audit record for a single content pull by an AI agent.

    Logged asynchronously (fire-and-forget) to avoid blocking the hot path.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda value: value.isoformat()},
        validate_default=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    pocket_id: str = Field(..., min_length=1, description="Pocket that was pulled")
    agent_key_id: str = Field(..., min_length=1, description="Agent API key used")
    creator_id: str = Field(..., min_length=1, description="Creator who owns the pocket")
    provider: str = Field(..., min_length=1, max_length=50, description="Agent provider")
    compensation_amount: float = Field(
        ..., ge=0.0, description="Amount credited to creator for this pull"
    )
    response_time_ms: float = Field(
        default=0.0, ge=0.0, description="Server-side response time in milliseconds"
    )
    pulled_at: datetime = Field(default_factory=_utc_now, description="Pull timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional pull metadata")
