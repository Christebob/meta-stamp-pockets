"""
Agreement models for META-STAMP V3 Pockets "iTunes agreement" system.

When an AI agent connects to the MCP server, the connection itself constitutes
acceptance of the content license terms. No human negotiation required.
The protocol IS the contract.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


CURRENT_TERMS_VERSION = "1.0.0"


class Agreement(BaseModel):
    """
    Agreement record created when an AI agent first connects to the MCP server.

    Connection = acceptance. This document is the legal record that the agent's
    provider accepted the content license terms at a specific time.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda value: value.isoformat()},
        validate_default=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    agent_key_id: str = Field(..., min_length=1, description="Reference to AgentAPIKey._id")
    provider: str = Field(..., min_length=1, max_length=50, description="Agent provider name")
    terms_version: str = Field(default=CURRENT_TERMS_VERSION, description="Version of terms accepted")
    accepted_at: datetime = Field(default_factory=_utc_now, description="When agreement was accepted")
    ip_address: str | None = Field(
        default=None, max_length=45, description="IP address at time of acceptance"
    )
    user_agent: str | None = Field(
        default=None, max_length=500, description="User-Agent header at acceptance"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional agreement metadata")

    @field_validator("terms_version")
    @classmethod
    def validate_terms_version(cls, value: str) -> str:
        """Ensure terms_version follows semver-like format."""
        normalized = value.strip()
        parts = normalized.split(".")
        if len(parts) != 3:
            raise ValueError("terms_version must follow X.Y.Z format")
        return normalized


class AgreementResponse(BaseModel):
    """Response schema for agreement status."""

    model_config = ConfigDict(json_encoders={datetime: lambda value: value.isoformat()})

    id: str = Field(..., description="Agreement identifier")
    agent_key_id: str = Field(..., description="Agent API key identifier")
    provider: str = Field(..., description="Agent provider")
    terms_version: str = Field(..., description="Terms version accepted")
    accepted_at: datetime = Field(..., description="Acceptance timestamp")


class TermsResponse(BaseModel):
    """Response schema for current terms of service."""

    version: str = Field(..., description="Current terms version")
    effective_date: str = Field(..., description="When these terms became effective (ISO 8601 date)")
    freshness_window_days: int = Field(
        ...,
        description="Number of days after which cached content access expires and must be re-pulled",
    )
    summary: str = Field(..., description="Brief summary of terms")
    key_terms: dict[str, Any] = Field(..., description="Structured key terms for machine consumption")
    full_text: str = Field(..., description="Full terms text")
