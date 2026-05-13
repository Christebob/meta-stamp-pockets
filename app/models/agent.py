"""
Agent API key models for META-STAMP V3 Pockets MCP authentication.

AI agent companies (OpenAI, Anthropic, Google, etc.) authenticate to the
MCP server using API keys. This module defines the AgentAPIKey model for
storing and validating agent credentials.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AgentProvider(StrEnum):
    """Known AI agent providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    PERPLEXITY = "perplexity"
    CUSTOM = "custom"


class AgentAPIKey(BaseModel):
    """
    API key record for an AI agent connecting to the MCP server.

    Each AI company gets one or more API keys. The key_hash is stored
    (never the raw key). Rate limits are per-key.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda value: value.isoformat()},
        validate_default=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    key_hash: str = Field(
        ..., min_length=64, max_length=128, description="SHA-256 hash of the API key"
    )
    key_prefix: str = Field(
        ..., min_length=8, max_length=12, description="First 8 chars of key for identification"
    )
    provider: AgentProvider = Field(..., description="AI agent provider")
    provider_name: str = Field(..., min_length=1, max_length=100, description="Human-readable provider name")
    rate_limit_per_minute: int = Field(
        default=100, ge=1, le=10000, description="Max requests per minute"
    )
    is_active: bool = Field(default=True, description="Whether this key is currently active")
    created_at: datetime = Field(default_factory=_utc_now, description="Key creation timestamp")
    last_used_at: datetime | None = Field(
        default=None, description="Last successful authentication timestamp"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional key metadata")

    @field_validator("key_hash")
    @classmethod
    def validate_key_hash(cls, value: str) -> str:
        """Ensure key_hash looks like a hex-encoded SHA-256 digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64:
            raise ValueError("key_hash must be a 64-character hex string (SHA-256)")
        try:
            int(normalized, 16)
        except ValueError as exc:
            raise ValueError("key_hash must be a valid hexadecimal string") from exc
        return normalized


class AgentAPIKeyCreateRequest(BaseModel):
    """Request schema for creating a new agent API key."""

    provider: AgentProvider = Field(..., description="AI agent provider")
    provider_name: str = Field(..., min_length=1, max_length=100, description="Human-readable provider name")
    rate_limit_per_minute: int = Field(
        default=100, ge=1, le=10000, description="Max requests per minute"
    )


class AgentAPIKeyResponse(BaseModel):
    """Response schema for agent API key (never exposes the raw key after creation)."""

    model_config = ConfigDict(json_encoders={datetime: lambda value: value.isoformat()})

    id: str = Field(..., description="Key identifier")
    key_prefix: str = Field(..., description="First 8 chars of key for identification")
    provider: str = Field(..., description="AI agent provider")
    provider_name: str = Field(..., description="Human-readable provider name")
    rate_limit_per_minute: int = Field(..., description="Max requests per minute")
    is_active: bool = Field(..., description="Whether key is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used_at: datetime | None = Field(default=None, description="Last use timestamp")
