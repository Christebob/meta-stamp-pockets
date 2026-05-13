"""
Agent API key management endpoints for META-STAMP V3 Pockets.

Admin endpoints for creating and managing agent API keys.
These are used by the Meta-Stamp team to onboard AI companies.
"""

import logging

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.core.auth import get_current_user
from app.models.agent import AgentAPIKeyCreateRequest, AgentAPIKeyResponse
from app.services.agent_auth_service import AgentAuthService


logger = logging.getLogger(__name__)
router = APIRouter(tags=["agents"])


def get_agent_auth_service() -> AgentAuthService:
    """Dependency provider for AgentAuthService."""
    return AgentAuthService()


@router.post(
    "/keys",
    status_code=status.HTTP_201_CREATED,
    summary="Create agent API key",
    description="Create a new API key for an AI agent provider. The raw key is only returned once.",
)
async def create_agent_key(
    request: AgentAPIKeyCreateRequest,
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentAuthService = Depends(get_agent_auth_service),
) -> JSONResponse:
    """Create a new agent API key. Returns the raw key only once."""
    try:
        raw_key, agent_key = await service.create_agent_key(
            provider=request.provider,
            provider_name=request.provider_name,
            rate_limit_per_minute=request.rate_limit_per_minute,
        )
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "raw_key": raw_key,  # Only shown once!
                "key": AgentAPIKeyResponse(
                    id=agent_key.id or "",
                    key_prefix=agent_key.key_prefix,
                    provider=(
                        agent_key.provider.value
                        if hasattr(agent_key.provider, "value")
                        else str(agent_key.provider)
                    ),
                    provider_name=agent_key.provider_name,
                    rate_limit_per_minute=agent_key.rate_limit_per_minute,
                    is_active=agent_key.is_active,
                    created_at=agent_key.created_at,
                    last_used_at=agent_key.last_used_at,
                ).model_dump(mode="json"),
            },
        )
    except Exception:
        logger.exception("Error creating agent API key")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent API key",
        ) from None


@router.get(
    "/keys",
    summary="List agent API keys",
    description="List all agent API keys, optionally filtered by provider.",
)
async def list_agent_keys(
    provider: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentAuthService = Depends(get_agent_auth_service),
) -> JSONResponse:
    """List agent API keys."""
    try:
        keys = await service.list_agent_keys(provider=provider, limit=limit)
        response = [
            AgentAPIKeyResponse(
                id=key.id or "",
                key_prefix=key.key_prefix,
                provider=key.provider.value if hasattr(key.provider, "value") else str(key.provider),
                provider_name=key.provider_name,
                rate_limit_per_minute=key.rate_limit_per_minute,
                is_active=key.is_active,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
            ).model_dump(mode="json")
            for key in keys
        ]
        return JSONResponse(content=response)
    except Exception:
        logger.exception("Error listing agent keys")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list agent API keys",
        ) from None


@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_200_OK,
    summary="Deactivate agent API key",
    description="Deactivate an agent API key (soft delete).",
)
async def deactivate_agent_key(
    key_id: str,
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentAuthService = Depends(get_agent_auth_service),
) -> JSONResponse:
    """Deactivate an agent API key."""
    success = await service.deactivate_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent key not found",
        )
    return JSONResponse(content={"status": "deactivated", "key_id": key_id})
