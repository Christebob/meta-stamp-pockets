"""
MCP middleware — authentication, agreement, and rate limiting for AI agents.

These are FastAPI dependencies that run on every MCP request before
the tool handler executes.
"""

import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.redis_client import get_redis_client
from app.models.agent import AgentAPIKey
from app.services.agent_auth_service import AgentAuthService, AgentKeyNotFoundError
from app.services.agreement_service import AgreementService


logger = logging.getLogger(__name__)

# Bearer token extractor for MCP routes
# auto_error=False so we can allow unauthenticated initialize/tools/list requests through
mcp_security = HTTPBearer(
    scheme_name="MCP Bearer",
    description="Agent API key for MCP authentication. Format: Bearer pkt_...",
    auto_error=False,
)

# Methods that are allowed without authentication (for health checks and discovery)
PUBLIC_METHODS = {"initialize", "tools/list"}


async def get_current_agent(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(mcp_security),
) -> AgentAPIKey | None:
    """
    FastAPI dependency: authenticate the AI agent via API key.

    Extracts the Bearer token, validates it against the agent_keys collection,
    and returns the AgentAPIKey model.

    Public methods (initialize, tools/list) are allowed through without auth.

    Raises:
        HTTPException 401: If the API key is invalid or inactive.
    """
    # Check if this is a public method by peeking at the request body
    try:
        body = await request.json()
        method = body.get("method", "") if isinstance(body, dict) else ""
    except Exception:
        method = ""

    # Allow public methods through without auth
    if method in PUBLIC_METHODS:
        return None

    # Require auth for all other methods
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_key = credentials.credentials

    auth_service = AgentAuthService()
    try:
        agent = await auth_service.validate_api_key(raw_key)
        # Store agent in request state for downstream use
        request.state.agent = agent
        return agent
    except AgentKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive agent API key",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def check_agreement(
    request: Request,
    agent: AgentAPIKey | None = Depends(get_current_agent),
) -> None:
    """
    FastAPI dependency: ensure the agent has accepted current terms.

    If no agreement exists for the current terms version, one is created
    automatically. This implements the "iTunes agreement" model where
    connection = acceptance.
    """
    agreement_service = AgreementService()

    # Get client IP and user-agent for the agreement record
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    if agent is None:
        return  # Public method — skip agreement check

    await agreement_service.ensure_agreement(
        agent_key_id=agent.id or "",
        provider=agent.provider.value if hasattr(agent.provider, "value") else str(agent.provider),
        ip_address=client_ip,
        user_agent=user_agent,
    )


async def check_rate_limit(
    _request: Request,
    agent: AgentAPIKey | None = Depends(get_current_agent),
) -> None:
    """
    FastAPI dependency: enforce per-agent rate limiting.

    Uses a sliding window counter in Redis. If the agent exceeds their
    rate_limit_per_minute, a 429 Too Many Requests error is raised.
    """
    if agent is None:
        return  # Public method — skip rate limiting

    redis_client = get_redis_client()
    rate_key = f"rate_limit:{agent.key_prefix}"

    try:
        # Increment counter
        current = await redis_client.get(rate_key)
        current_count = int(current) if current else 0

        if current_count >= agent.rate_limit_per_minute:
            logger.warning(
                "Rate limit exceeded for agent %s: %d/%d",
                agent.key_prefix,
                current_count,
                agent.rate_limit_per_minute,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {agent.rate_limit_per_minute} requests per minute",
                headers={"Retry-After": "60"},
            )

        # Increment and set TTL
        if current is None:
            await redis_client.set(rate_key, "1", ttl=60)
        else:
            # Use raw Redis client for INCR
            if redis_client._client:
                await redis_client._client.incr(rate_key)
    except HTTPException:
        raise
    except Exception:
        # Don't block requests if rate limiting fails
        logger.exception("Rate limiting error for agent %s", agent.key_prefix)
