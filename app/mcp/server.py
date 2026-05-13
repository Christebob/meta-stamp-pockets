"""
MCP Server core — JSON-RPC 2.0 handler mounted as FastAPI router.

This is the main entry point for AI agents connecting to Pockets.
It handles JSON-RPC 2.0 requests and routes them to the appropriate
tool functions.

Transport: MCP Streamable HTTP (2025-03-26) — POST /mcp always returns
           Content-Type: text/event-stream with SSE-wrapped JSON-RPC responses.
Auth: Bearer token with agent API key.
"""

import json
import logging
import time
import uuid

from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.mcp.middleware import check_agreement, check_rate_limit, get_current_agent
from app.mcp.tools import MCPTools
from app.models.agent import AgentAPIKey


logger = logging.getLogger(__name__)

mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _jsonrpc_error(code: int, message: str, request_id: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id,
    }


def _jsonrpc_success(result: Any, request_id: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id,
    }


def _sse_response(payload: dict[str, Any], session_id: str, status_code: int = 200) -> StreamingResponse:
    """
    Wrap a JSON-RPC payload in an SSE StreamingResponse.

    Per MCP Streamable HTTP spec (2025-03-26), all POST /mcp responses MUST be
    returned as Content-Type: text/event-stream with the body formatted as:
        data: {json}\n\n
    The Mcp-Session-Id header MUST be present on every response.
    """
    async def _stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        _stream(),
        status_code=status_code,
        media_type="text/event-stream",
        headers={
            "Mcp-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# MCP tool registry — maps method names to descriptions for tool discovery
MCP_TOOL_MANIFEST = {
    "tools": [
        {
            "name": "pull_content",
            "description": (
                "Pull licensed creator content from a specific pocket by ID. Use this tool "
                "when an AI agent needs to retrieve verified, provenance-tracked content for "
                "generation, RAG, or training purposes. Do NOT use for browsing or discovery "
                "\u2014 use search_pockets or list_pockets instead. Requires a valid Bearer token "
                "for authentication; unauthenticated requests return HTTP 401. Successful "
                "pulls trigger a metered charge ($0.001\u2013$0.25 depending on content tier) "
                "and the transaction is logged for creator royalty distribution. The "
                "pocket_id parameter is a 24-character hex string identifying the specific "
                "content pocket to pull from. Returns the full content payload with "
                "provenance metadata including creator attribution and license terms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pocket_id": {
                        "type": "string",
                        "description": (
                            "The 24-character hex identifier of the content pocket to pull from. "
                            "Obtain this from search_pockets or list_pockets results."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional natural language focus query used to narrow or contextualize "
                            "the returned content within the selected pocket."
                        ),
                    },
                },
                "required": ["pocket_id"],
            },
        },
        {
            "name": "search_pockets",
            "description": (
                "Search the Meta-Stamp Pockets catalog by keyword, creator name, or content "
                "category. Use this tool when an AI agent needs to discover available "
                "licensed content before pulling it. Ideal for finding relevant pockets when "
                "the agent knows what topic or creator it needs but not the specific pocket "
                "ID. Does NOT retrieve content \u2014 use pull_content with the returned pocket_id "
                "to access actual content. Requires a valid Bearer token. The query parameter "
                "accepts natural language search terms, creator names, or category keywords. "
                "Returns matching pockets with their IDs, titles, descriptions, creators, "
                "content types, and pricing tiers. Use the limit parameter to control page size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search terms, creator names, or category keywords "
                            "to find matching content pockets."
                        ),
                    },
                    "content_type": {
                        "type": "string",
                        "description": (
                            "Optional filter to show only pockets in a specific content category "
                            "or source type, such as 'youtube', 'webpage', 'video', 'text', "
                            "'image', or 'audio'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return per page. Defaults to 10, "
                            "maximum 50 for search results."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_pockets",
            "description": (
                "List all available content pockets in the Meta-Stamp Pockets catalog, "
                "optionally filtered by creator. Use this tool when an AI agent needs to "
                "browse the full catalog or see everything available from a specific creator. "
                "Prefer search_pockets when looking for specific topics. Does NOT retrieve "
                "content \u2014 use pull_content with the returned pocket_id to access actual "
                "content. Requires a valid Bearer token. Optional creator_id parameter filters "
                "results to a single creator's pockets. Returns a paginated list of pockets "
                "with IDs, titles, descriptions, creators, content types, and pricing tiers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "creator_id": {
                        "type": "string",
                        "description": (
                            "Optional filter to show only pockets from a specific creator. Use "
                            "the creator's unique identifier."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return per page. Defaults to 50, "
                            "maximum 200 for catalog browsing."
                        ),
                    },
                },
                "required": [],
            },
        },
    ],
    "terms": (
        "By connecting to this MCP server, you accept the Pockets Content License Terms. "
        "Each content pull is metered and billed to your agent account. Creators are "
        "compensated automatically per pull. See /api/v1/agreements/terms for full terms."
    ),
}


_NEW_DESCRIPTION = (
    "Meta-Stamp Pockets \u2014 Licensed creator content for AI agents. "
    "Tiered per-pull pricing: $0.001 (generic/blogs), $0.01 (premium creator \u2014 YouTube, Substack), "
    "$0.025 (flagship creators), $0.10\u2013$0.25 (verifiable institutional \u2014 news, legal, scientific). "
    "Annual enterprise licensing also available. 85% of revenue goes directly to creators. "
    "Provenance-verified content from authorized creators."
)

_SERVER_INFO = {
    "name": "meta-stamp-pockets",
    "version": "1.0.0",
    "description": _NEW_DESCRIPTION,
    "transport": "http+sse",
    "auth_required": True,
    "status": "healthy",
}

_INIT_RESULT = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {
        "name": "meta-stamp-pockets",
        "version": "1.0.0",
        "description": _NEW_DESCRIPTION,
    },
}


def _tools_list_response() -> dict[str, Any]:
    """Build the MCP spec-compatible tools/list discovery payload."""
    return {
        "tools": [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["parameters"],
            }
            for tool in MCP_TOOL_MANIFEST["tools"]
        ]
    }


async def _sse_init_stream(session_id: str) -> AsyncIterator[str]:
    """Yield a single SSE event with the MCP initialize result."""
    event = {"jsonrpc": "2.0", "result": _INIT_RESULT, "id": 0}
    yield f"data: {json.dumps(event)}\n\n"


@mcp_router.get(
    "",
    summary="MCP Server Info / SSE transport (public)",
    description=(
        "Public GET handler for health checks and discovery. "
        "Returns SSE stream when Accept: text/event-stream is set (Glama), "
        "otherwise returns JSON server info."
    ),
    include_in_schema=False,
)
@mcp_router.get(
    "/",
    summary="MCP Server Info / SSE transport (public, trailing slash)",
    include_in_schema=False,
)
async def mcp_get_info(request: Request):  # type: ignore[return]
    """
    Public GET endpoint for Glama/Smithery health checks.

    - If Accept contains 'text/event-stream': returns SSE with MCP initialize result
      and Mcp-Session-Id header (satisfies Glama health check).
    - Otherwise: returns plain JSON server info.
    """
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        session_id = str(uuid.uuid4())
        return StreamingResponse(
            _sse_init_stream(session_id),
            media_type="text/event-stream",
            headers={
                "Mcp-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    return JSONResponse(content=_SERVER_INFO)


@mcp_router.post(
    "/initialize",
    summary="MCP Initialize (public)",
    description="Public MCP initialization for server discovery. No auth required.",
    include_in_schema=False,
)
async def mcp_initialize() -> JSONResponse:
    """Public initialize endpoint for MCP discovery tools like Smithery."""
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "meta-stamp-pockets",
                "version": "1.0.0",
                "description": _NEW_DESCRIPTION,
            },
        },
        "id": 1,
    })


@mcp_router.get(
    "/tools/list",
    summary="MCP Tools List (public)",
    description="Public MCP tools/list discovery for directory crawlers. No auth required.",
    include_in_schema=False,
)
async def get_mcp_tools_list() -> JSONResponse:
    """Return MCP spec-compatible tool schemas without requiring authentication."""
    return JSONResponse(content=_tools_list_response())


@mcp_router.post(
    "/tools/list",
    summary="MCP Tools List (public)",
    description="Public tools list for Smithery/Glama capability discovery. No auth required.",
    include_in_schema=False,
)
async def mcp_tools_list() -> JSONResponse:
    """Public tools/list endpoint so Smithery can display capabilities."""
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "result": MCP_TOOL_MANIFEST,
        "id": 1,
    })


@mcp_router.get(
    "/manifest",
    summary="MCP Tool Manifest",
    description="Returns the list of available MCP tools and their schemas.",
)
async def get_manifest() -> JSONResponse:
    """Return the MCP tool manifest for agent discovery."""
    return JSONResponse(content=MCP_TOOL_MANIFEST)


@mcp_router.get(
    "/capabilities",
    summary="MCP Capabilities (public)",
    description=(
        "Public canonical discovery endpoint. Returns full tool schemas without auth. "
        "Used by mcp.so, Glama, Smithery, and any directory that wants to list our tools."
    ),
)
async def get_capabilities() -> JSONResponse:
    """
    Public capabilities endpoint — full tool schemas, no auth required.

    This is the canonical endpoint for any MCP directory that wants to
    enumerate available tools and their input schemas.
    """
    return JSONResponse(content={
        "server": _SERVER_INFO,
        "protocol_version": "2024-11-05",
        "tools": MCP_TOOL_MANIFEST["tools"],
        "terms_url": "/api/v1/agreements/terms",
        "pricing": {"model": "tiered-per-pull", "tiers": [{"type": "generic", "amount": 0.001}, {"type": "premium_creator", "amount": 0.01}, {"type": "flagship", "amount": 0.025}, {"type": "institutional", "amount": 0.10}], "currency": "USD", "enterprise": "annual license available"},
    })


@mcp_router.post(
    "",
    summary="MCP JSON-RPC 2.0 Endpoint (no trailing slash)",
    description="Handle MCP tool calls via JSON-RPC 2.0 protocol — no trailing slash variant.",
    include_in_schema=False,
)
@mcp_router.post(
    "/",
    summary="MCP JSON-RPC 2.0 Endpoint",
    description="Handle MCP tool calls via JSON-RPC 2.0 protocol.",
)
async def handle_jsonrpc(
    request: Request,
    agent: AgentAPIKey = Depends(get_current_agent),
    _agreement: Any = Depends(check_agreement),
    _rate_limit: Any = Depends(check_rate_limit),
) -> StreamingResponse:
    """
    Main MCP endpoint — processes JSON-RPC 2.0 requests from AI agents.

    Per MCP Streamable HTTP spec (2025-03-26), ALL responses from this POST
    handler are returned as Content-Type: text/event-stream with the body:
        data: {json-rpc-response}\n\n

    The middleware chain (get_current_agent → check_agreement → check_rate_limit)
    runs before this handler, ensuring the agent is authenticated, has accepted
    terms, and is within rate limits.
    """
    start_time = time.monotonic()
    session_id = str(uuid.uuid4())

    # Parse JSON-RPC request
    try:
        body = await request.json()
    except Exception:
        return _sse_response(
            _jsonrpc_error(PARSE_ERROR, "Parse error: invalid JSON"),
            session_id,
        )

    # Validate JSON-RPC structure
    if not isinstance(body, dict):
        return _sse_response(
            _jsonrpc_error(INVALID_REQUEST, "Invalid request: expected JSON object"),
            session_id,
        )

    jsonrpc_version = body.get("jsonrpc")
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    if jsonrpc_version != "2.0":
        return _sse_response(
            _jsonrpc_error(
                INVALID_REQUEST,
                "Invalid request: jsonrpc must be '2.0'",
                request_id,
            ),
            session_id,
        )

    if not isinstance(method, str) or not method:
        return _sse_response(
            _jsonrpc_error(
                INVALID_REQUEST,
                "Invalid request: method must be a non-empty string",
                request_id,
            ),
            session_id,
        )

    if not isinstance(params, dict):
        return _sse_response(
            _jsonrpc_error(INVALID_PARAMS, "Invalid params: expected JSON object", request_id),
            session_id,
        )

    # Handle public methods without auth
    if method == "initialize":
        return _sse_response(
            {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "meta-stamp-pockets",
                        "version": "1.0.0",
                        "description": _NEW_DESCRIPTION,
                    },
                },
                "id": request_id,
            },
            session_id,
        )

    if method == "tools/list":
        return _sse_response(
            {
                "jsonrpc": "2.0",
                "result": MCP_TOOL_MANIFEST,
                "id": request_id,
            },
            session_id,
        )

    # All other methods require auth
    if agent is None:
        return _sse_response(
            _jsonrpc_error(401, "Authentication required", request_id),
            session_id,
            status_code=401,
        )

    # Route to tool handler
    tools = MCPTools()

    try:
        if method == "pull_content":
            result = await tools.pull_content(
                agent=agent,
                pocket_id=params.get("pocket_id", ""),
                query=params.get("query"),
            )
        elif method == "search_pockets":
            result = await tools.search_pockets(
                query=params.get("query", ""),
                content_type=params.get("content_type"),
                limit=params.get("limit", 10),
            )
        elif method == "list_pockets":
            result = await tools.list_pockets(
                creator_id=params.get("creator_id"),
                limit=params.get("limit", 50),
            )
        else:
            return _sse_response(
                _jsonrpc_error(METHOD_NOT_FOUND, f"Method not found: {method}", request_id),
                session_id,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "MCP %s completed in %.2fms for agent %s",
            method,
            elapsed_ms,
            agent.key_prefix,
        )

        return _sse_response(_jsonrpc_success(result, request_id), session_id)

    except ValueError as exc:
        return _sse_response(
            _jsonrpc_error(INVALID_PARAMS, str(exc), request_id),
            session_id,
        )
    except Exception:
        logger.exception("MCP internal error for method %s", method)
        return _sse_response(
            _jsonrpc_error(INTERNAL_ERROR, "Internal server error", request_id),
            session_id,
        )
