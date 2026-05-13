"""
AI Crawler Detection and Redirection Middleware

This middleware detects AI crawler user agents (GPTBot, ClaudeBot, Google-Extended, PerplexityBot)
and returns 402 Payment Required responses with the X-MCP-License-Endpoint header, directing
them to the MCP licensing endpoint. It also logs all crawler activity to MongoDB.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.database import get_db_client


logger = logging.getLogger(__name__)

# AI crawler user agents to detect
AI_CRAWLER_USER_AGENTS = [
    "GPTBot",
    "ClaudeBot",
    "Google-Extended",
    "PerplexityBot",
]

# MCP license endpoint URL
MCP_LICENSE_ENDPOINT = "https://metastampv3-production.up.railway.app/mcp"

# Collection name for crawler activity logs
CRAWLER_ACTIVITY_COLLECTION = "crawler_activity"


class AICrawlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detecting AI crawlers and returning 402 Payment Required responses.

    This middleware:
    1. Detects AI crawler user agents (GPTBot, ClaudeBot, Google-Extended, PerplexityBot)
    2. Returns 402 Payment Required with X-MCP-License-Endpoint header
    3. Logs all crawler hits to MongoDB crawler_activity collection

    The 402 response informs AI crawlers that content requires licensing via MCP protocol.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and detect AI crawlers.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            Response: 402 Payment Required for AI crawlers, or normal response
        """
        # Get user agent from request headers
        user_agent = request.headers.get("user-agent", "")

        # Check if the user agent is an AI crawler
        is_ai_crawler = any(crawler in user_agent for crawler in AI_CRAWLER_USER_AGENTS)

        if is_ai_crawler:
            # Log crawler activity to MongoDB
            await self._log_crawler_activity(request, user_agent)

            # Return 402 Payment Required with MCP license endpoint header
            return JSONResponse(
                status_code=402,
                content={
                    "error": "Payment Required",
                    "message": (
                        "This content requires licensing. "
                        "Please connect via MCP protocol using the endpoint "
                        "provided in the X-MCP-License-Endpoint header."
                    ),
                    "mcp_endpoint": MCP_LICENSE_ENDPOINT,
                },
                headers={
                    "X-MCP-License-Endpoint": MCP_LICENSE_ENDPOINT,
                },
            )

        # Not an AI crawler, proceed with normal request handling
        return await call_next(request)

    async def _log_crawler_activity(self, request: Request, user_agent: str) -> None:
        """
        Log AI crawler activity to MongoDB.

        Args:
            request: The incoming HTTP request
            user_agent: The User-Agent header value
        """
        try:
            db_client = get_db_client()
            crawler_collection = db_client.get_database()[CRAWLER_ACTIVITY_COLLECTION]

            # Extract request information
            client_ip = request.client.host if request.client else None
            path = str(request.url.path)
            method = request.method
            query_params = dict(request.query_params)

            # Determine which crawler
            detected_crawler = next(
                (crawler for crawler in AI_CRAWLER_USER_AGENTS if crawler in user_agent),
                "Unknown",
            )

            # Create log entry
            log_entry: dict[str, Any] = {
                "timestamp": datetime.now(UTC),
                "crawler_type": detected_crawler,
                "user_agent": user_agent,
                "ip_address": client_ip,
                "method": method,
                "path": path,
                "query_params": query_params,
                "response_status": 402,
                "mcp_endpoint": MCP_LICENSE_ENDPOINT,
            }

            # Insert log entry into MongoDB
            await crawler_collection.insert_one(log_entry)

            logger.info(
                f"AI crawler detected: {detected_crawler} | "
                f"IP: {client_ip} | "
                f"Path: {path} | "
                f"Returned 402 with MCP endpoint"
            )

        except Exception:
            # Don't block the request if logging fails
            logger.exception("Failed to log crawler activity to MongoDB")
