"""
MCP tool implementations — the business logic behind each MCP method.

Tools:
- pull_content: Pull pre-indexed content from a Pocket (<30ms target)
- search_pockets: Search for Pockets by keyword
- list_pockets: List available Pockets
"""

import asyncio
import logging
import time

from typing import Any

from bson import ObjectId

from app.config import get_settings
from app.core.database import get_db_client
from app.core.redis_client import get_redis_client
from app.models.agent import AgentAPIKey
from app.models.pocket import PocketStatus
from app.services.metering_service import MeteringService


logger = logging.getLogger(__name__)

POCKET_CACHE_PREFIX = "pocket_snapshot"


class MCPTools:
    """MCP tool implementations for AI agent content access."""

    def __init__(self) -> None:
        self.metering = MeteringService()
        self.settings = get_settings()

    async def pull_content(
        self,
        agent: AgentAPIKey,
        pocket_id: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        """
        Pull pre-indexed content from a Pocket.

        This is the critical hot path — target <30ms server-side.

        Flow:
        1. Check Redis cache for pocket snapshot (~1ms)
        2. If cache miss, read from MongoDB and warm cache (~5-10ms)
        3. Log pull and credit creator async (fire-and-forget, ~0ms blocking)
        4. Return structured content

        Args:
            agent: Authenticated agent making the request.
            pocket_id: ID of the Pocket to pull.
            query: Optional query to filter content.

        Returns:
            Structured content dict with snapshot, metadata, and license info.

        Raises:
            ValueError: If pocket_id is invalid or pocket not found/not active.
        """
        start_time = time.monotonic()

        if not pocket_id or not pocket_id.strip():
            raise ValueError("pocket_id is required")

        if query is not None and not isinstance(query, str):
            raise ValueError("query must be a string when provided")

        pocket_id = pocket_id.strip()

        # 1. Try Redis cache first (fast path)
        redis_client = get_redis_client()
        cache_key = f"{POCKET_CACHE_PREFIX}:{pocket_id}"
        cached_snapshot = await redis_client.get_json(cache_key)

        if cached_snapshot is not None:
            # Cache hit — fastest path
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Fire-and-forget metering
            asyncio.create_task(
                self.metering.log_pull(
                    pocket_id=pocket_id,
                    agent_key_id=agent.id or "",
                    creator_id=cached_snapshot.get("creator_id", ""),
                    provider=agent.provider.value if hasattr(agent.provider, "value") else str(agent.provider),
                    compensation_amount=cached_snapshot.get(
                        "price_per_pull",
                        self.settings.mcp_default_price_per_pull,
                    ),
                    response_time_ms=elapsed_ms,
                )
            )

            return {
                "pocket_id": pocket_id,
                "content": cached_snapshot.get("snapshot_text", ""),
                "content_type": cached_snapshot.get("content_type", "unknown"),
                "metadata": cached_snapshot.get("source_metadata", {}),
                "license": {
                    "terms_version": self.settings.mcp_terms_version,
                    "pull_price_usd": cached_snapshot.get(
                        "price_per_pull",
                        self.settings.mcp_default_price_per_pull,
                    ),
                    "licensed": True,
                },
                "response_time_ms": round(elapsed_ms, 2),
            }

        # 2. Cache miss — read from MongoDB
        if not ObjectId.is_valid(pocket_id):
            raise ValueError(f"Invalid pocket_id format: {pocket_id}")

        db_client = get_db_client()
        pockets = db_client.get_pockets_collection()
        doc = await pockets.find_one({"_id": ObjectId(pocket_id)})

        if doc is None:
            raise ValueError(f"Pocket not found: {pocket_id}")

        if doc.get("status") != PocketStatus.ACTIVE.value:
            raise ValueError(f"Pocket is not active (status: {doc.get('status')})")

        if not doc.get("snapshot_text"):
            raise ValueError("Pocket has no indexed content")

        # Warm the cache for next time
        snapshot_data = {
            "pocket_id": pocket_id,
            "creator_id": doc.get("creator_id", ""),
            "snapshot_text": doc.get("snapshot_text", ""),
            "content_type": doc.get("content_type", "unknown"),
            "source_metadata": doc.get("source_metadata", {}),
            "price_per_pull": doc.get("price_per_pull", self.settings.mcp_default_price_per_pull),
        }
        await redis_client.set_json(
            cache_key,
            snapshot_data,
            ttl=self.settings.mcp_pocket_cache_ttl_seconds,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Fire-and-forget metering
        asyncio.create_task(
            self.metering.log_pull(
                pocket_id=pocket_id,
                agent_key_id=agent.id or "",
                creator_id=doc.get("creator_id", ""),
                provider=agent.provider.value if hasattr(agent.provider, "value") else str(agent.provider),
                compensation_amount=doc.get("price_per_pull", self.settings.mcp_default_price_per_pull),
                response_time_ms=elapsed_ms,
            )
        )

        return {
            "pocket_id": pocket_id,
            "content": doc.get("snapshot_text", ""),
            "content_type": doc.get("content_type", "unknown"),
            "metadata": doc.get("source_metadata", {}),
            "license": {
                "terms_version": self.settings.mcp_terms_version,
                "pull_price_usd": doc.get("price_per_pull", self.settings.mcp_default_price_per_pull),
                "licensed": True,
            },
            "response_time_ms": round(elapsed_ms, 2),
        }

    async def search_pockets(
        self,
        query: str,
        content_type: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search for Pockets by keyword in title, description, or snapshot text.

        Args:
            query: Search query string.
            content_type: Optional content type filter.
            limit: Max results (default 10, max 50).

        Returns:
            Dict with list of matching pocket metadata.
        """
        if not query or not query.strip():
            raise ValueError("query is required")

        limit = min(max(limit, 1), 50)

        db_client = get_db_client()
        pockets = db_client.get_pockets_collection()

        # Build MongoDB text search query
        search_filter: dict[str, Any] = {
            "status": PocketStatus.ACTIVE.value,
            "$or": [
                {"snapshot_text": {"$regex": query.strip(), "$options": "i"}},
                {"content_url": {"$regex": query.strip(), "$options": "i"}},
                {"content_type": {"$regex": query.strip(), "$options": "i"}},
            ],
        }

        if content_type:
            search_filter["content_type"] = content_type.strip().lower()

        cursor = (
            pockets.find(
                search_filter,
                {
                    "_id": 1,
                    "creator_id": 1,
                    "content_url": 1,
                    "content_type": 1,
                    "pull_count": 1,
                    "created_at": 1,
                    "source_metadata": 1,
                },
            )
            .sort("pull_count", -1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)

        results = []
        for doc in docs:
            results.append(
                {
                    "pocket_id": str(doc["_id"]),
                    "creator_id": doc.get("creator_id", ""),
                    "content_url": doc.get("content_url", ""),
                    "content_type": doc.get("content_type", "unknown"),
                    "pull_count": doc.get("pull_count", 0),
                    "title": doc.get("source_metadata", {}).get("title", ""),
                    "description": doc.get("source_metadata", {}).get("description", ""),
                }
            )

        return {
            "query": query.strip(),
            "count": len(results),
            "results": results,
        }

    async def list_pockets(
        self,
        creator_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        List available Pockets, optionally filtered by creator.

        Args:
            creator_id: Optional creator ID filter.
            limit: Max results (default 50, max 200).

        Returns:
            Dict with list of pocket metadata.
        """
        limit = min(max(limit, 1), 200)

        db_client = get_db_client()
        pockets = db_client.get_pockets_collection()

        query_filter: dict[str, Any] = {"status": PocketStatus.ACTIVE.value}
        if creator_id:
            query_filter["creator_id"] = creator_id.strip()

        cursor = (
            pockets.find(
                query_filter,
                {
                    "_id": 1,
                    "creator_id": 1,
                    "content_url": 1,
                    "content_type": 1,
                    "pull_count": 1,
                    "compensation_earned": 1,
                    "created_at": 1,
                    "source_metadata": 1,
                },
            )
            .sort("created_at", -1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)

        results = []
        for doc in docs:
            results.append(
                {
                    "pocket_id": str(doc["_id"]),
                    "creator_id": doc.get("creator_id", ""),
                    "content_url": doc.get("content_url", ""),
                    "content_type": doc.get("content_type", "unknown"),
                    "pull_count": doc.get("pull_count", 0),
                    "title": doc.get("source_metadata", {}).get("title", ""),
                    "description": doc.get("source_metadata", {}).get("description", ""),
                }
            )

        return {
            "count": len(results),
            "results": results,
        }
