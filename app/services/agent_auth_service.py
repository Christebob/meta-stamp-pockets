"""
Agent authentication service for META-STAMP V3 Pockets MCP.

Manages API key creation, validation, and lifecycle for AI agent providers.
Agent keys use SHA-256 hashing — the raw key is only shown once at creation.
"""

import logging
import secrets

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from bson import ObjectId

from app.core.database import get_db_client
from app.core.redis_client import get_redis_client
from app.models.agent import AgentAPIKey, AgentProvider


logger = logging.getLogger(__name__)

AGENT_SESSION_TTL = 300  # 5 minutes cache for validated agent sessions
AGENT_SESSION_PREFIX = "agent_session"


class AgentAuthError(Exception):
    """Base exception for agent authentication failures."""


class AgentKeyNotFoundError(AgentAuthError):
    """Raised when an agent API key is not found or inactive."""


class AgentAuthService:
    """Service for managing agent API keys and authentication."""

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (raw_key, key_hash, key_prefix).
            The raw_key is only returned once — store it securely.
        """
        raw_key = f"pkt_{secrets.token_urlsafe(32)}"
        key_hash = sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]
        return raw_key, key_hash, key_prefix

    async def create_agent_key(
        self,
        provider: AgentProvider,
        provider_name: str,
        rate_limit_per_minute: int = 100,
    ) -> tuple[str, AgentAPIKey]:
        """
        Create a new agent API key.

        Returns:
            Tuple of (raw_key, AgentAPIKey model). The raw_key is only
            available at creation time.
        """
        raw_key, key_hash, key_prefix = self.generate_api_key()

        db_client = get_db_client()
        collection = db_client.get_agent_keys_collection()

        now = datetime.now(UTC)
        doc: dict[str, Any] = {
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "provider": provider.value,
            "provider_name": provider_name,
            "rate_limit_per_minute": rate_limit_per_minute,
            "is_active": True,
            "created_at": now,
            "last_used_at": None,
            "metadata": {},
        }

        result = await collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)

        agent_key = AgentAPIKey.model_validate(doc)
        logger.info(
            "Created agent API key for %s (%s), prefix: %s",
            provider_name,
            provider.value,
            key_prefix,
        )
        return raw_key, agent_key

    async def validate_api_key(self, raw_key: str) -> AgentAPIKey:
        """
        Validate an API key and return the associated AgentAPIKey.

        First checks Redis cache, then falls back to MongoDB.
        Updates last_used_at timestamp.

        Raises:
            AgentKeyNotFoundError: If key is invalid or inactive.
        """
        key_hash = sha256(raw_key.encode()).hexdigest()

        # Check Redis cache first
        redis_client = get_redis_client()
        cache_key = f"{AGENT_SESSION_PREFIX}:{key_hash}"
        cached = await redis_client.get_json(cache_key)
        if cached is not None:
            return AgentAPIKey.model_validate(cached)

        # Fall back to MongoDB
        db_client = get_db_client()
        collection = db_client.get_agent_keys_collection()

        doc = await collection.find_one({"key_hash": key_hash, "is_active": True})
        if doc is None:
            raise AgentKeyNotFoundError("Invalid or inactive API key")

        await collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"last_used_at": datetime.now(UTC)}},
        )

        doc["_id"] = str(doc["_id"])
        agent_key = AgentAPIKey.model_validate(doc)

        # Cache in Redis
        await redis_client.set_json(
            cache_key,
            agent_key.model_dump(mode="json"),
            ttl=AGENT_SESSION_TTL,
        )

        return agent_key

    async def deactivate_key(self, key_id: str) -> bool:
        """Deactivate an agent API key."""
        if not ObjectId.is_valid(key_id):
            return False

        db_client = get_db_client()
        collection = db_client.get_agent_keys_collection()

        result = await collection.update_one(
            {"_id": ObjectId(key_id)},
            {"$set": {"is_active": False}},
        )
        return result.modified_count > 0

    async def list_agent_keys(self, provider: str | None = None, limit: int = 50) -> list[AgentAPIKey]:
        """List agent API keys, optionally filtered by provider."""
        db_client = get_db_client()
        collection = db_client.get_agent_keys_collection()

        query: dict[str, Any] = {}
        if provider:
            query["provider"] = provider

        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)

        result = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            result.append(AgentAPIKey.model_validate(doc))
        return result
