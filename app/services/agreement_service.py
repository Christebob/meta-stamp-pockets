"""
Agreement service for META-STAMP V3 Pockets "iTunes agreement" model.

Connection = contract acceptance. When an AI agent first connects to the
MCP server, an Agreement record is automatically created. This is the
legal record of terms acceptance.
"""

import logging

from datetime import UTC, datetime
from typing import Any

from app.core.database import get_db_client
from app.models.agreement import CURRENT_TERMS_VERSION, Agreement


logger = logging.getLogger(__name__)


class AgreementService:
    """Service for managing agent agreement records."""

    async def ensure_agreement(
        self,
        agent_key_id: str,
        provider: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> Agreement:
        """
        Ensure an agreement exists for the agent and current terms version.

        If no agreement exists for the current terms version, one is created
        automatically. This implements the "connection = acceptance" model.

        Returns:
            Agreement: The existing or newly created agreement.
        """
        db_client = get_db_client()
        collection = db_client.get_agreements_collection()

        # Check for existing agreement with current terms version
        existing = await collection.find_one(
            {
                "agent_key_id": agent_key_id,
                "terms_version": CURRENT_TERMS_VERSION,
            }
        )

        if existing is not None:
            existing["_id"] = str(existing["_id"])
            return Agreement.model_validate(existing)

        # Create new agreement — connection = acceptance
        now = datetime.now(UTC)
        doc: dict[str, Any] = {
            "agent_key_id": agent_key_id,
            "provider": provider,
            "terms_version": CURRENT_TERMS_VERSION,
            "accepted_at": now,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "metadata": {},
        }

        result = await collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)

        logger.info(
            "Agreement created for agent %s (provider: %s, terms: %s)",
            agent_key_id,
            provider,
            CURRENT_TERMS_VERSION,
        )
        return Agreement.model_validate(doc)

    async def get_agreement(self, agent_key_id: str) -> Agreement | None:
        """Get the current agreement for an agent, if any."""
        db_client = get_db_client()
        collection = db_client.get_agreements_collection()

        doc = await collection.find_one(
            {
                "agent_key_id": agent_key_id,
                "terms_version": CURRENT_TERMS_VERSION,
            }
        )

        if doc is None:
            return None

        doc["_id"] = str(doc["_id"])
        return Agreement.model_validate(doc)

    async def list_agreements(self, provider: str | None = None, limit: int = 50) -> list[Agreement]:
        """List agreements, optionally filtered by provider."""
        db_client = get_db_client()
        collection = db_client.get_agreements_collection()

        query: dict[str, Any] = {}
        if provider:
            query["provider"] = provider

        cursor = collection.find(query).sort("accepted_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)

        result = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            result.append(Agreement.model_validate(doc))
        return result
