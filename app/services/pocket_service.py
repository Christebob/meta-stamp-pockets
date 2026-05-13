"""
Pockets business logic service for META-STAMP V3.

This service manages creator pocket creation from URL content, snapshot storage,
listing pockets, and simulated pull events with compensation increments.
"""

import logging

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo import ReturnDocument

from app.core.database import get_db_client
from app.models.pocket import Pocket, PocketStatus
from app.services.url_processor_service import URLProcessorService


DEFAULT_COMPENSATION_PER_PULL = 0.01
MAX_SNAPSHOT_LENGTH = 100_000
MAX_LIST_LIMIT = 200

logger = logging.getLogger(__name__)


class PocketServiceError(Exception):
    """Base exception for pocket service failures."""


class PocketValidationError(PocketServiceError):
    """Raised when pocket inputs are invalid."""


class PocketNotFoundError(PocketServiceError):
    """Raised when a pocket cannot be found for the creator."""


class PocketStateError(PocketServiceError):
    """Raised when an operation is invalid for the current pocket status."""


class PocketService:
    """Service for managing pocket lifecycle and pull compensation behavior."""

    def __init__(
        self,
        url_processor: URLProcessorService | None = None,
        compensation_per_pull: float = DEFAULT_COMPENSATION_PER_PULL,
    ) -> None:
        """
        Initialize PocketService dependencies.

        Args:
            url_processor: Optional URLProcessorService instance.
            compensation_per_pull: USD amount added for each pull simulation.
        """
        if compensation_per_pull < 0:
            raise PocketValidationError("compensation_per_pull cannot be negative")

        self.url_processor = url_processor or URLProcessorService()
        self.compensation_per_pull = round(compensation_per_pull, 6)

    async def create_pocket(self, creator_id: str, content_url: str) -> Pocket:
        """
        Create a pocket by indexing URL content into a stored text snapshot.

        The pocket is created with `indexing` status, then updated to `active`
        on successful content extraction or `failed` if processing fails.
        """
        normalized_creator_id = self._normalize_creator_id(creator_id)
        normalized_content_url = self._normalize_content_url(content_url)

        db_client = get_db_client()
        pockets_collection = db_client.get_pockets_collection()

        now = datetime.now(UTC)
        base_doc: dict[str, Any] = {
            "creator_id": normalized_creator_id,
            "content_url": normalized_content_url,
            "content_type": "unknown",
            "status": PocketStatus.INDEXING.value,
            "pull_count": 0,
            "compensation_earned": 0.0,
            "snapshot_text": None,
            "error_message": None,
            "source_metadata": {},
            "created_at": now,
            "updated_at": now,
        }

        insert_result = await pockets_collection.insert_one(base_doc)
        pocket_object_id = insert_result.inserted_id

        try:
            processed_result = await self.url_processor.process_url(normalized_content_url)
        except Exception as exc:
            logger.exception("Unexpected error processing pocket URL")
            processed_result = {
                "success": False,
                "platform": "unknown",
                "error": f"Unexpected URL processing error: {exc!s}",
            }

        snapshot_text = self._extract_snapshot_text(processed_result)
        content_type = self._extract_content_type(processed_result)
        source_metadata = self._extract_source_metadata(processed_result)

        is_success = bool(processed_result.get("success")) and bool(snapshot_text)

        update_doc: dict[str, Any] = {
            "content_type": content_type,
            "snapshot_text": snapshot_text if snapshot_text else None,
            "source_metadata": source_metadata,
            "updated_at": datetime.now(UTC),
        }

        if is_success:
            update_doc["status"] = PocketStatus.ACTIVE.value
            update_doc["error_message"] = None
        else:
            update_doc["status"] = PocketStatus.FAILED.value
            update_doc["error_message"] = (
                str(processed_result.get("error"))
                if processed_result.get("error")
                else "Failed to extract indexable snapshot content"
            )

        updated_doc = await pockets_collection.find_one_and_update(
            {"_id": pocket_object_id},
            {"$set": update_doc},
            return_document=ReturnDocument.AFTER,
        )

        if updated_doc is None:
            raise RuntimeError("Failed to retrieve pocket after creation")

        return self._to_pocket_model(updated_doc)

    async def list_pockets(self, creator_id: str, limit: int = 50) -> list[Pocket]:
        """List pockets owned by the creator ordered by newest first."""
        normalized_creator_id = self._normalize_creator_id(creator_id)
        if limit < 1 or limit > MAX_LIST_LIMIT:
            raise PocketValidationError(f"limit must be between 1 and {MAX_LIST_LIMIT}")

        db_client = get_db_client()
        pockets_collection = db_client.get_pockets_collection()

        cursor = (
            pockets_collection.find({"creator_id": normalized_creator_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        documents = await cursor.to_list(length=limit)
        return [self._to_pocket_model(document) for document in documents]

    async def get_pocket_by_id(self, pocket_id: str) -> Pocket:
        """
        Retrieve any active pocket by ID regardless of owner.

        Used by the paywall pull endpoint so any authenticated user can
        discover and pay for content.
        """
        normalized_pocket_id = pocket_id.strip()
        if not ObjectId.is_valid(normalized_pocket_id):
            raise PocketValidationError("Invalid pocket ID format")

        db_client = get_db_client()
        pockets_collection = db_client.get_pockets_collection()
        doc = await pockets_collection.find_one({"_id": ObjectId(normalized_pocket_id)})

        if doc is None:
            raise PocketNotFoundError("Pocket not found")

        return self._to_pocket_model(doc)

    async def pull_pocket(self, creator_id: str, pocket_id: str) -> Pocket:
        """
        Simulate an AI agent pull for an active pocket.

        Increments both pull_count and compensation_earned atomically.
        """
        normalized_creator_id = self._normalize_creator_id(creator_id)
        normalized_pocket_id = pocket_id.strip()

        if not ObjectId.is_valid(normalized_pocket_id):
            raise PocketValidationError("Invalid pocket ID format")

        db_client = get_db_client()
        pockets_collection = db_client.get_pockets_collection()
        pocket_object_id = ObjectId(normalized_pocket_id)

        existing_doc = await pockets_collection.find_one(
            {"_id": pocket_object_id, "creator_id": normalized_creator_id}
        )
        if existing_doc is None:
            raise PocketNotFoundError("Pocket not found")

        existing_status = existing_doc.get("status")
        if existing_status != PocketStatus.ACTIVE.value:
            raise PocketStateError("Only active pockets can be pulled")

        updated_doc = await pockets_collection.find_one_and_update(
            {
                "_id": pocket_object_id,
                "creator_id": normalized_creator_id,
                "status": PocketStatus.ACTIVE.value,
            },
            {
                "$inc": {
                    "pull_count": 1,
                    "compensation_earned": self.compensation_per_pull,
                },
                "$set": {"updated_at": datetime.now(UTC)},
            },
            return_document=ReturnDocument.AFTER,
        )

        if updated_doc is None:
            raise PocketStateError("Pocket status changed before pull could be applied")

        return self._to_pocket_model(updated_doc)

    def _normalize_creator_id(self, creator_id: str) -> str:
        """Normalize and validate creator ID."""
        normalized = creator_id.strip()
        if not normalized:
            raise PocketValidationError("creator_id is required")
        return normalized

    def _normalize_content_url(self, content_url: str) -> str:
        """Normalize and validate source content URL."""
        normalized = content_url.strip()
        if not normalized:
            raise PocketValidationError("content_url is required")
        if not normalized.startswith(("http://", "https://")):
            raise PocketValidationError("content_url must start with http:// or https://")
        return normalized

    def _extract_content_type(self, processed_result: dict[str, Any]) -> str:
        """Extract normalized content type from URL processing output."""
        platform = processed_result.get("platform")
        if isinstance(platform, str) and platform.strip():
            return platform.strip().lower()
        return "unknown"

    def _extract_snapshot_text(self, processed_result: dict[str, Any]) -> str:
        """Build a single snapshot text blob from platform-specific fields."""
        platform = self._extract_content_type(processed_result)
        parts: list[str] = []

        if platform == "youtube":
            metadata = processed_result.get("metadata")
            if isinstance(metadata, dict):
                parts.extend(
                    self._extract_text_values(metadata, ["title", "description", "channel"])
                )
            parts.extend(self._extract_text_values(processed_result, ["transcript"]))

        elif platform == "vimeo":
            metadata = processed_result.get("metadata")
            if isinstance(metadata, dict):
                parts.extend(
                    self._extract_text_values(metadata, ["title", "description", "author"])
                )

        else:
            parts.extend(
                self._extract_text_values(processed_result, ["title", "description", "content"])
            )

        if not parts:
            parts.extend(self._extract_text_values(processed_result, ["transcript", "content"]))

        snapshot = "\n\n".join(part for part in parts if part)
        cleaned = snapshot.strip()
        if len(cleaned) > MAX_SNAPSHOT_LENGTH:
            return cleaned[:MAX_SNAPSHOT_LENGTH]
        return cleaned

    def _extract_source_metadata(self, processed_result: dict[str, Any]) -> dict[str, Any]:
        """Extract small metadata payload for pocket debugging and transparency."""
        metadata: dict[str, Any] = {}

        for key in ["platform", "processed_at", "title", "description", "word_count", "video_id"]:
            value = processed_result.get(key)
            if value is not None:
                metadata[key] = value

        source_metadata = processed_result.get("metadata")
        if isinstance(source_metadata, dict) and source_metadata:
            metadata["metadata"] = source_metadata

        return metadata

    def _extract_text_values(self, source: dict[str, Any], keys: list[str]) -> list[str]:
        """Collect non-empty string values from a dictionary by key list."""
        extracted: list[str] = []
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    extracted.append(normalized)
        return extracted

    def _to_pocket_model(self, document: dict[str, Any]) -> Pocket:
        """Convert a MongoDB document into a validated Pocket model."""
        serialized = dict(document)
        if "_id" in serialized:
            serialized["_id"] = str(serialized["_id"])
        return Pocket.model_validate(serialized)


__all__ = [
    "DEFAULT_COMPENSATION_PER_PULL",
    "PocketNotFoundError",
    "PocketService",
    "PocketServiceError",
    "PocketStateError",
    "PocketValidationError",
]
