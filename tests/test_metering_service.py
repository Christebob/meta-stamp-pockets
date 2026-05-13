"""
Tests for MeteringService — pull logging and compensation tracking.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.metering_service import MeteringService


class TestMeteringService:
    """Tests for the metering service."""

    @pytest.mark.asyncio
    async def test_log_pull_does_not_raise(self) -> None:
        """Metering should never raise — it's fire-and-forget."""
        with patch("app.services.metering_service.get_db_client") as mock_db:
            mock_client = MagicMock()
            mock_collection = AsyncMock()
            mock_collection.insert_one = AsyncMock()
            mock_collection.update_one = AsyncMock()
            mock_client.get_pull_logs_collection.return_value = mock_collection
            mock_client.get_pockets_collection.return_value = mock_collection
            mock_client.get_wallet_collection.return_value = mock_collection
            mock_db.return_value = mock_client

            service = MeteringService()
            # Should not raise
            await service.log_pull(
                pocket_id="pocket-1",
                agent_key_id="agent-1",
                creator_id="creator-1",
                provider="openai",
                compensation_amount=0.01,
                response_time_ms=5.0,
            )
