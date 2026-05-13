"""
META-STAMP V3 Pockets API Smoke Tests.

Covers basic endpoint wiring for:
- POST /api/v1/pockets/
- GET /api/v1/pockets/
- POST /api/v1/pockets/{pocket_id}/pull
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fastapi.testclient import TestClient

from app.api.v1.pockets import get_pocket_service
from app.core.auth import get_current_user
from app.main import app
from app.models.pocket import Pocket, PocketStatus


class _MockPocketService:
    """Small async mock service used for endpoint smoke testing."""

    compensation_per_pull = 0.01

    async def create_pocket(self, creator_id: str, content_url: str) -> Pocket:
        return Pocket(
            _id="pocket-1",
            creator_id=creator_id,
            content_url=content_url,
            content_type="youtube",
            status=PocketStatus.ACTIVE,
            pull_count=0,
            compensation_earned=0.0,
            price_per_pull=0.01,
            snapshot_text="Sample indexed snapshot",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    async def list_pockets(self, creator_id: str, limit: int = 50) -> list[Pocket]:
        return [
            Pocket(
                _id="pocket-1",
                creator_id=creator_id,
                content_url="https://youtube.com/watch?v=abc123",
                content_type="youtube",
                status=PocketStatus.ACTIVE,
                pull_count=2,
                compensation_earned=0.02,
                price_per_pull=0.01,
                snapshot_text="Snapshot one",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            ),
            Pocket(
                _id="pocket-2",
                creator_id=creator_id,
                content_url="https://example.com/post",
                content_type="webpage",
                status=PocketStatus.INDEXING,
                pull_count=0,
                compensation_earned=0.0,
                price_per_pull=0.01,
                snapshot_text=None,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            ),
        ][:limit]

    async def get_pocket_by_id(self, pocket_id: str) -> Pocket:
        return Pocket(
            _id=pocket_id,
            creator_id="creator-test-id",
            content_url="https://youtube.com/watch?v=abc123",
            content_type="youtube",
            status=PocketStatus.ACTIVE,
            pull_count=2,
            compensation_earned=0.02,
            price_per_pull=0.01,
            snapshot_text="Pulled snapshot content",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    async def pull_pocket(self, creator_id: str, pocket_id: str) -> Pocket:
        return Pocket(
            _id=pocket_id,
            creator_id=creator_id,
            content_url="https://youtube.com/watch?v=abc123",
            content_type="youtube",
            status=PocketStatus.ACTIVE,
            pull_count=3,
            compensation_earned=0.03,
            price_per_pull=0.01,
            snapshot_text="Pulled snapshot content",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


def _mock_current_user() -> dict[str, str]:
    """Mock authenticated creator payload."""
    return {"_id": "creator-test-id"}


def _mock_pocket_service() -> _MockPocketService:
    """Mock pocket service factory for dependency override."""
    return _MockPocketService()


@pytest.fixture
def pockets_client() -> TestClient:
    """
    FastAPI TestClient with auth and pocket service dependencies overridden.
    """
    app.dependency_overrides[get_current_user] = _mock_current_user
    app.dependency_overrides[get_pocket_service] = _mock_pocket_service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_create_pocket_smoke(pockets_client: TestClient) -> None:
    """
    Smoke test for POST /api/v1/pockets/ endpoint.
    """
    response = pockets_client.post(
        "/api/v1/pockets/",
        json={"content_url": "https://youtube.com/watch?v=abc123"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == "pocket-1"
    assert payload["creator_id"] == "creator-test-id"
    assert payload["status"] == "active"
    assert payload["pull_count"] == 0


def test_list_pockets_smoke(pockets_client: TestClient) -> None:
    """
    Smoke test for GET /api/v1/pockets/ endpoint.
    """
    response = pockets_client.get("/api/v1/pockets/?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["id"] == "pocket-1"
    assert payload[1]["id"] == "pocket-2"


def test_pull_pocket_smoke(pockets_client: TestClient) -> None:
    """
    Smoke test for POST /api/v1/pockets/{pocket_id}/pull endpoint.
    """
    response = pockets_client.post("/api/v1/pockets/pocket-1/pull")

    assert response.status_code == 200
    payload = response.json()
    assert payload["pocket"]["id"] == "pocket-1"
    assert payload["pocket"]["pull_count"] == 3
    assert payload["compensation_increment"] == 0.01
    assert payload["retrieved_content"] == "Pulled snapshot content"


def test_pull_pocket_returns_402_for_unpaid_non_owner(pockets_client: TestClient) -> None:
    """
    Non-owners must pay before pulling active pocket content.
    """
    app.dependency_overrides[get_current_user] = lambda: {"_id": "agent-test-id"}

    with patch("app.api.v1.pockets._get_wallet_balance", new_callable=AsyncMock) as mock_balance:
        mock_balance.return_value = 0.0
        response = pockets_client.post("/api/v1/pockets/pocket-1/pull")

    assert response.status_code == 402
    payload = response.json()
    assert payload["error"] == "payment_required"
    assert payload["price"] == 0.01
    assert payload["pocket_id"] == "pocket-1"
