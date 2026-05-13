"""
META-STAMP V3 Pockets MCP Server Tests.

Tests for:
- MCP JSON-RPC 2.0 endpoint
- Agent authentication
- Agreement auto-creation
- Tool execution (pull_content, search_pockets, list_pockets)
- Rate limiting
- Error handling
"""

import json

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.middleware import check_agreement, check_rate_limit, get_current_agent
from app.models.agent import AgentAPIKey, AgentProvider


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_agent() -> AgentAPIKey:
    """Create a mock AgentAPIKey for testing."""
    return AgentAPIKey(
        _id="agent-key-123",
        key_hash="a" * 64,
        key_prefix="pkt_test1234",
        provider=AgentProvider.OPENAI,
        provider_name="OpenAI Test",
        rate_limit_per_minute=100,
        is_active=True,
        created_at=datetime.now(UTC),
        last_used_at=None,
    )


def _jsonrpc_from_sse(response) -> dict:
    """Extract the single JSON-RPC payload from an MCP SSE response."""
    assert response.headers["content-type"].startswith("text/event-stream")
    payload = response.text.strip()
    assert payload.startswith("data: ")
    return json.loads(payload.removeprefix("data: ").strip())


@pytest.fixture
def mcp_client() -> TestClient:
    """TestClient with MCP middleware overrides."""
    mock_agent = _make_mock_agent()

    app.dependency_overrides[get_current_agent] = lambda: mock_agent
    app.dependency_overrides[check_agreement] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


# ============================================================================
# Manifest Tests
# ============================================================================


class TestMCPManifest:
    """Tests for the MCP tool manifest endpoint."""

    def test_get_manifest_returns_tools(self, mcp_client: TestClient) -> None:
        response = mcp_client.get("/mcp/manifest")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) >= 3  # At least the original 3 tools
        tool_names = [tool["name"] for tool in data["tools"]]
        assert "pull_content" in tool_names
        assert "search_pockets" in tool_names
        assert "list_pockets" in tool_names

    def test_manifest_includes_terms_notice(self, mcp_client: TestClient) -> None:
        response = mcp_client.get("/mcp/manifest")
        data = response.json()
        assert "terms" in data
        assert "accept" in data["terms"].lower()

    def test_get_tools_list_returns_mcp_spec_shape(self, mcp_client: TestClient) -> None:
        response = mcp_client.get("/mcp/tools/list")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) >= 3

        pull_tool = next(tool for tool in data["tools"] if tool["name"] == "pull_content")
        assert "description" in pull_tool
        assert "inputSchema" in pull_tool
        assert pull_tool["inputSchema"]["type"] == "object"
        assert "pocket_id" in pull_tool["inputSchema"]["properties"]

    def test_server_card_nested_well_known_path(self, mcp_client: TestClient) -> None:
        response = mcp_client.get("/.well-known/mcp/server-card.json")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "meta-stamp-pockets"
        assert data["url"] == "https://metastampv3-production.up.railway.app/mcp"
        assert data["capabilities"]["tools"] is True
        assert data["capabilities"]["resources"] is False
        assert data["capabilities"]["prompts"] is False


# ============================================================================
# JSON-RPC Protocol Tests
# ============================================================================


class TestJSONRPCProtocol:
    """Tests for JSON-RPC 2.0 protocol compliance."""

    def test_invalid_json_returns_parse_error(self, mcp_client: TestClient) -> None:
        response = mcp_client.post(
            "/mcp/",
            content="not json",
            headers={"Content-Type": "application/json", "Authorization": "Bearer pkt_test"},
        )
        assert response.status_code == 200
        data = _jsonrpc_from_sse(response)
        assert data["error"]["code"] == -32700

    def test_missing_jsonrpc_version(self, mcp_client: TestClient) -> None:
        response = mcp_client.post(
            "/mcp/",
            json={"method": "list_pockets", "id": 1},
            headers={"Authorization": "Bearer pkt_test"},
        )
        data = _jsonrpc_from_sse(response)
        assert data["error"]["code"] == -32600

    def test_unknown_method_returns_method_not_found(self, mcp_client: TestClient) -> None:
        response = mcp_client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "method": "nonexistent", "params": {}, "id": 1},
            headers={"Authorization": "Bearer pkt_test"},
        )
        data = _jsonrpc_from_sse(response)
        assert data["error"]["code"] == -32601

    def test_valid_request_structure(self, mcp_client: TestClient) -> None:
        """Test that a valid JSON-RPC request returns proper structure."""
        with patch("app.mcp.tools.MCPTools.list_pockets", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = {"count": 0, "results": []}
            response = mcp_client.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "method": "list_pockets", "params": {}, "id": 1},
                headers={"Authorization": "Bearer pkt_test"},
            )
            data = _jsonrpc_from_sse(response)
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert "result" in data


# ============================================================================
# Tool Tests
# ============================================================================


class TestMCPTools:
    """Tests for MCP tool execution."""

    def test_list_pockets_returns_results(self, mcp_client: TestClient) -> None:
        with patch("app.mcp.tools.MCPTools.list_pockets", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = {
                "count": 1,
                "results": [
                    {
                        "pocket_id": "pocket-1",
                        "creator_id": "creator-1",
                        "content_url": "https://example.com",
                        "content_type": "webpage",
                        "pull_count": 5,
                        "title": "Test Pocket",
                        "description": "Test description",
                    }
                ],
            }
            response = mcp_client.post(
                "/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "method": "list_pockets",
                    "params": {"limit": 10},
                    "id": 1,
                },
                headers={"Authorization": "Bearer pkt_test"},
            )
            data = _jsonrpc_from_sse(response)
            assert "result" in data
            assert data["result"]["count"] == 1

    def test_search_pockets_requires_query(self, mcp_client: TestClient) -> None:
        with patch("app.mcp.tools.MCPTools.search_pockets", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = ValueError("query is required")
            response = mcp_client.post(
                "/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "method": "search_pockets",
                    "params": {"query": ""},
                    "id": 2,
                },
                headers={"Authorization": "Bearer pkt_test"},
            )
            data = _jsonrpc_from_sse(response)
            assert "error" in data
            assert data["error"]["code"] == -32602

    def test_pull_content_returns_licensed_content(self, mcp_client: TestClient) -> None:
        with patch("app.mcp.tools.MCPTools.pull_content", new_callable=AsyncMock) as mock_pull:
            mock_pull.return_value = {
                "pocket_id": "pocket-1",
                "content": "Pre-indexed content here",
                "content_type": "youtube",
                "metadata": {"title": "Test Video"},
                "license": {
                    "terms_version": "1.0.0",
                    "pull_price_usd": 0.01,
                    "licensed": True,
                },
                "response_time_ms": 5.2,
            }
            response = mcp_client.post(
                "/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "method": "pull_content",
                    "params": {"pocket_id": "pocket-1"},
                    "id": 3,
                },
                headers={"Authorization": "Bearer pkt_test"},
            )
            data = _jsonrpc_from_sse(response)
            assert "result" in data
            result = data["result"]
            assert result["license"]["licensed"] is True
            assert result["content"] == "Pre-indexed content here"
            assert result["response_time_ms"] > 0


# ============================================================================
# Agent Model Tests
# ============================================================================


class TestAgentModels:
    """Tests for agent-related Pydantic models."""

    def test_agent_api_key_validates_hash(self) -> None:
        key = AgentAPIKey(
            key_hash="a" * 64,
            key_prefix="pkt_test1234",
            provider=AgentProvider.OPENAI,
            provider_name="OpenAI",
        )
        assert key.key_hash == "a" * 64

    def test_agent_api_key_rejects_short_hash(self) -> None:
        with pytest.raises(ValueError):
            AgentAPIKey(
                key_hash="tooshort",
                key_prefix="pkt_test1234",
                provider=AgentProvider.OPENAI,
                provider_name="OpenAI",
            )

    def test_agent_api_key_rejects_non_hex_hash(self) -> None:
        with pytest.raises(ValueError):
            AgentAPIKey(
                key_hash="z" * 64,
                key_prefix="pkt_test1234",
                provider=AgentProvider.OPENAI,
                provider_name="OpenAI",
            )


# ============================================================================
# Agreement Model Tests
# ============================================================================


class TestAgreementModels:
    """Tests for agreement-related Pydantic models."""

    def test_agreement_validates_terms_version(self) -> None:
        from app.models.agreement import Agreement

        agreement = Agreement(
            agent_key_id="key-1",
            provider="openai",
            terms_version="1.0.0",
        )
        assert agreement.terms_version == "1.0.0"

    def test_agreement_rejects_invalid_version(self) -> None:
        from app.models.agreement import Agreement

        with pytest.raises(ValueError):
            Agreement(
                agent_key_id="key-1",
                provider="openai",
                terms_version="invalid",
            )


# ============================================================================
# Terms Endpoint Tests
# ============================================================================


class TestTermsEndpoint:
    """Tests for the terms of service endpoint."""

    def test_get_terms_returns_current_version(self, mcp_client: TestClient) -> None:
        response = mcp_client.get("/api/v1/agreements/terms")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"
        assert "full_text" in data
        assert len(data["full_text"]) > 0


# ============================================================================
# Agent Key Management Tests
# ============================================================================


class TestAgentKeyManagement:
    """Tests for agent API key CRUD endpoints."""

    def test_create_agent_key(self, mcp_client: TestClient) -> None:
        from app.api.v1.agents import get_agent_auth_service
        from app.core.auth import get_current_user

        mock_service = AsyncMock()
        mock_key = _make_mock_agent()
        mock_service.create_agent_key.return_value = ("agent-key-created-for-test", mock_key)

        app.dependency_overrides[get_current_user] = lambda: {"_id": "admin-1", "email": "admin@test.com"}
        app.dependency_overrides[get_agent_auth_service] = lambda: mock_service

        response = mcp_client.post(
            "/api/v1/agents/keys",
            json={
                "provider": "openai",
                "provider_name": "OpenAI Test",
                "rate_limit_per_minute": 100,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "raw_key" in data
        assert data["raw_key"] == "agent-key-created-for-test"
