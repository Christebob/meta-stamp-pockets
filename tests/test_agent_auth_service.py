"""
Tests for AgentAuthService — API key generation, validation, and lifecycle.
"""

from app.services.agent_auth_service import AgentAuthService


class TestAgentAuthService:
    """Tests for agent authentication service."""

    def test_generate_api_key_format(self) -> None:
        raw_key, key_hash, key_prefix = AgentAuthService.generate_api_key()
        assert raw_key.startswith("pkt_")
        assert len(key_hash) == 64
        assert key_prefix == raw_key[:12]

    def test_generate_api_key_uniqueness(self) -> None:
        keys = set()
        for _ in range(100):
            raw_key, _, _ = AgentAuthService.generate_api_key()
            keys.add(raw_key)
        assert len(keys) == 100  # All unique
