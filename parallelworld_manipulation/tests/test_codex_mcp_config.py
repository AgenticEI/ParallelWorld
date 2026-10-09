import json

import pytest

from parallelworld_manipulation.planner.codex import _codex_mcp_config_overrides


def _as_dict(overrides: list[str]) -> dict[str, object]:
    result = {}
    for override in overrides:
        key, raw_value = override.split("=", 1)
        result[key] = json.loads(raw_value)
    return result


def test_parallelworld_mcp_timeout_defaults_to_four_hours(monkeypatch):
    monkeypatch.delenv("PARALLELWORLD_MCP_TOOL_TIMEOUT_S", raising=False)

    config = _as_dict(_codex_mcp_config_overrides(
        mcp_url="http://127.0.0.1:1234/mcp/", base_url=None,
    ))

    assert config["mcp_servers.parallelworld_manipulation.url"] == "http://127.0.0.1:1234/mcp/"
    assert config["mcp_servers.parallelworld_manipulation.tool_timeout_sec"] == 14400.0


def test_parallelworld_mcp_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PARALLELWORLD_MCP_TOOL_TIMEOUT_S", "1800")

    config = _as_dict(_codex_mcp_config_overrides(
        mcp_url="http://127.0.0.1:1234/mcp/", base_url="https://example.test",
    ))

    assert config["mcp_servers.parallelworld_manipulation.tool_timeout_sec"] == 1800.0
    assert config["model_providers.parallelworld_manipulation_proxy.base_url"] == "https://example.test/v1"


def test_parallelworld_preserves_explicit_api_base_path(monkeypatch):
    monkeypatch.delenv("PARALLELWORLD_MCP_TOOL_TIMEOUT_S", raising=False)

    config = _as_dict(_codex_mcp_config_overrides(
        mcp_url="http://127.0.0.1:1234/mcp/",
        base_url="https://example.test/api/",
    ))

    assert config["model_providers.parallelworld_manipulation_proxy.base_url"] == "https://example.test/api"
    assert config["model_providers.parallelworld_manipulation_proxy.stream_max_retries"] == 10
    assert config["model_providers.parallelworld_manipulation_proxy.request_max_retries"] == 10


def test_parallelworld_mcp_timeout_must_be_positive(monkeypatch):
    monkeypatch.setenv("PARALLELWORLD_MCP_TOOL_TIMEOUT_S", "0")

    with pytest.raises(ValueError, match="must be greater than zero"):
        _codex_mcp_config_overrides(
            mcp_url="http://127.0.0.1:1234/mcp/", base_url=None,
        )
