"""Offline retry-budget regressions; no external API or GPU requests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from parallelworld_manipulation.parallelworld.prompt_enhancer import PromptEnhancer
from parallelworld_manipulation.parallelworld.verifier import (
    VerifierAgent, _AgentRequestBudget, _make_agent, _run_agent_output,
)


class ProviderError(Exception):
    def __init__(self, status_code=503, message="service unavailable"):
        super().__init__(message)
        self.status_code = status_code


class SequenceAgent:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def run_sync(self, content):
        self.calls.append(content)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(output=output)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("parallelworld_manipulation.parallelworld.verifier.time.sleep", recorded.append)
    return recorded


def test_fifth_attempt_recovers_with_identical_images(sleeps, caplog):
    content = ["candidate A", b"unchanged image"]
    agent = SequenceAgent([ProviderError() for _ in range(4)] + ["recovered"])
    assert _run_agent_output(agent, content, operation="verifier") == "recovered"
    assert len(agent.calls) == 5
    assert all(call is content for call in agent.calls)
    assert sleeps == [1, 2, 4, 8]
    assert "operation=verifier attempt=4/5" in caplog.text


def test_five_failures_reraise_original_without_sixth_call(sleeps, caplog):
    final_error = ProviderError()
    agent = SequenceAgent([ProviderError() for _ in range(4)] + [final_error, "unused"])
    with pytest.raises(ProviderError) as caught:
        _run_agent_output(agent, "request")
    assert caught.value is final_error
    assert len(agent.calls) == 5
    assert sleeps == [1, 2, 4, 8]
    assert "attempt=5/5" in caplog.text
    assert "retry=False delay_s=0" in caplog.text


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_statuses_retry(status, sleeps):
    agent = SequenceAgent([ProviderError(status), "ok"])
    assert _run_agent_output(agent, "request") == "ok"
    assert len(agent.calls) == 2
    assert sleeps == [1]


@pytest.mark.parametrize("error", [
    ProviderError(400), ProviderError(401), ProviderError(403), ProviderError(404),
    ProviderError(429, "insufficient_quota"),
    ProviderError(429, "insufficient_user_quota"),
    ProviderError(503, "余额不足"),
    ValueError("invalid input"), AttributeError("another attribute error"),
])
def test_permanent_or_local_errors_do_not_transport_retry(error, sleeps):
    agent = SequenceAgent([error, "unused"])
    with pytest.raises(type(error)):
        _run_agent_output(agent, "request")
    assert len(agent.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ConnectionError("reset")])
def test_transport_errors_retry(error, sleeps):
    agent = SequenceAgent([error, "ok"])
    assert _run_agent_output(agent, "request") == "ok"
    assert sleeps == [1]


def test_budget_is_shared_and_empty_budget_makes_no_request(sleeps):
    budget = _AgentRequestBudget()
    agent = SequenceAgent(["bad structure"] + [ProviderError() for _ in range(4)] + ["unused"])
    assert _run_agent_output(agent, "initial", budget=budget) == "bad structure"
    with pytest.raises(ProviderError):
        _run_agent_output(agent, "repair", budget=budget)
    assert budget.used == 5
    assert budget.remaining == 0
    with pytest.raises(RuntimeError, match="budget exhausted"):
        _run_agent_output(agent, "unused", budget=budget)
    assert len(agent.calls) == 5
    assert sleeps == [1, 2, 4]


@pytest.mark.parametrize("recover", [True, False])
def test_verifier_only_falls_back_after_five_transient_failures(recover, sleeps):
    verifier = object.__new__(VerifierAgent)
    decision = json.dumps({"keep": ["second"], "dominant_id": "second",
                           "continue_search": False, "reason": "visual comparison"})
    verifier._agent = SequenceAgent([ProviderError() for _ in range(4)] + [
        decision if recover else ProviderError(), "unused",
    ])
    result = verifier.decide(depth=1, candidates=[{"id": "first"}, {"id": "second"}])
    assert len(verifier._agent.calls) == 5
    assert result["dominant_id"] == ("second" if recover else "first")
    assert result["reason"].startswith("verifier_fallback:") is not recover


def enhance(outputs):
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = SequenceAgent(outputs)
    candidate = {
        "id": "release", "primitive": "release", "arguments": {},
        "visual_action": {
            "subject": "robot gripper",
            "motion": "opens over the target and releases the held bowl",
            "target": "lower patterned bowl beside cookie package",
            "target_qualifier": "the lower one, not the upper-right bowl",
            "spatial_relation": "directly above", "gripper_state": "open",
            "contact": "contact remains stable only for this immediate action",
            "intended_outcome": "the bowl rests on the support while the gripper stays open",
        },
    }
    result = enhancer.enhance_batch(
        candidates=[candidate], task="release the lower patterned bowl onto the support",
        fronts={"release": np.zeros((7, 11, 3), dtype=np.uint8)}, rollout_steps=49,
    )
    return enhancer._agent, result["release"]


def valid_enhancement():
    return json.dumps({"candidates": [{
        "candidate_id": "release",
        "target_identity": "lower patterned bowl beside cookie package",
        "initial_state": "gripper starts above bowl",
        "motion_phases": ["open gripper and release bowl"],
        "contact_and_gripper": "fingers open over support",
        "object_response": "released bowl settles naturally",
        "terminal_state": "bowl rests gripper stays open",
        "candidate_contrast": "distinct release endpoint",
        "natural_instruction": (
            "The gripper opens over the support and releases the lower patterned bowl. "
            "The bowl settles naturally while the open gripper stays at the same pose."
        ),
    }]})


def test_enhancer_recovers_on_fifth_initial_attempt(sleeps):
    agent, record = enhance([ProviderError() for _ in range(4)] + [valid_enhancement()])
    assert len(agent.calls) == 5
    assert all(call is agent.calls[0] for call in agent.calls)
    assert record["source"] == "llm"
    assert record["repair_attempted"] is False
    assert record["fallback_reason"] is None


def test_enhancer_five_failures_skip_repair_without_another_five_calls(sleeps):
    agent, record = enhance([ProviderError() for _ in range(5)] + ["unused"])
    assert len(agent.calls) == 5
    assert record["source"] == "semantic_fallback"
    assert record["repair_attempted"] is False
    assert "budget exhausted after 5 attempts" in record["fallback_reason"]


def test_enhancer_fifth_invalid_response_still_has_full_repair_budget(sleeps):
    agent, record = enhance([ProviderError() for _ in range(4)] + ["{}"]
                            + [ProviderError() for _ in range(3)] + [valid_enhancement()])
    assert len(agent.calls) == 9
    assert record["source"] == "llm_repair"
    assert record["repair_attempted"] is True
    assert sleeps == [1, 2, 4, 8, 1, 2, 4]


@pytest.mark.parametrize("recover", [True, False])
def test_enhancer_repair_retries_three_times(recover, sleeps):
    agent, record = enhance([ProviderError(), "{}", ProviderError(), ProviderError(), ProviderError(),
                             valid_enhancement() if recover else ProviderError(), "unused"])
    assert len(agent.calls) == 6
    assert agent.calls[0] is agent.calls[1]
    assert all(call is agent.calls[2] for call in agent.calls[2:])
    assert json.loads(agent.calls[0][0])["mode"] == "initial"
    assert json.loads(agent.calls[2][0])["mode"] == "repair"
    assert record["repair_attempted"] is True
    assert record["source"] == ("llm_repair" if recover else "semantic_fallback")
    assert sleeps == [1, 1, 2, 4]


def test_enhancer_invalid_repair_response_does_not_start_another_semantic_round(sleeps):
    agent, record = enhance(["{}", "{}", valid_enhancement()])
    assert len(agent.calls) == 2
    assert record["source"] == "semantic_fallback"
    assert sleeps == []


def test_enhancer_permanent_repair_error_is_not_retried(sleeps):
    agent, record = enhance(["{}", ProviderError(401), valid_enhancement()])
    assert len(agent.calls) == 2
    assert record["source"] == "semantic_fallback"
    assert sleeps == []


def test_enhancer_permanent_initial_error_does_not_start_repair(sleeps):
    agent, record = enhance([ProviderError(401), valid_enhancement()])
    assert len(agent.calls) == 1
    assert record["repair_attempted"] is False
    assert sleeps == []


def test_sdk_retries_disabled_on_shared_openai_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    agent = _make_agent(model="openai:gpt-5.5", base_url="https://example.invalid/v1")
    assert agent.model.client.max_retries == 0


@pytest.mark.parametrize("component", [VerifierAgent, PromptEnhancer])
def test_explicit_openai_agents_send_direct_nonstreaming_responses(monkeypatch, component):
    import base64
    import io
    from PIL import Image
    from openai import AsyncOpenAI
    from pydantic_ai import BinaryContent
    from parallelworld_manipulation import codex_transport

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(codex_transport.CodexJSONAgent, "__init__",
                        lambda *a, **kw: pytest.fail("default auxiliary started Codex"))
    sdk_globals = AsyncOpenAI.__init__.__globals__
    httpx = sdk_globals.get("httpx2") or sdk_globals["httpx"]
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-5.6-sol", "status": "completed",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": '{"ok":true}', "annotations": []}]}],
        })

    agent = component(model="openai:gpt-5.6-sol")._agent
    agent.model.client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    image = io.BytesIO()
    Image.new("RGB", (17, 9), "red").save(image, format="PNG")
    result = _run_agent_output(agent, ["front image", BinaryContent(
        data=image.getvalue(), media_type="image/png", vendor_metadata={"detail": "high"})])
    assert json.loads(result) == {"ok": True}
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/responses"
    assert calls[0].headers["authorization"] == "Bearer offline-test-key"
    body = json.loads(calls[0].content)
    assert body["model"] == "gpt-5.6-sol" and body.get("stream", False) is False
    assert body.get("text", {}).get("format", {}).get("type") != "json_schema"
    images = [part for item in body["input"] if isinstance(item.get("content"), list)
              for part in item["content"] if part["type"] == "input_image"]
    assert len(images) == 1 and images[0]["detail"] == "high"
    assert base64.b64decode(images[0]["image_url"].split(",", 1)[1]) == image.getvalue()


@pytest.mark.parametrize("status, expected", [(503, 5), (401, 1)])
def test_codex_expansion_retry_budget(monkeypatch, sleeps, tmp_path, status, expected):
    from parallelworld_manipulation import codex_transport
    from parallelworld_manipulation.planner.codex import CodexPlanner

    config = tmp_path / "api.sh"
    config.write_text('export OPENAI_API_KEY="offline-test-key"\n'
                      'export OPENAI_BASE_URL="https://example.invalid/v1"\n')
    monkeypatch.setenv("PARALLELWORLD_API_CONFIG", str(config))
    monkeypatch.setenv("PARALLELWORLD_CODEX_AUTH", "api")
    calls = []

    def request(self, content):
        calls.append(content)
        raise codex_transport.CodexRequestError("test failure", status_code=status, retryable=status == 503)

    monkeypatch.setattr(codex_transport.CodexJSONAgent, "run_sync", request)
    planner = CodexPlanner(output_dir=str(tmp_path), model="gpt-5.5",
                           dashboard_events=SimpleNamespace(emit=lambda _: None))
    with pytest.raises(codex_transport.CodexRequestError):
        planner.expand_parallel_routes(route={"id": "parent"}, limit=4)
    assert len(calls) == expected
    assert all(r is calls[0] for r in calls)
    assert sleeps == ([1, 2, 4, 8] if expected == 5 else [])


@pytest.mark.parametrize("failure, expected", [
    (503, 5), (429, 5), (401, 1), ("insufficient_quota", 1),
    ("timeout", 5), ("connection", 5),
])
def test_real_sdk_obeys_http_request_budget(monkeypatch, sleeps, failure, expected):
    from openai import AsyncOpenAI, APIConnectionError
    from pydantic_ai.exceptions import ModelAPIError

    # Use the HTTP implementation actually used by this installed SDK.
    sdk_globals = AsyncOpenAI.__init__.__globals__
    httpx = sdk_globals.get("httpx2") or sdk_globals["httpx"]
    calls = []

    async def respond(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("test timeout", request=request)
        if failure == "connection":
            raise httpx.ConnectError("test connection reset", request=request)
        code = "insufficient_quota" if failure == "insufficient_quota" else "server_error"
        return httpx.Response(429 if failure == "insufficient_quota" else failure,
                              json={"error": {"message": code, "type": code}})

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    agent = _make_agent(model="openai:gpt-5.5", base_url="https://example.invalid/v1")
    agent.model.client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with pytest.raises((ModelAPIError, APIConnectionError)):
        _run_agent_output(agent, "offline retry count check")
    assert len(calls) == expected
    assert sleeps == ([1, 2, 4, 8] if expected == 5 else [])
