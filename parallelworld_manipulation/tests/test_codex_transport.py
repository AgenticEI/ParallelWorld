"""Offline Codex wiring tests: no model requests, GPU jobs, or login mutations."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import openai_codex
import pytest
from PIL import Image
from pydantic_ai import BinaryContent

from parallelworld_manipulation import codex_transport as transport
from parallelworld_manipulation.parallelworld.prompt_enhancer import PromptEnhancer
from parallelworld_manipulation.parallelworld.verifier import VerifierAgent, _make_agent, _run_agent_output


@pytest.fixture(autouse=True)
def legacy_login_mode(monkeypatch):
    # Keep the original login regression cases explicit; API cases override this.
    monkeypatch.setenv("PARALLELWORLD_CODEX_AUTH", "chatgpt")
    monkeypatch.delenv("PARALLELWORLD_API_PROFILE", raising=False)


@pytest.fixture
def fake_codex(monkeypatch):
    state = SimpleNamespace(configs=[], accounts=0, starts=[], runs=[], closed=0,
                            outputs=[], account_type="chatgpt")

    class Client:
        def __init__(self, config):
            state.configs.append(config)
            self._client = SimpleNamespace(_request_raw=lambda *args: {
                "config": {"mcp_servers": {"unwanted": {"command": "/bin/false"}}},
            })

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed += 1

        def account(self, **kwargs):
            state.accounts += 1
            assert kwargs == {"refresh_token": False}
            return SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type=state.account_type)))

        def thread_start(self, **kwargs):
            state.starts.append(kwargs)
            return self

        def run(self, inputs, **kwargs):
            state.runs.append((inputs, kwargs))
            output = state.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return SimpleNamespace(final_response=output)

    monkeypatch.setattr(openai_codex, "Codex", Client)
    monkeypatch.setattr("parallelworld_manipulation.parallelworld.verifier.time.sleep", lambda _: None)
    return state


def png(width=17, height=9):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (23, 51, 92)).save(buf, format="PNG")
    return buf.getvalue()


def config_dict(config):
    import tomllib
    return {k: tomllib.loads("value=" + v)["value"] for k, v in (x.split("=", 1) for x in config.config_overrides)}


def test_environment_is_overlay_safe_and_leaves_image_configuration_unchanged(monkeypatch):
    for key in transport._API_ENV_KEYS:
        monkeypatch.setenv(key, "private-image-api-setting")
    monkeypatch.setenv("CODEX_HOME", "/tmp/existing-login")
    before = dict(os.environ)
    env = transport.chatgpt_environment()
    # Mimic the installed SDK's os.environ.copy(); env.update(config.env).
    merged = {**os.environ, **env}
    assert all(merged[key] == "" for key in transport._API_ENV_KEYS)
    assert merged["CODEX_HOME"] == "/tmp/existing-login"
    assert dict(os.environ) == before


def test_actual_sdk_subprocess_receives_no_image_credentials(tmp_path, monkeypatch):
    from openai_codex import client

    monkeypatch.setenv("OPENAI_API_KEY", "not-for-codex")
    monkeypatch.setenv("GPT_IMAGE_API_KEY", "image-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    seen = {}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise Captured()

    monkeypatch.setattr(client.subprocess, "Popen", capture)
    with pytest.raises(Captured):
        with openai_codex.Codex(config=transport._config(cwd=str(tmp_path))):
            pytest.fail("No child should actually start")
    assert seen["env"]["OPENAI_API_KEY"] == ""
    assert seen["env"]["GPT_IMAGE_API_KEY"] == ""
    assert seen["env"]["OPENAI_BASE_URL"] == ""
    assert os.environ["GPT_IMAGE_API_KEY"] == "image-only"


@pytest.mark.parametrize("kind", [None, "apiKey", "amazonBedrock"])
def test_non_chatgpt_account_fails_before_any_turn(fake_codex, kind):
    fake_codex.account_type = kind
    with pytest.raises(transport.CodexAuthenticationError, match="CODEX_CHATGPT_LOGIN_REQUIRED"):
        transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON").run_sync("test")
    assert fake_codex.runs == fake_codex.starts == []
    assert fake_codex.closed == 1


def test_auth_check_handles_real_sdk_root_model():
    from openai_codex.types import GetAccountResponse

    response = GetAccountResponse.model_validate({
        "account": {"type": "chatgpt", "email": "test@example.invalid", "planType": "plus"},
        "requiresOpenaiAuth": True,
    })
    transport.require_chatgpt_account(response)


def test_preflight_never_starts_a_model_turn(fake_codex):
    transport.check_chatgpt_login()
    assert fake_codex.accounts == 1
    assert not fake_codex.starts and not fake_codex.runs


def test_native_images_labels_and_order_survive_real_sdk_serialization(fake_codex):
    from openai_codex._inputs import _to_wire_input

    data = png()
    fake_codex.outputs = ['{"keep":["A"]}']
    content = ["candidate A; frame 0; front", BinaryContent(data=data, media_type="image/png"),
               "candidate B; frame 10; front", BinaryContent(data=png(21, 13), media_type="image/png")]
    agent = transport.CodexJSONAgent(model="codex:gpt-5.5", system_prompt="independent verifier",
                                    output_schema=transport.VERIFIER_OUTPUT_SCHEMA)
    assert agent.run_sync(content).output == '{"keep":["A"]}'
    wire = _to_wire_input(fake_codex.runs[0][0])
    assert [x["type"] for x in wire] == ["text", "image", "text", "image"]
    assert wire[0]["text"] == content[0] and wire[2]["text"] == content[2]
    actual = base64.b64decode(wire[1]["url"].split(",", 1)[1])
    assert actual == data
    assert Image.open(io.BytesIO(actual)).size == (17, 9)
    assert fake_codex.runs[0][1]["output_schema"] == transport.VERIFIER_OUTPUT_SCHEMA
    opts = fake_codex.starts[0]
    assert opts["model"] == "gpt-5.5" and opts["model_provider"] == transport.CHATGPT_PROVIDER
    assert opts["ephemeral"] is True
    assert opts["sandbox"] == openai_codex.Sandbox.read_only
    assert opts["approval_mode"] == openai_codex.ApprovalMode.deny_all
    assert opts["config"] == {"mcp_servers": {"unwanted": {"enabled": False}}}
    assert not Path(opts["cwd"]).exists()  # temporary context cleaned up
    config = config_dict(fake_codex.configs[0])
    assert config["mcp_servers"] == {}
    assert config["features.shell_tool"] is False
    assert config["web_search"] == "disabled"
    provider = config[f"model_providers.{transport.CHATGPT_PROVIDER}"]
    # The normal JSON agent leaves managed-login transport retries to the SDK;
    # only explicitly bounded configurations override these provider fields.
    assert "request_max_retries" not in provider
    assert "stream_max_retries" not in provider
    assert provider["requires_openai_auth"] is True
    assert "base_url" not in provider and "env_key" not in provider


def test_requests_use_independent_threads(fake_codex):
    fake_codex.outputs = ['{}', '{}']
    agent = transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON")
    agent.run_sync("first")
    agent.run_sync("second")
    assert len(fake_codex.starts) == 2 and fake_codex.closed == 2
    assert fake_codex.starts[0]["cwd"] != fake_codex.starts[1]["cwd"]


@pytest.mark.parametrize("error", [
    RuntimeError("unexpected status 503 Service Unavailable"),
    RuntimeError("status_code: 524"), RuntimeError("HTTP 429"),
    RuntimeError("stream disconnected before completion"),
    openai_codex.TransportClosedError("connection lost"),
    openai_codex.ServerBusyError(-32001, "busy"),
])
def test_codex_transient_errors_share_five_attempt_budget(fake_codex, error):
    fake_codex.outputs = [error] * 4 + ['{"ok":true}']
    agent = transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON")
    assert _run_agent_output(agent, "same", operation="verifier") == '{"ok":true}'
    assert len(fake_codex.runs) == 5 and fake_codex.closed == 5


def test_codex_exhaustion_stops_after_five(fake_codex):
    fake_codex.outputs = [RuntimeError("unexpected status 503")] * 6
    with pytest.raises(transport.CodexRequestError):
        _run_agent_output(transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON"), "same")
    assert len(fake_codex.runs) == 5


@pytest.mark.parametrize("message", ["HTTP 400 invalid input", "HTTP 401 unauthorized",
    "HTTP 429 usage limit reached", "HTTP 503 insufficient quota", "模型已禁用: gpt-5.5"])
def test_permanent_codex_error_not_retried(fake_codex, message):
    fake_codex.outputs = [RuntimeError(message), '{}']
    with pytest.raises(transport.CodexRequestError):
        _run_agent_output(transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON"), "same")
    assert len(fake_codex.runs) == 1


def test_explicit_codex_verifier_and_enhancer_without_openai_client(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: pytest.fail("API transport used"))
    assert isinstance(VerifierAgent(model="codex:gpt-5.5")._agent, transport.CodexJSONAgent)
    enhancer = PromptEnhancer(model="codex:gpt-5.5")
    assert isinstance(enhancer._agent, transport.CodexJSONAgent)
    assert enhancer._agent.output_schema == transport.ENHANCER_OUTPUT_SCHEMA
    assert enhancer._model == "codex:gpt-5.5"
    assert isinstance(_make_agent(model="codex:gpt-5.5", base_url="https://ignored.invalid"), transport.CodexJSONAgent)


def test_verifier_codex_integration(fake_codex):
    fake_codex.outputs = [json.dumps({"keep": ["B"], "dominant_id": "B", "continue_search": False, "reason": "visible improvement"})]
    v = VerifierAgent(model="codex:gpt-5.5")
    result = v.decide(depth=1, candidates=[{"id": "A"}, {"id": "B"}], frames=[
        {"candidate_id": "A", "frame_index": 0, "view": "front", "image": np.zeros((9, 17, 3), dtype=np.uint8)},
    ])
    assert result["dominant_id"] == "B" and "fallback" not in result["reason"]
    assert len(fake_codex.runs[0][0]) == 3


def test_enhancer_repair_has_four_independent_codex_attempts(fake_codex):
    fake_codex.outputs = [RuntimeError("HTTP 503")] * 4 + ['{"candidates":[]}'] + [RuntimeError("HTTP 503")] * 4
    candidate = {"id": "A", "primitive": "pi0_pick", "arguments": {}, "rationale": "pick the bowl",
                 "visual_action": {"subject": "robot gripper", "target": "bowl", "motion": "pick up", "gripper_state": "open"}}
    result = PromptEnhancer(model="codex:gpt-5.5").enhance_batch(candidates=[candidate], task="pick the bowl",
        fronts={"A": np.zeros((9, 17, 3), dtype=np.uint8)}, rollout_steps=1)
    assert len(fake_codex.runs) == 9
    assert result["A"]["source"] == "semantic_fallback"


def test_planner_login_mode_does_not_read_api_sh(tmp_path, monkeypatch):
    from parallelworld_manipulation.planner import codex
    monkeypatch.setenv("PARALLELWORLD_CODEX_AUTH", "chatgpt")
    monkeypatch.setenv("OPENAI_API_KEY", "image-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://image.invalid/v1")
    monkeypatch.setattr(transport, "_read_parallelworld_manipulation_api_config", lambda _: pytest.fail("read api.sh in login mode"))
    planner = codex.CodexPlanner(output_dir=str(tmp_path), repo_root=str(tmp_path), model="gpt-5.5",
                                 dashboard_events=SimpleNamespace(emit=lambda _: None))
    assert planner._api_key is None and planner._base_url is None
    config = planner._build_config("http://127.0.0.1:9999/mcp")
    opts = config_dict(config)
    assert opts["model_provider"] == transport.CHATGPT_PROVIDER
    assert opts["mcp_servers.parallelworld_manipulation.url"] == "http://127.0.0.1:9999/mcp"
    assert not any("parallelworld_manipulation_proxy" in x for x in config.config_overrides)
    assert config.env["OPENAI_API_KEY"] == ""
    assert os.environ["OPENAI_API_KEY"] == "image-only"


def test_expansion_uses_codex_and_only_parent_front(api_mode, fake_codex, monkeypatch):
    import openai

    from parallelworld_manipulation.planner.codex import CodexPlanner
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("direct API used"))
    fake_codex.outputs = ['[{"id":"next","primitive":"move_to","arguments":{}}]']
    planner = CodexPlanner(output_dir=str(api_mode.parent), model="gpt-5.5",
                           dashboard_events=SimpleNamespace(emit=lambda _: None))
    result = planner.expand_parallel_routes(route={"id": "parent", "primitive": "pi0_pick",
        "_imagined_front": np.zeros((9, 17, 3), dtype=np.uint8),
        "_imagined_wrist": np.zeros((9, 17, 3), dtype=np.uint8)}, limit=4)
    assert result[0]["id"] == "next"
    inputs = fake_codex.runs[0][0]
    assert len(inputs) == 3
    assert isinstance(inputs[2], openai_codex.ImageInput)
    assert "parent" in inputs[1].text and "front" in inputs[1].text
    assert "_imagined_wrist" not in inputs[0].text
    assert Image.open(io.BytesIO(base64.b64decode(inputs[2].url.split(",", 1)[1]))).size == (17, 9)
    assert fake_codex.configs[0].env[transport.PROVIDER_ENV_KEY] == "test-only-key"
    assert fake_codex.starts[0]["model"] == "gpt-5.5"
    assert fake_codex.closed == 1 and fake_codex.accounts == 0


@pytest.mark.parametrize("component", [VerifierAgent, PromptEnhancer])
def test_default_auxiliary_uses_codex_api_with_native_png(api_mode, fake_codex, component):
    fake_codex.outputs = ['{"ok":true}']
    agent = component()._agent
    data = png()
    result = agent.run_sync(["front image", BinaryContent(data=data, media_type="image/png")])
    assert json.loads(result.output) == {"ok": True}
    inputs = fake_codex.runs[0][0]
    assert base64.b64decode(inputs[1].url.split(",", 1)[1]) == data
    assert fake_codex.starts[0]["model"] == "gpt-5.6-sol"
    assert fake_codex.starts[0]["config"]["mcp_servers"]["unwanted"]["enabled"] is False
    assert fake_codex.accounts == 0
    assert fake_codex.configs[0].env[transport.PROVIDER_ENV_KEY] == "test-only-key"


def test_launcher_keeps_image_arguments_and_preflights_before_services():
    script = (Path(__file__).resolve().parents[1] / "evaluate_manipulation.sh").read_text(encoding="utf-8")
    assert '--model "${PW_IMAGE_MODEL:-gpt-image-2.5}" --quality low --size 1024x1024' in script
    assert '--request-timeout-s 900 --max-retries 0' in script
    assert '[[ -f api.sh ]] && source api.sh' in script
    assert 'export PARALLELWORLD_CODEX_AUTH=api' in script
    assert '--planner codex --model "$episode_planner_model"' in script
    assert '--wm-verifier-model "$episode_verifier_model"' in script
    assert '--wm-prompt-enhancer-model "$episode_enhancer_model"' in script
    assert script.index('"$python" -m parallelworld_manipulation.codex_transport') < script.index('mkdir -p "$root/services"')


def test_launcher_planner_defaults_and_explicit_overrides_without_services():
    project = Path(__file__).resolve().parents[1]
    batches = {
        "evaluate_spatial.sh": ["libero_spatial_task", "libero_spatial_swap"],
        "evaluate_manipulation.sh": ["libero_object_task", "libero_object_swap",
                                     "libero_goal_task", "libero_goal_swap", "libero_10_task", "libero_10_swap"],
    }
    for filename, suites in batches.items():
        script = (project / filename).read_text(encoding="utf-8")
        # Execute only the role resolver, never the launcher or robot services.
        resolver = "resolve_episode_models() {" + script.split("resolve_episode_models() {", 1)[1].split("episode_complete() {", 1)[0]
        program = '''set -Eeuo pipefail
unset PW_TRIAL_MODEL_CONFIG PW_PLANNER_MODEL
''' + resolver + "\nfor suite in " + " ".join(suites) + '''; do
  unset PW_PLANNER_MODEL
  resolve_episode_models "$suite" 0 1
  case "$suite" in
    libero_spatial_*|libero_object_*) expected=gpt-5.5 ;;
    *) expected=gpt-5.6-sol ;;
  esac
  [[ "$episode_planner_model" == "$expected" ]]
  PW_PLANNER_MODEL=test-planner-override resolve_episode_models "$suite" 0 1
  [[ "$episode_planner_model" == test-planner-override ]]
done
'''
        run = subprocess.run(["bash", "-s"], input=program.encode("utf-8"), capture_output=True)
        assert run.returncode == 0, (filename, run.stderr.decode("utf-8", errors="replace"))


def test_pinned_runtime_accepts_isolated_configuration_without_model_request(tmp_path, monkeypatch):
    """Real local app-server handshake/config read; no turn/start or inference."""
    isolated = tmp_path / "codex-home"
    isolated.mkdir()
    (isolated / "config.toml").write_text(
        'model_provider = "stale"\n'
        '[model_providers.stale]\nname = "old gateway"\nwire_api = "responses"\n'
        'base_url = "https://stale.invalid/v1"\nenv_key = "OPENAI_API_KEY"\n'
        '[model_providers.parallelworld_manipulation_chatgpt]\nname = "stale login provider"\nwire_api = "responses"\n'
        'base_url = "https://stale.invalid/v1"\nenv_key = "OPENAI_API_KEY"\n'
        '[mcp_servers.unwanted]\ncommand = "/bin/false"\n'
    )
    monkeypatch.setenv("CODEX_HOME", str(isolated))
    with openai_codex.Codex(config=transport._config(cwd=str(tmp_path))) as client:
        result = client._client._request_raw("config/read", {"includeLayers": False})
        config = result["config"]
        assert config["model_provider"] == transport.CHATGPT_PROVIDER
        provider = config["model_providers"][transport.CHATGPT_PROVIDER]
        assert not provider.get("base_url") and not provider.get("env_key")
        assert provider["requires_openai_auth"] is True
        assert provider["request_max_retries"] == provider["stream_max_retries"] == 0
        assert transport._disabled_mcp_config(client) == {"mcp_servers": {"unwanted": {"enabled": False}}}
        with pytest.raises(transport.CodexAuthenticationError):
            transport.require_chatgpt_account(client.account(refresh_token=False))
        # Creating the empty ephemeral thread validates runtime option support;
        # deliberately never start a turn, so no model inference occurs.
        thread = client.thread_start(
            model="gpt-5.5", model_provider=transport.CHATGPT_PROVIDER,
            cwd=str(tmp_path), sandbox=openai_codex.Sandbox.read_only,
            approval_mode=openai_codex.ApprovalMode.deny_all, ephemeral=True,
            base_instructions="Return JSON only; no tools.",
            config=transport._disabled_mcp_config(client),
        )
        assert thread.id


@pytest.fixture
def api_mode(tmp_path, monkeypatch):
    path = tmp_path / "api.sh"
    path.write_text('export OPENAI_API_KEY="test-only-key"\n'
                    'export OPENAI_BASE_URL="http://127.0.0.1:9876/custom/api/"\n')
    monkeypatch.delenv("PARALLELWORLD_CODEX_AUTH", raising=False)
    monkeypatch.setenv("PARALLELWORLD_API_CONFIG", str(path))
    monkeypatch.setenv("OPENAI_API_KEY", "stale-parent-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale-parent.invalid/v1")
    monkeypatch.setenv("GPT_IMAGE_API_KEY", "image-only-key")
    return path


def test_api_default_reads_file_and_leaves_parent_image_environment_unchanged(api_mode):
    before = dict(os.environ)
    assert transport.codex_auth_mode() == "api"
    key, url = transport.api_credentials()
    assert key == "test-only-key" and url == "http://127.0.0.1:9876/custom/api"
    config = transport._config(cwd=str(api_mode.parent))
    opts = config_dict(config)
    provider = opts[f"model_providers.{transport.API_PROVIDER}"]
    assert opts["model_provider"] == transport.API_PROVIDER
    assert provider["base_url"] == url
    assert provider["env_key"] == transport.PROVIDER_ENV_KEY
    assert provider["requires_openai_auth"] is False
    assert provider["request_max_retries"] == provider["stream_max_retries"] == 0
    assert key not in repr(config.config_overrides)
    assert config.env[transport.PROVIDER_ENV_KEY] == key
    assert config.env["GPT_IMAGE_API_KEY"] == "image-only-key"
    assert dict(os.environ) == before


def test_api_preflight_and_json_calls_never_read_login_account(api_mode, fake_codex):
    fake_codex.account_type = None
    transport.check_authentication()
    assert not fake_codex.configs
    fake_codex.outputs = ['{"ok":true}']
    agent = transport.CodexJSONAgent(model="gpt-5.5", system_prompt="JSON")
    assert agent.run_sync(["image", BinaryContent(data=png(), media_type="image/png")]).output == '{"ok":true}'
    assert fake_codex.accounts == 0
    assert fake_codex.starts[0]["model_provider"] == transport.API_PROVIDER
    assert fake_codex.configs[0].env[transport.PROVIDER_ENV_KEY] == "test-only-key"


def test_planner_uses_original_api_provider(api_mode, fake_codex):
    from parallelworld_manipulation.planner.codex import CodexPlanner
    planner = CodexPlanner(output_dir=str(api_mode.parent), model="gpt-5.5",
                           dashboard_events=SimpleNamespace(emit=lambda _: None))
    config = planner._build_config("http://127.0.0.1:9999/mcp")
    opts = config_dict(config)
    assert planner._auth_mode == "api"
    assert opts["model_provider"] == "parallelworld_manipulation_proxy"
    assert opts["model_providers.parallelworld_manipulation_proxy.wire_api"] == "responses"
    assert opts["model_providers.parallelworld_manipulation_proxy.base_url"] == planner._base_url
    assert opts["model_providers.parallelworld_manipulation_proxy.env_key"] == transport.PROVIDER_ENV_KEY
    assert config.env[transport.PROVIDER_ENV_KEY] == "test-only-key"
    assert opts["mcp_servers.parallelworld_manipulation.url"] == "http://127.0.0.1:9999/mcp"
    assert not fake_codex.configs and fake_codex.accounts == 0


def test_verifier_and_enhancer_api_retries_keep_budget(api_mode, fake_codex):
    fake_codex.account_type = None
    fake_codex.outputs = [RuntimeError("HTTP 503")] * 4 + [json.dumps({
        "keep": ["A"], "dominant_id": "A", "continue_search": False, "reason": "visible success"})]
    result = VerifierAgent(model="codex:gpt-5.5").decide(depth=1, candidates=[{"id": "A"}], frames=[])
    assert result["dominant_id"] == "A" and "fallback" not in result["reason"]
    assert len(fake_codex.runs) == 5
    fake_codex.outputs = [RuntimeError("HTTP 503")] * 5
    candidate = {"id": "A", "primitive": "pi0_pick", "arguments": {}, "rationale": "pick bowl"}
    result = PromptEnhancer(model="codex:gpt-5.5").enhance_batch(candidates=[candidate], task="pick bowl",
        fronts={"A": np.zeros((9, 17, 3), dtype=np.uint8)}, rollout_steps=1)
    assert result["A"]["source"] == "semantic_fallback"
    assert len(fake_codex.runs) == 10 and fake_codex.accounts == 0
    assert all(s["model_provider"] == transport.API_PROVIDER for s in fake_codex.starts)


@pytest.mark.parametrize("exports", [
    '', 'export OPENAI_API_KEY=""\n',
    'export OPENAI_API_KEY="$SOME_KEY"\n',
    'export OPENAI_BASE_URL="not-a-url"\n',
])
def test_invalid_api_config_fails_without_login_or_inference(api_mode, fake_codex, monkeypatch, exports):
    api_mode.write_text(exports)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CODEX_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(transport.CodexAuthenticationError):
        transport.check_authentication()
    assert not fake_codex.configs and not fake_codex.accounts


def test_missing_explicit_api_file_does_not_use_neighbor_scripts(api_mode):
    api_mode.rename(api_mode.with_name("unrelated.sh"))
    with pytest.raises(transport.CodexAuthenticationError, match="existing API file"):
        transport.api_credentials()


def test_backup_profile_uses_only_suffixed_file_credentials(api_mode, monkeypatch):
    api_mode.write_text(api_mode.read_text() +
        'export OPENAI_API_KEY_1="backup-test-key"\n'
        'export OPENAI_BASE_URL_1="https://backup.invalid/v1"\n')
    monkeypatch.setenv("PARALLELWORLD_API_PROFILE", "1")
    before = dict(os.environ)
    assert transport.api_credentials() == ("backup-test-key", "https://backup.invalid/v1")
    from parallelworld_manipulation.planner.codex import CodexPlanner
    planner = CodexPlanner(output_dir=str(api_mode.parent), model="gpt-5.5",
                           dashboard_events=SimpleNamespace(emit=lambda _: None))
    config = planner._build_config("http://127.0.0.1:9999/mcp")
    assert config.env[transport.PROVIDER_ENV_KEY] == "backup-test-key"
    assert config_dict(config)["model_providers.parallelworld_manipulation_proxy.base_url"] == "https://backup.invalid/v1"
    assert config.env["GPT_IMAGE_API_KEY"] == "image-only-key"
    assert dict(os.environ) == before


def test_missing_backup_does_not_fall_back_to_primary(api_mode, monkeypatch):
    monkeypatch.setenv("PARALLELWORLD_API_PROFILE", "1")
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL_1", raising=False)
    with pytest.raises(transport.CodexAuthenticationError, match="OPENAI_API_KEY_1"):
        transport.api_credentials()


def test_invalid_backup_profile_fails(api_mode, monkeypatch):
    monkeypatch.setenv("PARALLELWORLD_API_PROFILE", "wrong")
    with pytest.raises(transport.CodexAuthenticationError, match="PARALLELWORLD_API_PROFILE"):
        transport.api_credentials()


def test_api_provider_works_with_real_runtime_and_empty_login(api_mode, monkeypatch):
    isolated = api_mode.parent / "empty-codex-home"
    isolated.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(isolated))
    with openai_codex.Codex(config=transport._config(cwd=str(api_mode.parent))) as client:
        config = client._client._request_raw("config/read", {"includeLayers": False})["config"]
        assert config["model_provider"] == transport.API_PROVIDER
        provider = config["model_providers"][transport.API_PROVIDER]
        assert provider["base_url"] == "http://127.0.0.1:9876/custom/api"
        assert provider["env_key"] == transport.PROVIDER_ENV_KEY
        assert provider["requires_openai_auth"] is False
        assert provider["request_max_retries"] == provider["stream_max_retries"] == 0
        thread = client.thread_start(model="gpt-5.5", model_provider=transport.API_PROVIDER,
            cwd=str(api_mode.parent), sandbox=openai_codex.Sandbox.read_only,
            approval_mode=openai_codex.ApprovalMode.deny_all, ephemeral=True,
            base_instructions="Return JSON only; no tools.", config=transport._disabled_mcp_config(client))
        assert thread.id
