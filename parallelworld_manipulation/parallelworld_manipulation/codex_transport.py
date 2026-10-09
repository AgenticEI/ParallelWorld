"""Codex transport using api.sh credentials by default, with opt-in ChatGPT auth."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
# Codex deep-merges provider tables. A fresh ID avoids inheriting env_key,
# headers or a gateway URL from any pre-existing user/profile provider table.
CHATGPT_PROVIDER = "parallelworld_manipulation_chatgpt_" + uuid.uuid4().hex
API_PROVIDER = "parallelworld_manipulation_api_" + uuid.uuid4().hex
PROVIDER_ENV_KEY = "PARALLELWORLD_CODEX_PROVIDER_KEY"
_API_CONFIG_PATH = Path(__file__).resolve().parents[1] / "api.sh"

# Remove only in the Codex child environment, never from os.environ. The image
# service must retain its existing API key, base URL, model and retry settings.
_API_ENV_KEYS = {
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CODEX_BASE_URL",
    "PARALLELWORLD_CODEX_PROVIDER_KEY", "PARALLELWORLD_API_CONFIG", "GPT_IMAGE_API_KEY",
    "GPT_IMAGE_BASE_URL", "CODEX_ACCESS_TOKEN",
}


class CodexAuthenticationError(RuntimeError):
    """A configuration/login failure, not a transient inference failure."""


class CodexRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def codex_auth_mode() -> str:
    mode = os.environ.get("PARALLELWORLD_CODEX_AUTH", "api").strip().lower()
    if mode not in {"chatgpt", "api"}:
        raise ValueError("PARALLELWORLD_CODEX_AUTH must be chatgpt or api")
    return mode


def _api_profile_suffix() -> str:
    profile = os.environ.get("PARALLELWORLD_API_PROFILE", "default").strip()
    if profile not in {"default", "1"}:
        raise CodexAuthenticationError("PARALLELWORLD_API_PROFILE must be default or 1")
    return "_1" if profile == "1" else ""


def _read_parallelworld_manipulation_api_config(path: str | Path) -> tuple[str | None, str | None]:
    """Read literal exports from the selected file without executing shell code."""
    config = Path(path)
    try:
        text = config.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError:
        raise CodexAuthenticationError("Cannot read the configured parallelworld API file") from None
    suffix = _api_profile_suffix()
    key_name, url_name = "OPENAI_API_KEY" + suffix, "OPENAI_BASE_URL" + suffix
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        name, sep, raw = line[7:].partition("=")
        if not sep or name not in {key_name, url_name}:
            continue
        # Do not evaluate substitutions or accidentally treat them as credentials.
        if "$" in raw or "`" in raw:
            raise CodexAuthenticationError(f"{name} in the API file must be a literal value")
        try:
            parsed = shlex.split(raw, comments=True, posix=True)
        except ValueError:
            raise CodexAuthenticationError(f"Invalid {name} export in the API file") from None
        if len(parsed) != 1 or not parsed[0].strip():
            raise CodexAuthenticationError(f"Missing literal value for {name} in the API file")
        values[name] = parsed[0]
    return values.get(key_name), values.get(url_name)


def api_credentials() -> tuple[str, str]:
    """Resolve the same credentials for planner, enhancer, verifier and expansion."""
    path = Path(os.environ.get("PARALLELWORLD_API_CONFIG", str(_API_CONFIG_PATH)))
    if "PARALLELWORLD_API_CONFIG" in os.environ and not path.is_file():
        raise CodexAuthenticationError("PARALLELWORLD_API_CONFIG must identify an existing API file")
    file_key, file_url = _read_parallelworld_manipulation_api_config(path)
    suffix = _api_profile_suffix()
    if suffix:
        # An explicit backup selection must never silently borrow primary credentials.
        key = file_key or os.environ.get("OPENAI_API_KEY" + suffix)
        base_url = file_url or os.environ.get("OPENAI_BASE_URL" + suffix)
    else:
        key = file_key or os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = file_url or os.environ.get("CODEX_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not key or not key.strip() or not base_url or not base_url.strip():
        raise CodexAuthenticationError(
            f"PARALLELWORLD_API_CONFIG_REQUIRED: set OPENAI_API_KEY{suffix} and OPENAI_BASE_URL{suffix} in api.sh "
            "(or PARALLELWORLD_API_CONFIG); both are required for API authentication."
        )
    base_url = base_url.strip().rstrip("/")
    url = urlsplit(base_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise CodexAuthenticationError("API base URL must be an HTTP(S) endpoint without embedded credentials, query or fragment")
    # Preserve explicitly configured gateway paths, including /api and /v1.
    if not url.path:
        base_url += "/v1"
    return key, base_url


def api_environment(api_key: str, base_url: str) -> dict[str, str]:
    """Pass resolved credentials only to the child; leave image configuration intact."""
    env = dict(os.environ)
    env.update({PROVIDER_ENV_KEY: api_key, "OPENAI_API_KEY": api_key,
                "OPENAI_BASE_URL": base_url})
    return env


def api_overrides(base_url: str, *, bounded_retries: bool = False) -> list[str]:
    provider: dict[str, Any] = {
        "name": "parallelworld API", "base_url": base_url, "env_key": PROVIDER_ENV_KEY,
        "wire_api": "responses", "requires_openai_auth": False,
    }
    # Keep the isolated inference provider, but use the same transport retry
    # policy as the planner. The bounded mode is retained for callers that
    # explicitly need one SDK attempt; verifier/enhancer use the planner-like
    # mode below.
    provider.update({
        "request_max_retries": 0 if bounded_retries else 10,
        "stream_max_retries": 0 if bounded_retries else 10,
    })
    table = "{ " + ", ".join(f"{key} = {json.dumps(value)}" for key, value in provider.items()) + " }"
    return [f"model_provider={json.dumps(API_PROVIDER)}",
            f"model_providers.{API_PROVIDER}={table}"]


def chatgpt_environment() -> dict[str, str]:
    # CodexConfig.env is an overlay on os.environ, not a replacement. Explicit
    # empty values prevent SDK subprocess inheritance from restoring secrets.
    env = dict(os.environ)
    env.update({key: "" for key in _API_ENV_KEYS})
    return env


def chatgpt_overrides(*, bounded_retries: bool = False) -> list[str]:
    # Do not set forced_login_method: some runtimes clear incompatible cached
    # credentials. Instead inspect account/read and fail without logging out.
    # The pinned runtime forbids overriding reserved built-in provider IDs.
    # A dedicated provider uses managed OpenAI auth, with NO base_url/env_key:
    # Codex selects its official endpoint from the authenticated account type.
    provider: dict[str, Any] = {
        "name": "parallelworld ChatGPT login", "requires_openai_auth": True,
        "wire_api": "responses",
    }
    if bounded_retries:
        provider.update({
            "request_max_retries": 0,
            "stream_max_retries": 0,
        })
    # CLI -c values are TOML, not JSON objects. The fresh provider ID above
    # ensures this table has no lower-precedence gateway settings to merge.
    table = "{ " + ", ".join(f"{key} = {json.dumps(value)}" for key, value in provider.items()) + " }"
    return [f"model_provider={json.dumps(CHATGPT_PROVIDER)}",
            f"model_providers.{CHATGPT_PROVIDER}={table}"]


def require_chatgpt_account(account_response: Any) -> None:
    account = getattr(account_response, "account", None)
    if isinstance(account_response, dict):
        account = account_response.get("account")
    # Account is a generated Pydantic RootModel in the pinned SDK.
    account = getattr(account, "root", account)
    kind = account.get("type") if isinstance(account, dict) else getattr(account, "type", None)
    kind = getattr(kind, "value", kind)
    if kind != "chatgpt":
        raise CodexAuthenticationError(
            "CODEX_CHATGPT_LOGIN_REQUIRED: log in on this server with "
            "codex login --device-auth using the same CODEX_HOME; "
            "API-key login is not accepted for this pipeline."
        )


def _config(*, cwd: str, bounded_retries: bool = True, auth_mode: str | None = None):
    import openai_codex

    mode = auth_mode or codex_auth_mode()
    if mode == "api":
        api_key, base_url = api_credentials()
        env = api_environment(api_key, base_url)
        overrides = api_overrides(base_url, bounded_retries=bounded_retries)
    else:
        env = chatgpt_environment()
        overrides = chatgpt_overrides(bounded_retries=bounded_retries)
    options: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "config_overrides": tuple(overrides + [
            'web_search="disabled"',
            'features.shell_tool=false',
            'features.unified_exec=false',
            'features.apply_patch_freeform=false',
            'features.multi_agent=false',
            'features.image_generation=false',
            'mcp_servers={}',
        ]),
        "experimental_api": False,
    }
    if binary := os.environ.get("CODEX_BIN"):
        options["codex_bin"] = binary
    return openai_codex.CodexConfig(**options)


def check_chatgpt_login() -> None:
    """Local app-server account check only: no model turn or GPU startup."""
    import openai_codex

    with tempfile.TemporaryDirectory(prefix="parallelworld_manipulation_codex_auth_") as cwd:
        with openai_codex.Codex(config=_config(cwd=cwd, auth_mode="chatgpt")) as codex:
            require_chatgpt_account(codex.account(refresh_token=False))


def check_authentication() -> None:
    """Validate selected auth before services start; never send a model request."""
    if codex_auth_mode() == "api":
        api_credentials()
    else:
        check_chatgpt_login()


def _request_error(exc: Exception) -> CodexRequestError:
    import openai_codex

    message = str(exc)
    lowered = message.lower()
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        match = re.search(r"(?:status(?:_code)?[\s:=]*|http[\s:=]+)([45]\d\d)\b", lowered)
        status = int(match[1]) if match else None
    permanent = any(term in lowered for term in (
        "usage limit", "usage_limit", "insufficient", "quota", "credit balance",
        "billing", "not logged in", "unauthorized", "authentication", "模型已禁用",
    ))
    retryable = not permanent and (
        (status is not None and (status in {408, 409, 429} or 500 <= status < 600))
        or isinstance(exc, (openai_codex.TransportClosedError, openai_codex.ServerBusyError,
                            TimeoutError, ConnectionError))
        or any(term in lowered for term in (
            "stream disconnected", "stream_disconnected", "high demand", "overloaded",
            "connection reset", "timed out", "timeout occurred",
        ))
    )
    return CodexRequestError(message, status_code=status, retryable=retryable)


def _inputs(content: Any) -> list[Any]:
    import openai_codex
    from pydantic_ai import BinaryContent

    items = [content] if isinstance(content, str) else content
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(openai_codex.TextInput(item))
        elif isinstance(item, BinaryContent) and item.media_type.startswith("image/"):
            # Preserve every original byte, aspect ratio, label and list position.
            # The runtime controls model-side image detail; no local resizing.
            data = base64.b64encode(item.data).decode("ascii")
            result.append(openai_codex.ImageInput(f"data:{item.media_type};base64,{data}"))
        else:
            raise TypeError("Codex inference accepts labelled text and image BinaryContent only")
    return result


def _disabled_mcp_config(codex: Any) -> dict[str, Any]:
    # An empty mcp_servers table does NOT clear inherited entries in Codex.
    # Read effective local config and explicitly disable every inherited server
    # on the inference thread. config/read sends no model request.
    config = codex._client._request_raw("config/read", {"includeLayers": False})["config"]
    return {"mcp_servers": {name: {"enabled": False}
                            for name in config.get("mcp_servers", {})}}


class CodexJSONAgent:
    """One independent, read-only Codex turn per outer request attempt."""

    def __init__(self, *, model: str, system_prompt: str,
                 output_schema: dict[str, Any] | None = None) -> None:
        self.model = model.removeprefix("codex:")
        if not self.model or ":" in self.model:
            raise ValueError("Codex model must be a bare model ID or codex:<model>")
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        self._auth_mode = codex_auth_mode()
        if self._auth_mode == "api":
            api_credentials()
        logger.info("parallelworld_codex auth=%s model=%s attempts=5 runtime_retries=10", self._auth_mode, self.model)

    def run_sync(self, content: Any) -> Any:
        import openai_codex

        inputs = _inputs(content)
        # A clean cwd prevents project AGENTS.md and planner context from leaking
        # into verifier/enhancer. No parallelworld_manipulation MCP endpoint is attached here.
        with tempfile.TemporaryDirectory(prefix="parallelworld_manipulation_codex_inference_") as cwd:
            try:
                with openai_codex.Codex(
                    config=_config(cwd=cwd, auth_mode=self._auth_mode, bounded_retries=False)
                ) as codex:
                    if self._auth_mode == "chatgpt":
                        require_chatgpt_account(codex.account(refresh_token=False))
                    thread = codex.thread_start(
                        model=self.model,
                        model_provider=API_PROVIDER if self._auth_mode == "api" else CHATGPT_PROVIDER,
                        cwd=cwd,
                        approval_mode=openai_codex.ApprovalMode.deny_all,
                        sandbox=openai_codex.Sandbox.read_only, ephemeral=True,
                        config=_disabled_mcp_config(codex),
                        base_instructions=self.system_prompt,
                        developer_instructions=(
                            "This is an isolated robotics inference call, not a coding task. "
                            "Use only the supplied text and images. Do not call tools, read "
                            "files, execute commands, or control the robot. Return JSON only."
                        ),
                    )
                    response = thread.run(inputs, output_schema=self.output_schema)
                    output = response.final_response
                    if not isinstance(output, str) or not output.strip():
                        raise ValueError("Codex returned empty final JSON output")
                    return SimpleNamespace(output=output)
            except CodexAuthenticationError:
                raise
            except Exception as exc:
                raise _request_error(exc) from exc


VERIFIER_OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}},
        "dominant_id": {"type": ["string", "null"]},
        "continue_search": {"type": "boolean"}, "reason": {"type": "string"},
    },
    "required": ["keep", "dominant_id", "continue_search", "reason"],
}

_ENHANCER_FIELDS = [
    "candidate_id", "target_identity", "initial_state", "motion_phases",
    "contact_and_gripper", "object_response", "terminal_state",
    "candidate_contrast", "natural_instruction",
]
ENHANCER_OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["candidates"],
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {name: (
            {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}
            if name == "motion_phases" else {"type": "string"}
        ) for name in _ENHANCER_FIELDS},
        "required": _ENHANCER_FIELDS,
    }}},
}


if __name__ == "__main__":
    try:
        check_authentication()
    except CodexAuthenticationError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Codex {codex_auth_mode()} authentication configuration ready (no model request sent).")
