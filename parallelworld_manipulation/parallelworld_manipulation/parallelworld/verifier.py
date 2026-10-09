"""Independent LLM verifier for ParallelWorld branch selection."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

_RAW_STRING_OUTPUT_ERROR = "'str' object has no attribute 'output'"
_LLM_REQUEST_ATTEMPTS = 5
logger = logging.getLogger(__name__)


@dataclass
class _AgentRequestBudget:
    """One operation's request budget, shared with any structured-output repair."""

    limit: int = _LLM_REQUEST_ATTEMPTS
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def _unwrap_agent_output(result: Any) -> Any:
    """Accept both PydanticAI run results and adapters returning output directly."""
    missing = object()
    output = getattr(result, "output", missing)
    if output is not missing:
        return output
    legacy_data = getattr(result, "data", missing)
    return result if legacy_data is missing else legacy_data


def _retryable_agent_error(exc: Exception) -> bool:
    """Retry transient transport/provider failures, not billing or input errors."""
    from openai import APIConnectionError, APITimeoutError
    from parallelworld_manipulation.codex_transport import CodexRequestError

    current: BaseException | None = exc
    seen: set[int] = set()
    # PydanticAI wraps SDK timeouts/connections in ModelAPIError; inspect its
    # explicit cause instead of treating every ModelAPIError as transient.
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(term in message for term in (
            "insufficient_user_quota", "insufficient_quota", "insufficient balance",
            "insufficient_balance", "insufficient funds", "credit_balance_exhausted",
            "spend_limit_exceeded", "usage_limit_exceeded", "余额不足", "额度不足", "欠费",
        )):
            return False
        if isinstance(current, AttributeError):
            return str(current) == _RAW_STRING_OUTPUT_ERROR
        if isinstance(current, CodexRequestError):
            return current.retryable
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 429} or 500 <= status < 600
        if isinstance(current, (APIConnectionError, APITimeoutError, TimeoutError, ConnectionError)):
            return True
        current = current.__cause__
    return False


def _run_agent_output(
    agent: Any, content: Any, *, attempts: int = _LLM_REQUEST_ATTEMPTS,
    operation: str = "parallelworld_llm",
    budget: _AgentRequestBudget | None = None,
) -> Any:
    """Try identical content at most five times, including the initial request."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    budget = budget if budget is not None else _AgentRequestBudget(limit=attempts)
    attempts = min(attempts, budget.remaining)
    if attempts < 1:
        raise RuntimeError("agent request budget exhausted")
    for attempt in range(attempts):
        budget.used += 1
        try:
            return _unwrap_agent_output(agent.run_sync(content))
        except Exception as exc:
            retry = attempt + 1 < attempts and _retryable_agent_error(exc)
            delay = (
                0.25 * (attempt + 1)
                if isinstance(exc, AttributeError)
                else min(2 ** attempt, 8)
            ) if retry else 0
            logger.warning(
                "parallelworld_api operation=%s attempt=%d/%d error=%s status=%s retry=%s delay_s=%s",
                operation, budget.used, budget.limit, type(exc).__name__,
                getattr(exc, "status_code", None), retry, delay,
            )
            if not retry:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


def _accept_raw_string_responses(model: Any) -> Any:
    """Turn a compatibility gateway's top-level JSON string into model text."""
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.openai import OpenAIResponsesModel

    if not isinstance(model, OpenAIResponsesModel):
        return model

    class _RawStringCompatibleOpenAIResponsesModel(OpenAIResponsesModel):
        def _process_response(
            self, response: Any, model_settings: Any,
            model_request_parameters: Any,
        ) -> ModelResponse:
            if isinstance(response, str):
                return ModelResponse(
                    parts=[TextPart(content=response, provider_name=self.system)],
                    model_name=self.model_name,
                    provider_name=self.system,
                    provider_url=self.base_url,
                    provider_details={"compat_raw_string_response": True},
                )
            return super()._process_response(
                response, model_settings, model_request_parameters,
            )

    # PydanticAI's public model inference does not expose a hook for response
    # normalization. Reuse the already configured provider so credentials,
    # custom base URL, profile, and model settings remain exactly unchanged.
    return _RawStringCompatibleOpenAIResponsesModel(
        model.model_name,
        provider=model._provider,
        profile=model.profile,
        settings=model.settings,
    )


def _normalize_keep(raw_keep: Any, candidate_ids: set[str]) -> list[str] | None:
    """Normalize common valid verifier variants without changing their meaning."""
    values = raw_keep
    if isinstance(raw_keep, str):
        stripped = raw_keep.strip()
        if stripped in candidate_ids:
            values = [stripped]
        else:
            try:
                decoded = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                return None
            if not isinstance(decoded, list):
                return None
            values = decoded
    if not isinstance(values, list):
        return None

    keep: list[str] = []
    for value in values:
        candidate_id = str(value)
        if candidate_id in candidate_ids and candidate_id not in keep:
            keep.append(candidate_id)
    return keep


_VERIFIER_SYSTEM_PROMPT = (
    "You are an independent verifier for a robotics planner. "
    "The visual inputs are not videos: they are chronological still "
    "images from the front view of each generated rollout. They are "
    "sampled once every ten generated frames, and the final frame is "
    "always appended when it is not already sampled. Each image is "
    "explicitly labeled with its candidate ID, original frame index, "
    "and front view. Each candidate also has one wrist_input image. "
    "The wrist_input is the real current close-range view and must be "
    "used to judge gripper contact and held-object geometry. "
    "The original task language supplied in the verifier request is authoritative. "
    "Select only from supplied candidates. Return exactly one JSON "
    "object and no Markdown. `keep` MUST be a JSON array of zero or "
    "more supplied candidate ID strings, even when retaining only one "
    "candidate; never return `keep` as a bare string, object, or null. "
    "`dominant_id` must be one supplied candidate ID or null, "
    "`continue_search` must be a JSON boolean, and `reason` must be a string."
)


def _make_agent(*, model: str, base_url: str | None = None,
                system_prompt: str | None = None, output_schema: dict | None = None):
    if model.startswith("codex:"):
        from parallelworld_manipulation.codex_transport import CodexJSONAgent, VERIFIER_OUTPUT_SCHEMA

        return CodexJSONAgent(
            model=model, system_prompt=system_prompt or _VERIFIER_SYSTEM_PROMPT,
            output_schema=output_schema or (VERIFIER_OUTPUT_SCHEMA if system_prompt is None else None),
        )

    import inspect

    from pydantic_ai import Agent
    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider, infer_provider_class

    api_key = os.environ.get("OPENAI_API_KEY")
    def provider_factory(provider_name: str):
        provider_cls = infer_provider_class(provider_name)
        params = inspect.signature(provider_cls.__init__).parameters
        kwargs = {}
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("CODEX_BASE_URL")
        if resolved_url and "base_url" in params:
            kwargs["base_url"] = resolved_url
        if api_key and "api_key" in params:
            kwargs["api_key"] = api_key
        return provider_cls(**kwargs) if kwargs else infer_provider(provider_name)
    resolved_model = infer_model(model, provider_factory=provider_factory)
    # The shared runner owns the retry budget: five means five requests, not
    # five outer calls multiplied by the SDK's default three HTTP attempts.
    from openai import AsyncOpenAI

    client = getattr(resolved_model, "client", None)
    if isinstance(client, AsyncOpenAI):
        client.max_retries = 0
    logger.info("parallelworld_api retry_policy=v1 attempts=5 sdk_retries=0 model=%s", model)
    resolved_model = _accept_raw_string_responses(resolved_model)
    return Agent(resolved_model, system_prompt=system_prompt or _VERIFIER_SYSTEM_PROMPT)


class VerifierAgent:
    """A separate, stateless model call used only for branch selection."""

    def __init__(self, *, model: str = "codex:gpt-5.6-sol",
                 base_url: str | None = None) -> None:
        self._agent = _make_agent(model=model, base_url=base_url)

    @staticmethod
    def _fallback(candidates: list[dict[str, Any]], reason: str) -> dict[str, Any]:
        """Keep the pipeline live when the verifier provider returns no usable output."""
        first = str(candidates[0]["id"]) if candidates else None
        return {
            "keep": [first] if first is not None else [],
            "dominant_id": first,
            "continue_search": False,
            "reason": f"verifier_fallback: {reason}",
        }

    def decide(self, *, depth: int, candidates: list[dict[str, Any]],
               frames: list[dict[str, Any]] | None = None,
               task_language: str = "") -> dict[str, Any]:
        frame_manifest = [
            {
                "candidate_id": str(frame.get("candidate_id", "candidate")),
                "frame_index": int(frame.get("frame_index", 0)),
                "view": str(frame.get("view", "unknown")),
            }
            for frame in (frames or [])
        ]
        prompt = json.dumps({"depth": depth, "task_language": task_language,
                             "candidates": candidates,
                             "image_count": len(frame_manifest),
                             "required_output_schema": {
                                 "keep": ["candidate_id"],
                                 "dominant_id": "candidate_id or null",
                                 "continue_search": False,
                                 "reason": "brief visual comparison",
                             },
                             "output_rules": (
                                 "keep must always be a JSON array, including for a "
                                 "single candidate; use only supplied candidate IDs"
                             ),
                             "visual_input": (
                                 "Front-view chronological still-image sequences, not "
                                 "videos, plus one wrist_input image per candidate. "
                                 "The wrist_input is the real current close-range view "
                                 "and must be used to judge gripper contact and held-object "
                                 "geometry. Every image is preceded by its candidate ID, "
                                 "original frame index, and view label."
                              ),
                             "frames": frame_manifest}, ensure_ascii=True)
        try:
            if frames:
                from PIL import Image
                from pydantic_ai import BinaryContent

                content: list[Any] = [prompt]
                for frame in frames:
                    candidate_id = str(frame.get("candidate_id", "candidate"))
                    frame_index = int(frame.get("frame_index", 0))
                    view = str(frame.get("view", "unknown"))
                    content.append(
                        f"Candidate ID: {candidate_id}; frame index: {frame_index}; "
                        f"view: {view}."
                    )
                    buf = io.BytesIO()
                    Image.fromarray(frame["image"]).convert("RGB").save(buf, format="PNG")
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("._")
                    content.append(BinaryContent(
                        data=buf.getvalue(), media_type="image/png",
                        identifier=(
                            f"{safe_id or 'candidate'}_frame_{frame_index:03d}_{view}.png"
                        ),
                        vendor_metadata={"detail": "high"},
                    ))
                request_content: Any = content
            else:
                request_content = prompt
            text = str(_run_agent_output(
                self._agent, request_content, operation="verifier",
            ) or "").strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return self._fallback(candidates, f"empty_or_non_json_output: {text[:200]}")
            decision = json.loads(match.group(0))
        except Exception as exc:
            return self._fallback(candidates, str(exc))
        ids = {str(candidate["id"]) for candidate in candidates}
        raw_keep = decision.get("keep", [])
        keep = _normalize_keep(raw_keep, ids)
        if keep is None:
            return self._fallback(
                candidates,
                f"keep has unsupported type or value: {type(raw_keep).__name__}",
            )
        dominant = decision.get("dominant_id")
        if dominant is not None and str(dominant) not in ids:
            dominant = None
        return {
            "keep": keep,
            "dominant_id": str(dominant) if dominant is not None else None,
            "continue_search": bool(decision.get("continue_search", False)),
            "reason": str(decision.get("reason", "")),
        }
