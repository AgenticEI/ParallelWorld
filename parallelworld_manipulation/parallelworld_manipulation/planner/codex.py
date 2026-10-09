"""Codex SDK planner.

A thin, SDK-first backend. ``solve()`` prepares artifacts, drives one Codex
SDK turn, and assembles a ``PlannerResult``. parallelworld tools are exposed via an
in-process HTTP MCP server (``HttpMcpServer``) so the Codex binary can reach
the shared toolkit without spawning a subprocess; this backend does not
register tools in process. Event rendering and stats live in a single
``_Recorder``. Compare with ``claude_code.py`` which uses the in-process
``create_sdk_mcp_server`` instead of an HTTP transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex

from parallelworld_manipulation.cli.tui import next_user_line
from parallelworld_manipulation.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from parallelworld_manipulation.dashboard.interaction import DashboardInteractionPort
from parallelworld_manipulation.dashboard.planner_control import DashboardPlannerControl
from parallelworld_manipulation.planner.base import PlannerResult, strip_mcp_prefix
from parallelworld_manipulation.planner.utils.http_mcp_server import HttpMcpServer
from parallelworld_manipulation.tools.toolkit import Toolkit
from parallelworld_manipulation.utils.config import get_repo_root
from parallelworld_manipulation.utils.logging import get_logger

logger = get_logger("codex")

PROVIDER_ID = "parallelworld_manipulation_proxy"
PROVIDER_ENV_KEY = "PARALLELWORLD_CODEX_PROVIDER_KEY"

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexPlanner:
    """Planner backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        timeout_s: int | None = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._model = model or os.environ.get("CODEX_MODEL") or "gpt-5.6-sol"
        from parallelworld_manipulation.codex_transport import api_credentials, codex_auth_mode

        self._auth_mode = codex_auth_mode()
        self._base_url = self._api_key = None
        if self._auth_mode == "api":
            self._api_key, self._base_url = api_credentials()
        self._dashboard_events = dashboard_events
        self._turn_options = {
            "approval_mode": openai_codex.ApprovalMode.deny_all,
            "cwd": self._repo_root,
            "model": self._model,
            "sandbox": openai_codex.Sandbox.full_access,
        }

    def expand_parallel_routes(self, *, route: dict, limit: int) -> list[dict]:
        """Generate next-step candidates through an isolated Codex API turn."""
        import io

        from PIL import Image
        from pydantic_ai import BinaryContent

        from parallelworld_manipulation.codex_transport import CodexJSONAgent
        from parallelworld_manipulation.parallelworld.verifier import _run_agent_output

        model = getattr(self, "_model", None) or os.environ.get("CODEX_MODEL") or os.environ.get("MODEL", "gpt-5.6-sol")
        context = {k: v for k, v in route.items() if not k.startswith("_")}
        prompt = (f"Expand this retained LIBERO route. Return ONLY a JSON array with at most {int(limit)} "
                  "next physical primitive candidates; each object must contain id, primitive, arguments, rationale, "
                  "and visual_action. visual_action must contain a grounded subject, target and distinguishing "
                  "qualifier, relative motion, gripper state, contact behavior, and intended visible outcome, using "
                  "ordinary visual language only. Keep coordinates, primitive names, and control parameters out of "
                  "visual_action. "
                  "Candidates may use different primitives and must be genuinely diverse, not merely parameter "
                  "variants of the parent primitive. "
                  "Use the attached imagined front image as visual evidence. Do not return markdown. Context: " + json.dumps(context, default=str))
        content: list[Any] = [prompt]
        image = route.get("_imagined_front")
        if image is not None:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            content.extend([
                "Parent candidate ID: " + str(route.get("id", "parent")) + "; imagined front.",
                BinaryContent(data=buf.getvalue(), media_type="image/png"),
            ])
        agent = CodexJSONAgent(
            model=model,
            system_prompt="Generate grounded robot action candidates from the supplied context and image. Return only a JSON array.",
        )
        text = str(_run_agent_output(agent, content, operation="planner_expansion") or "[]")
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        out = json.loads(text)
        if not isinstance(out, list):
            raise RuntimeError("Planner expansion returned non-list output")
        return out[:int(limit)]

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run one or more Codex SDK turns for the given prompt."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        if dashboard_interaction is not None:
            return asyncio.run(
                self._solve_dashboard(
                    prompt=prompt,
                    initial_user_text=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    interaction=dashboard_interaction,
                )
            )
        output_path, raw_stream_path, last_message_path = self._output_paths()
        recorder = _Recorder(
            max_turns=max_turns,
            dashboard_events=self._dashboard_events,
        )
        state: dict[str, Any] = {}

        # Start the in-thread MCP HTTP server so Codex can reach the
        # shared toolkit without spawning a subprocess.
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        logger.info("mcp http endpoint: %s", mcp_url)

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        timeout_label = f"{self._timeout_s}s" if self._timeout_s is not None else "disabled"
        logger.info("invoking Codex SDK model %s (timeout=%s)", model_desc, timeout_label)

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                output_path,
                raw_stream_path,
                last_message_path,
                recorder,
                state,
                mcp_url,
                input_queue,
            ),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        error: str | None = None
        try:
            wait_started = time.monotonic()
            finish_timed_out = False
            while worker.is_alive():
                now = time.monotonic()
                if (
                    recorder.finish_completed_at is not None
                    and now - recorder.finish_completed_at >= 120.0
                ):
                    finish_timed_out = True
                    break
                if self._timeout_s is not None and now - wait_started >= self._timeout_s:
                    break
                worker.join(timeout=0.25)

            if finish_timed_out:
                message = "finish completed; stopped waiting for Codex after 120s"
                logger.info(message)
                with open(output_path, "a") as out_f:
                    out_f.write(f"\n[codex-planner] {message}\n")
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "finish_timeout", "message": message})
                # SDK interrupt/close may itself wait on a stalled connection.
                # Bound cleanup so the caller can still save episode artifacts.
                threading.Thread(
                    target=_interrupt, args=(state,), daemon=True,
                    name="codex-finish-cleanup",
                ).start()
                worker.join(timeout=15)
            elif worker.is_alive() and self._timeout_s is not None:
                error = f"Codex SDK timed out after {self._timeout_s}s"
                _interrupt(state)
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                worker.join(timeout=15)
            elif "error" in state:
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            mcp_server.stop()

        elapsed = time.time() - started
        text = state.get("text", "") or output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _output_paths(self) -> tuple[Path, Path, Path]:
        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_sdk_task_", delete=False
            ) as file_obj:
                output_path = Path(file_obj.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return (
            output_path,
            output_path.with_suffix(output_path.suffix + ".stream.jsonl"),
            output_path.with_suffix(output_path.suffix + ".last"),
        )

    def _run_session(
        self,
        prompt: str,
        output_path: Path,
        raw_stream_path: Path,
        last_message_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        mcp_url: str,
        input_queue: "queue.Queue[str | None] | None" = None,
    ) -> None:
        try:
            chunks: list[str] = []
            with openai_codex.Codex(config=self._build_config(mcp_url)) as codex:
                state["codex"] = codex
                if self._auth_mode == "chatgpt":
                    from parallelworld_manipulation.codex_transport import require_chatgpt_account
                    require_chatgpt_account(codex.account(refresh_token=False))
                thread = codex.thread_start(**self._turn_options)
                state["thread"] = thread

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    write_lock = threading.Lock()

                    turn = thread.turn(prompt, **self._turn_options)
                    state["turn"] = turn

                    stop_steer: threading.Event | None = None
                    if input_queue is not None:
                        stop_steer = threading.Event()

                        def _steer() -> None:
                            while True:
                                nxt = next_user_line(input_queue)
                                if stop_steer.is_set():
                                    return
                                if nxt is None:
                                    try:
                                        turn.interrupt()
                                    except Exception:
                                        pass
                                    return
                                rendered = f"\n[user] {nxt}\n"
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                                try:
                                    turn.steer(nxt)
                                except Exception as e:
                                    rendered = f"\n[codex-planner] steer failed: {e}\n"
                                    with write_lock:
                                        chunks.append(rendered)
                                        out_f.write(rendered)
                                        out_f.flush()
                                    logger.info(rendered.strip())
                                    return

                        threading.Thread(
                            target=_steer,
                            name="codex-steer",
                            daemon=True,
                        ).start()

                    try:
                        for event in turn.stream():
                            _write_jsonl(raw_f, _message_to_json(event))
                            if rendered := recorder.observe(event):
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                    finally:
                        if stop_steer is not None:
                            stop_steer.set()
                            if input_queue is not None:
                                input_queue.put(None)

            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                last_message_path.write_text(recorder.final_response)
        except Exception as e:
            state["error"] = e

    async def _solve_dashboard(
        self,
        *,
        prompt: str,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort,
    ) -> PlannerResult:
        """Run a controllable sequence of turns on one Codex thread."""
        output_path, raw_stream_path, last_message_path = self._output_paths()
        recorder = _Recorder(
            max_turns=max_turns, dashboard_events=self._dashboard_events
        )
        chunks: list[str] = []
        error: str | None = None
        started = time.time()

        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        try:
            with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:

                def emit_event(event: Any) -> None:
                    _write_jsonl(raw_f, _message_to_json(event))
                    rendered = recorder.observe(event)
                    if rendered:
                        chunks.append(rendered)
                        out_f.write(rendered)
                        out_f.flush()

                def emit_user(text: str, *, initial: bool = False) -> None:
                    display = (
                        "[initial task instructions submitted]" if initial else text
                    )
                    rendered = f"\n[user] {display}\n"
                    chunks.append(rendered)
                    out_f.write(rendered)
                    out_f.flush()
                    self._dashboard_events.emit(
                        TranscriptEvent(
                            {"type": "initial_prompt"}
                            if initial
                            else {"type": "user", "text": text}
                        )
                    )

                control = DashboardPlannerControl(
                    interaction=interaction,
                    cancel_active_and_wait=toolkit.cancel_active_and_wait,
                    emit_user=emit_user,
                    emit_initial_user=lambda: emit_user(
                        initial_user_text, initial=True
                    ),
                )
                session = _CodexDashboardSession(
                    config=self._build_config(mcp_url),
                    auth_mode=self._auth_mode,
                    options=self._turn_options,
                    recorder=recorder,
                    emit_event=emit_event,
                    control=control,
                )
                try:
                    if self._timeout_s is None:
                        await session.run(prompt)
                    else:
                        await asyncio.wait_for(session.run(prompt), timeout=self._timeout_s)
                except asyncio.TimeoutError:
                    error = f"Codex SDK timed out after {self._timeout_s}s"
                    control.end()
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    control.end()
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                finally:
                    try:
                        await control.cancel_active_toolkit()
                    except Exception as exc:
                        cleanup_error = (
                            "Codex toolkit cancellation failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        logger.warning(cleanup_error)
                        error = error or cleanup_error
                    await session.close()
        finally:
            mcp_server.stop()

        if recorder.final_response is not None:
            last_message_path.write_text(recorder.final_response)
        text = "".join(chunks)
        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(time.time() - started, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error or session.error or recorder.error,
        )

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        from parallelworld_manipulation.codex_transport import (
            api_environment,
            chatgpt_environment,
            chatgpt_overrides,
        )

        if self._auth_mode == "chatgpt":
            env = chatgpt_environment()
            auth_overrides = chatgpt_overrides()
        else:
            env = api_environment(self._api_key, self._base_url)
            auth_overrides = []
        kwargs: dict[str, Any] = {
            "config_overrides": tuple(
                _codex_mcp_config_overrides(
                    mcp_url=mcp_url,
                    base_url=self._base_url if self._auth_mode == "api" else None,
                )
            ) + tuple(auth_overrides),
            "cwd": self._repo_root,
            "env": env,
            # Disable experimental API features (namespace tools,
            # web_search, image_generation).  This forces the binary to
            # convert namespace MCP tools to function tools internally
            # while preserving its own name→namespace mapping so that
            # ``function_call`` responses can be routed through MCP.
            "experimental_api": False,
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            kwargs["codex_bin"] = codex_bin
        return openai_codex.CodexConfig(**kwargs)


class _CodexDashboardSession:
    """Own one Codex thread and its current interruptible turn."""

    def __init__(
        self,
        *,
        config: Any,
        options: dict[str, Any],
        recorder: "_Recorder",
        emit_event,
        control: DashboardPlannerControl,
        auth_mode: str = "chatgpt",
    ) -> None:
        self._config = config
        self._options = options
        self._recorder = recorder
        self._emit_event = emit_event
        self._control = control
        self._auth_mode = auth_mode
        self._codex: Any | None = None
        self._thread: Any | None = None
        self._turn: Any | None = None
        self._turn_done: asyncio.Event | None = None
        self._turn_task: asyncio.Task[Any] | None = None
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        self._codex = openai_codex.AsyncCodex(self._config)
        if self._auth_mode == "chatgpt":
            from parallelworld_manipulation.codex_transport import require_chatgpt_account
            require_chatgpt_account(await self._codex.account(refresh_token=False))
        self._thread = await self._codex.thread_start(**self._options)
        await self.submit(prompt)
        await self._control.start()
        await self._control.run(self)

    async def submit(self, text: str) -> int:
        if self._closing or self._thread is None:
            raise RuntimeError("Codex conversation is closed")
        if self._turn is not None:
            await self._turn.steer(text)
            return 0
        self._turn = await self._thread.turn(text, **self._options)
        self._turn_done = asyncio.Event()
        self._turn_task = asyncio.create_task(
            self._consume_turn(self._turn, self._turn_done)
        )
        return 1

    async def interrupt(self) -> int:
        turn = self._turn
        done = self._turn_done
        if turn is None or done is None:
            return 0
        await turn.interrupt()
        try:
            await asyncio.wait_for(done.wait(), timeout=15)
        except asyncio.TimeoutError:
            if self._turn_task is not None:
                self._turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._turn_task
            self._turn = None
            self._turn_done = None
            self._turn_task = None
            return 1
        # ``_consume_turn`` reports the matching completed turn boundary.
        return 0

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        turn = self._turn
        done = self._turn_done
        if turn is not None and done is not None:
            with contextlib.suppress(Exception):
                await turn.interrupt()
            try:
                await asyncio.wait_for(done.wait(), timeout=15)
            except asyncio.TimeoutError:
                if self._turn_task is not None:
                    self._turn_task.cancel()
        if self._turn_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        if self._codex is not None:
            with contextlib.suppress(Exception):
                await self._codex.close()

    async def _consume_turn(self, turn: Any, done: asyncio.Event) -> None:
        limit_reached = False
        try:
            async for event in turn.stream():
                self._emit_event(event)
                method = str(_get(event, "method", ""))
                payload = _get(event, "payload")

                if (
                    method == "item/completed"
                    and _is_tool_item(_get(payload, "item"))
                    and self._recorder.finish_result is None
                    and self._recorder.turns < self._recorder.max_turns
                ):
                    await self._control.tool_completed(self)

                if (
                    method != "turn/completed"
                    and not limit_reached
                    and self._recorder.finish_result is None
                    and self._recorder.turns >= self._recorder.max_turns
                ):
                    limit_reached = True
                    await turn.interrupt()

                if method != "turn/completed":
                    continue
                status = _status(_get(payload, "turn"))
                self._turn = None
                self._turn_done = None
                done.set()
                if self._closing:
                    return
                if status == "failed":
                    self.error = self._recorder.error or "Codex turn failed"
                if (
                    self.error is not None
                    or self._recorder.finish_result is not None
                    or self._recorder.turns >= self._recorder.max_turns
                ):
                    self._control.end()
                else:
                    await self._control.complete(self)
                return
            raise RuntimeError("Codex turn stream ended without turn/completed")
        except Exception as exc:
            self._turn = None
            self._turn_done = None
            done.set()
            self.error = f"{type(exc).__name__}: {exc}"
            if not self._closing:
                self._control.end()


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    dashboard_events: DashboardEventSink
    turns: int = 0
    tool_calls: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    final_response: str | None = None
    finish_result: dict[str, Any] | None = None
    finish_completed_at: float | None = None
    error: str | None = None

    def stats(self) -> dict[str, int]:
        return {"turns_used": self.turns, "tool_calls": self.tool_calls, **self.usage}

    def observe(self, event: Any) -> str:
        method = str(_get(event, "method", ""))
        payload = _get(event, "payload")

        if method in {"thread/started", "turn/started"}:
            return f"[codex-system] {method}\n"
        if method == "item/completed":
            return self._render_item(_get(payload, "item"))
        if method == "thread/tokenUsage/updated":
            self._set_usage(_get(payload, "token_usage"))
            return ""
        if method == "turn/completed":
            return self._render_turn_completed(_get(payload, "turn"))
        if "requestApproval" in method:
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            self.error = _short_json(_jsonable(payload), limit=500)
            return f"[codex-error] {self.error}\n"
        return ""

    # -- per-item handlers -------------------------------------------------

    def _render_item(self, item: Any) -> str:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))

        if item_type == "userMessage":
            text = _extract_text(_get(item, "content"))
            return f"\n[codex][user] {text}\n" if text else ""

        if item_type in {"hookPrompt", "plan"}:
            return ""

        if item_type == "agentMessage":
            text = str(_get(item, "text", "")).strip()
            if not text:
                return ""
            self.final_response = text
            self.turns += 1
            self.dashboard_events.emit(TranscriptEvent({"type": "text", "text": text}))
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if item_type == "reasoning":
            text = _extract_text(_get(item, "summary") or _get(item, "content"))
            if text:
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "thinking", "text": text})
                )
            return f"[codex-reasoning] {text}\n" if text else ""

        if _is_tool_item(item):
            self.tool_calls += 1
            if item_type in {"mcpToolCall", "dynamicToolCall"}:
                name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
                self._maybe_capture_finish(name, item)
            elif item_type == "commandExecution":
                name = str(_get(item, "command", item_type))
            else:
                name = "fileChange"
            payload = _summarise_item(item)
            data = _jsonable(item)
            args = data.get("arguments", {}) if isinstance(data, dict) else {}
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_call", "tool": name, "args": args})
            )
            self.dashboard_events.emit(
                TranscriptEvent(
                    {"type": "tool_result", "tool": name, "result": payload}
                )
            )
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        status = str(_get(_get(turn, "status"), "value", _get(turn, "status", "")))
        duration_ms = _get(turn, "duration_ms")
        if error := _get(turn, "error"):
            self.error = str(_get(error, "message", str(error)))

        parts = ["[codex-result]", status]
        if duration_ms is not None:
            parts.append(f"duration={float(duration_ms) / 1000:.1f}s")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cached={self.usage['total_cached_input_tokens']} "
            f"out={self.usage['total_output_tokens']} "
            f"reasoning={self.usage['total_reasoning_output_tokens']} "
            f"tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- helpers -----------------------------------------------------------

    def _set_usage(self, usage: Any) -> None:
        if usage is None:
            return
        total = _get(usage, "total", usage)
        self.usage = {
            "total_input_tokens": _int_attr(total, "input_tokens"),
            "total_cached_input_tokens": _int_attr(total, "cached_input_tokens"),
            "total_output_tokens": _int_attr(total, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                total, "reasoning_output_tokens"
            ),
        }
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            return
        if name.lower() != "finish":
            return
        status = _status(item)
        if status and status != "completed":
            return
        if _get(item, "error") not in (None, ""):
            return
        data = _jsonable(item)
        args = data.get("arguments") if isinstance(data, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}
            self.finish_completed_at = time.monotonic()


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    mcp_url: str,
    base_url: str | None,
) -> list[str]:
    tool_timeout_s = float(os.environ.get("PARALLELWORLD_MCP_TOOL_TIMEOUT_S", "14400"))
    if tool_timeout_s <= 0:
        raise ValueError("PARALLELWORLD_MCP_TOOL_TIMEOUT_S must be greater than zero")
    config: list[tuple[str, Any]] = [
        ("mcp_servers.parallelworld_manipulation.url", mcp_url),
        ("mcp_servers.parallelworld_manipulation.tool_timeout_sec", tool_timeout_s),
    ]
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith(("/v1", "/api")):
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
                (f"model_providers.{PROVIDER_ID}.stream_max_retries", 10),
                (f"model_providers.{PROVIDER_ID}.request_max_retries", 10),
            ]
        )
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _interrupt(state: dict[str, Any]) -> None:
    if (turn := state.get("turn")) is not None:
        try:
            turn.interrupt()
        except Exception:
            pass
    if (codex := state.get("codex")) is not None:
        try:
            codex.close()
        except Exception:
            pass


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _message_to_json(message: Any) -> dict[str, Any]:
    return {
        "method": _get(message, "method", ""),
        "payload": _jsonable(_get(message, "payload")),
    }


def _jsonable(value: Any) -> Any:
    value = _unwrap(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _unwrap(value: Any) -> Any:
    return getattr(value, "root", value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_tool_item(item: Any) -> bool:
    return str(_get(_unwrap(item), "type", "")) in {
        "mcpToolCall",
        "dynamicToolCall",
        "commandExecution",
        "fileChange",
    }


def _status(item: Any) -> str:
    value = _get(item, "status", "")
    return str(getattr(value, "value", value))


def _summarise_item(item: Any) -> dict[str, Any]:
    data = _jsonable(item)
    if not isinstance(data, dict):
        return {"size": _payload_size(data)}

    summary: dict[str, Any] = {}
    for key in ("path", "file_path", "filename", "status", "state", "exit_code"):
        value = data.get(key)
        if value not in (None, ""):
            summary[key] = value
    if command := (data.get("command") or data.get("cmd")):
        command_text = str(command)
        if len(command_text) > 200:
            command_text = command_text[:200] + f"...(+{len(command_text) - 200})"
        summary["command"] = command_text
    for key in ("content", "text", "output", "stdout", "stderr", "result"):
        if key in data and data[key] not in (None, ""):
            summary[f"{key}_size"] = _payload_size(data[key])

    if not summary:
        summary["keys"] = sorted(
            key for key in data if key not in {"content", "text", "output"}
        )
    return summary


def _extract_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        text = value.strip()
        if "data:image" in text or (
            "base64" in text and ("image" in text or "iVBOR" in text)
        ):
            return "<image omitted>"
        return text
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _payload_size(value: Any) -> int:
    return len(str(value or ""))


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _int_attr(value: Any, key: str) -> int:
    return int(_get(value, key, 0) or 0)
