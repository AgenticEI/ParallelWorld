"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import io
import re
from functools import partial
from typing import Any

import numpy as np
from PIL import Image
from html import escape
import json

from robots.libero import tools as libero_tools
from parallelworld_manipulation.dashboard.events import DashboardEventSink
from parallelworld_manipulation.tools.state import EnvState
from parallelworld_manipulation.tools.toolkit import Toolkit, ToolResult
from parallelworld_manipulation.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    _FRAME_ARTIFACTS = {
        "camera": "agentview.png",
        "wrist": "wrist.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        runtime_kwargs = dict(primitives_kwargs)
        parallelworld = runtime_kwargs.pop("_parallelworld", None)
        self.init_primitives_clean(primitives_kwargs=runtime_kwargs)
        self._register_libero_tools()
        self._parallelworld_gate = None
        if parallelworld is not None:
            self._register_parallelworld(parallelworld)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        # These read-only handlers need the run's EnvState bound in. Every
        # other spec binds to a primitive-driver method and captures state by
        # default unless that method is explicitly marked @readonly.
        state_handlers = {
            "view_env_state": partial(
                libero_tools.view_env_state, state=self._state
            ),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(libero_tools.back_project, state=self._state),
            "segment": partial(self._primitives.segment, state=self._state),
        }
        for spec in libero_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue  # spec without a backing primitive method
            self.add_tool(name, spec, handler)

    def _register_parallelworld(self, config: dict[str, Any]) -> None:
        from robots.libero.parallelworld import PARALLELWORLD_TOOL_SPEC
        from parallelworld_manipulation.parallelworld import ParallelWorldExplorer

        explorer_config = dict(config)
        explorer = ParallelWorldExplorer(primitives=self._primitives, **explorer_config)
        # Internal runtime phase; intentionally not Planner-visible.
        self._parallelworld_gate = True
        self._parallelworld_hidden_tools = {
            "move_to", "move_pose", "navigate_to", "move_base",
            "pi0_pick", "pi0_doubled", "release", "set_gripper",
            "rotate_wrist", "rotate_pitch",
        }
        self._parallelworld_allowed_primitives = set(
            self._parallelworld_hidden_tools
        )
        self._parallelworld_explorer = explorer
        self._parallelworld_planner = None
        self._parallelworld_trace: list[dict[str, Any]] = []
        self._parallelworld_timing_summary: list[dict[str, Any]] = []
        self._pending_parallel_execution: dict[str, Any] | None = None
        # Expose one route-search tool instead of primitive-specific tools. This
        # prevents the planner from anchoring on one primitive and changing only args.
        self.add_tool(
            PARALLELWORLD_TOOL_SPEC["name"],
            PARALLELWORLD_TOOL_SPEC,
            self._parallelworld_decide_tool,
        )

    def get_tools_spec(self) -> list[dict[str, Any]]:
        specs = super().get_tools_spec()
        hidden = getattr(self, "_parallelworld_hidden_tools", set())
        return [spec for spec in specs if spec.get("name") not in hidden]

    def _parallelworld_decide_tool(self, candidate_routes, depth: int = 1) -> dict[str, Any]:
        return {"error": "parallelworld_decide is dispatched by execute_tool"}

    def set_parallelworld_planner(self, planner: Any) -> None:
        """Inject the Planner expansion callback used after Verifier top-k."""
        self._parallelworld_planner = planner

    def _parallelworld_select(self, name: str, input_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Evaluate the complete Planner route set before one real action."""
        explorer = getattr(self, "_parallelworld_explorer", None)
        if explorer is None:
            raise RuntimeError("ParallelWorld explorer is not initialized")
        routes = input_dict.pop("candidate_routes", None)
        if not isinstance(routes, list) or not routes:
            raise ValueError("candidate_routes is required and must be non-empty")
        candidates = []
        for index, route in enumerate(routes[:explorer.N_MAX]):
            if not isinstance(route, dict) or not route.get("primitive") or route.get("arguments") is None:
                raise ValueError(f"candidate_routes[{index}] must contain primitive and arguments")
            if route["primitive"] not in self._parallelworld_allowed_primitives:
                raise ValueError(
                    f"candidate_routes[{index}].primitive must be one of "
                    f"{sorted(self._parallelworld_allowed_primitives)}"
                )
            if not isinstance(route["arguments"], dict):
                raise ValueError(f"candidate_routes[{index}].arguments must be an object")
            visual_action = route.get("visual_action")
            if visual_action is not None and not isinstance(visual_action, dict):
                raise ValueError(
                    f"candidate_routes[{index}].visual_action must be an object"
                )
            item = dict(route)
            item.setdefault("id", f"planner_route_{index}")
            if not isinstance(item.get("next_candidates"), list):
                item["next_candidates"] = None
            candidates.append(item)
        # A single planner route has no branch-selection problem. Execute it
        # directly so it does not incur prompt enhancement, image generation,
        # or verifier latency and does not create misleading ParallelWorld
        # artifacts. ParallelWorld remains active only for genuine alternatives.
        if len(candidates) == 1:
            selected = candidates[0]
            self._pending_parallel_execution = None
            return str(selected["primitive"]), dict(selected["arguments"])
        def expand(route, limit):
            planner = self._parallelworld_planner
            if planner is None or not hasattr(planner, "expand_parallel_routes"):
                raise RuntimeError(
                    "Planner must generate next_candidates after Verifier top-k"
                )
            children = planner.expand_parallel_routes(route=route, limit=limit)
            if not isinstance(children, list) or not children:
                raise RuntimeError("Planner returned no next_candidates")
            # Expansion is another planner boundary: validate every child before
            # it reaches world-model prediction or verifier selection. Otherwise
            # an invented primitive can survive as a candidate and fail only at
            # execution time after the verifier has selected it.
            valid = []
            for child in children[:limit]:
                if not isinstance(child, dict):
                    continue
                primitive = child.get("primitive")
                arguments = child.get("arguments")
                if primitive not in self._parallelworld_allowed_primitives:
                    logger.warning("dropping unsupported expanded primitive: %r", primitive)
                    continue
                if not isinstance(arguments, dict):
                    logger.warning("dropping expanded route with invalid arguments: %r", child)
                    continue
                valid.append(child)
            if not valid:
                raise RuntimeError("Planner expansion returned no supported routes")
            return valid[:limit]
        # The next state record is created by the real primitive after this
        # method returns, so reserve its index for the offline timeline now.
        latest_step = self._state.latest_step
        action_step = 0 if latest_step is None else latest_step + 1
        decision = explorer.search(candidates, next_candidate_generator=expand)
        self._save_parallelworld_artifacts(
            decision,
            action_step=action_step,
            action_name=name,
            requested_candidates=candidates,
        )
        selected = decision.get("_selected_candidate")
        if not isinstance(selected, dict):
            raise RuntimeError("No valid route available for execution")
        search_id = int(getattr(explorer, "_search_counter", 0))
        calibration_images = getattr(explorer, "_last_real_images", None)
        if calibration_images is None:
            initial_front = np.ascontiguousarray(
                np.asarray(
                    self._primitives._last_obs["main_images"], dtype=np.uint8
                )[:, ::-1]
            )
            initial_wrist = np.ascontiguousarray(
                np.asarray(
                    self._primitives._last_obs["wrist_images"], dtype=np.uint8
                )[:, ::-1]
            )
        else:
            initial_front = np.asarray(calibration_images[0], dtype=np.uint8).copy()
            initial_wrist = np.asarray(calibration_images[1], dtype=np.uint8).copy()
        self._pending_parallel_execution = {
            "step_dir": (
                f"parallelworld/step_{action_step:03d}"
                f"__trigger_{self._safe_name(name)}"
                f"__selected_{self._safe_name(str(selected['primitive']))}"
            ),
            "search_id": search_id,
            "requested_action": name,
            "selected_candidate": self._parallelworld_candidate_summary(selected),
            "initial_front": initial_front,
            "initial_wrist": initial_wrist,
            "parallelworld_timing": decision.get("_search_timing", {}),
        }
        return str(selected["primitive"]), dict(selected["arguments"])

    @staticmethod
    def _parallelworld_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(candidate.get("id", "")),
            "parent_id": candidate.get("parent_id") or candidate.get("_parent_id"),
            "primitive": candidate.get("primitive"),
            "arguments": candidate.get("arguments", {}),
            "rationale": candidate.get("rationale", ""),
            "visual_action": candidate.get("visual_action"),
        }

    def _save_parallelworld_artifacts(
        self,
        decision: dict[str, Any],
        *,
        action_step: int,
        action_name: str,
        requested_candidates: list[dict[str, Any]],
    ) -> None:
        """Persist a readable step/search/candidate timeline."""
        explorer = self._parallelworld_explorer
        history = getattr(explorer, "_last_search_history", [])
        search_id = int(history[0]["search_id"]) if history else int(
            getattr(explorer, "_search_counter", 0)
        )
        selected = decision.get("_selected_candidate")
        selected_action = (
            str(selected.get("primitive"))
            if isinstance(selected, dict) and selected.get("primitive")
            else "none"
        )
        step_dir = (
            f"parallelworld/step_{action_step:03d}"
            f"__trigger_{self._safe_name(action_name)}"
            f"__selected_{self._safe_name(selected_action)}"
        )
        if not history:
            failure = {
                "step": action_step,
                "requested_action": action_name,
                "selected_action": selected_action,
                "search_id": search_id,
                "status": "failed",
                "error": decision.get("error", "ParallelWorld search produced no completed depth"),
                "requested_candidates": [
                    self._parallelworld_candidate_summary(candidate)
                    for candidate in requested_candidates
                ],
                "selected_candidate": (
                    self._parallelworld_candidate_summary(decision["_selected_candidate"])
                    if isinstance(decision.get("_selected_candidate"), dict) else None
                ),
            }
            self._state.save(f"{step_dir}/failure.json", failure, step=None)
            self._append_parallelworld_index(
                action_step=action_step,
                action_name=action_name,
                search_id=search_id,
                step_dir=step_dir,
                decision=decision,
                levels=[],
            )
            self._save_parallelworld_viewer()
            return

        levels = []
        for item in history:
            depth = int(item["depth"])
            depth_dir = f"{step_dir}/search_{search_id:03d}/depth_{depth:02d}"
            summary = item["summary"]
            # Internal fields contain raw image/video bytes and must never enter
            # EnvState JSON artifacts or the persistent trace.
            level = {key: value for key, value in summary.items() if not key.startswith("_")}
            level["depth"] = depth
            level["search_id"] = search_id
            level["timing"] = summary.get("_timing", {})
            artifacts = []
            for view in ("real_front", "real_wrist"):
                data = item.get(view)
                if data is not None:
                    filename = f"{depth_dir}/input_{view}.png"
                    self._state.save(filename, data, step=None)
                    artifacts.append(filename)
            for view in ("front", "wrist"):
                data = item.get(view)
                if data:
                    image = Image.open(io.BytesIO(data)).convert("RGB")
                    filename = f"{depth_dir}/candidates_{view}.png"
                    self._state.save(filename, image, step=None)
                    artifacts.append(filename)

            verifier = level.get("verifier") or {}
            keep = {str(value) for value in verifier.get("keep", [])}
            dominant = verifier.get("dominant_id")
            branch_by_id = {
                str(branch.get("id")): branch
                for branch in level.get("branches", [])
                if isinstance(branch, dict) and branch.get("id") is not None
            }
            videos_by_id = {
                str(video.get("id")): video
                for video in item.get("videos", [])
            }
            candidate_dirs = []
            for candidate_index, candidate in enumerate(
                level.get("fallback_candidates", [])
            ):
                candidate_id = str(candidate.get("id", candidate_index))
                # The fallback list is the planner's original route set. Merge
                # the runtime branch record so prompt/timing fields produced by
                # the enhancer and World Model are not lost during persistence.
                runtime_candidate = branch_by_id.get(candidate_id, {})
                candidate_record = dict(candidate)
                candidate_record.update(runtime_candidate)
                candidate_dir = (
                    f"{depth_dir}/candidate_{candidate_index:02d}"
                    f"__action_{self._safe_name(candidate.get('primitive', 'unknown'))}"
                    f"__id_{self._safe_name(candidate_id)}"
                )
                candidate_record["verifier_keep"] = candidate_id in keep
                candidate_record["verifier_dominant"] = str(dominant) == candidate_id
                prompt_file = f"{candidate_dir}/prompt.json"
                self._state.save(
                    prompt_file,
                    {
                        "candidate_id": candidate_id,
                        "primitive": candidate_record.get("primitive"),
                        "candidate_prompt": candidate_record.get("candidate_prompt"),
                        "enhanced_prompt": candidate_record.get("enhanced_prompt", candidate_record.get("prompt")),
                        "world_model_instruction": candidate_record.get("world_model_instruction", candidate_record.get("prompt")),
                        "prompt_enhancement": candidate_record.get("prompt_enhancement", {}),
                        "timing": candidate_record.get("timing", {}),
                    },
                    step=None,
                )
                candidate_record["artifacts"] = {"prompt": prompt_file}
                artifacts.append(prompt_file)
                video = videos_by_id.get(candidate_id)
                if video is not None:
                    front_frames = np.asarray(video["front"], dtype=np.uint8)
                    action_label = self._safe_name(candidate.get("primitive", "unknown"))
                    front_video = f"{candidate_dir}/world_model_preview__action_{action_label}__front.mp4"
                    final_front = f"{candidate_dir}/result__action_{action_label}__front.png"
                    self._state.save(front_video, front_frames, step=None, fps=24)
                    self._state.save(final_front, front_frames[-1], step=None)
                    if video.get("wrist") is not None:
                        wrist_frames = np.asarray(video["wrist"], dtype=np.uint8)
                        wrist_video = f"{candidate_dir}/world_model_preview__action_{action_label}__wrist.mp4"
                        final_wrist = f"{candidate_dir}/result__action_{action_label}__wrist.png"
                        self._state.save(wrist_video, wrist_frames, step=None, fps=24)
                        self._state.save(final_wrist, wrist_frames[-1], step=None)
                        candidate_record["artifacts"].update({
                            "world_model_preview_wrist": wrist_video,
                            "final_wrist": final_wrist,
                        })
                        artifacts.extend([wrist_video, final_wrist])
                    frame_dir = f"{candidate_dir}/frames"
                    frame_files = []
                    initial_front = video.get("initial_front")
                    if initial_front is not None:
                        self._state.save(f"{frame_dir}/frame_000_initial__front.png", initial_front, step=None)
                        frame_files.append(f"{frame_dir}/frame_000_initial__front.png")
                    for frame_index, front_frame in enumerate(front_frames, start=1):
                        front_file = f"{frame_dir}/frame_{frame_index:03d}__front.png"
                        self._state.save(front_file, front_frame, step=None)
                        frame_files.append(front_file)
                    candidate_record["artifacts"].update({
                        "world_model_preview_front": front_video,
                        "final_front": final_front,
                        "frame_count": int(len(front_frames)),
                        "video_frame_count": int(len(front_frames)),
                        "frame_files": frame_files,
                    })
                    artifacts.extend([front_video, final_front] + frame_files)
                action_file = f"{candidate_dir}/action.json"
                self._state.save(action_file, candidate_record, step=None)
                artifacts.append(action_file)
                candidate_dirs.append(candidate_dir)

            verifier_file = f"{depth_dir}/verifier.json"
            decision_file = f"{depth_dir}/decision.json"
            self._state.save(verifier_file, verifier, step=None)
            self._state.save(
                decision_file,
                {
                    "search_id": search_id,
                    "step": action_step,
                    "action": action_name,
                    "depth": depth,
                    "verifier": verifier,
                    "candidate_directories": candidate_dirs,
                    "verifier_input": (
                        "Each candidate is provided as chronological front-view images "
                        "sampled once every ten generated frames, plus the final frame, "
                        "at source resolution with candidate ID and frame-index labels, "
                        "plus a wrist_input reference. Wrist predictions are supplied "
                        "only when the backend actually generates them."
                    ),
                },
                step=None,
            )
            artifacts.extend([verifier_file, decision_file])
            level["artifacts"] = artifacts
            levels.append(level)
            timing_record = {
                "step": action_step,
                "search_id": search_id,
                "depth": depth,
                "timing": summary.get("_timing", {}),
            }
            timing_file = f"{depth_dir}/timing.json"
            self._state.save(timing_file, timing_record, step=None)
            artifacts.append(timing_file)
            self._parallelworld_timing_summary.append(timing_record)
        trace_entry = {
            "step": action_step,
            "requested_action": action_name,
            "selected_action": selected_action,
            "artifact_directory": step_dir,
            "search_id": search_id,
            "levels": levels,
            "selected_candidate": (
                self._parallelworld_candidate_summary(selected)
                if isinstance(selected, dict) else None
            ),
            "selection_status": decision.get("_selection_status"),
            "error": decision.get("error"),
        }
        self._parallelworld_trace.append(trace_entry)
        self._state.save(
            "parallelworld_trace.json",
            self._parallelworld_trace,
            step=None,
        )
        search_timing = decision.get("_search_timing", {})
        self._parallelworld_timing_summary.append({
            "step": action_step,
            "search_id": search_id,
            "parallelworld_search": search_timing,
        })
        self._state.save(
            "parallelworld_timing_summary.json",
            self._parallelworld_timing_summary,
            step=None,
        )
        self._state.save(
            f"{step_dir}/selected.json",
            {
                "step": action_step,
                "requested_action": action_name,
                "selected_action": selected_action,
                "search_id": search_id,
                "selection_status": decision.get("_selection_status"),
                "selected_candidate": (
                    self._parallelworld_candidate_summary(selected)
                    if isinstance(selected, dict) else None
                ),
            },
            step=None,
        )
        self._append_parallelworld_index(
            action_step=action_step,
            action_name=action_name,
            search_id=search_id,
            step_dir=step_dir,
            decision=decision,
            levels=levels,
        )
        self._save_parallelworld_viewer()

    @staticmethod
    def _safe_name(value: Any) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return (safe or "unnamed")[:80]

    def _append_parallelworld_index(
        self,
        *,
        action_step: int,
        action_name: str,
        search_id: int,
        step_dir: str,
        decision: dict[str, Any],
        levels: list[dict[str, Any]],
    ) -> None:
        index_path = "parallelworld/index.json"
        existing = []
        try:
            existing = self._state.load(index_path, step=None)
        except (FileNotFoundError, ValueError, OSError):
            pass
        if not isinstance(existing, list):
            existing = []
        existing.append({
            "step": action_step,
            "requested_action": action_name,
            "selected_action": (
                str(decision["_selected_candidate"].get("primitive"))
                if isinstance(decision.get("_selected_candidate"), dict)
                and decision["_selected_candidate"].get("primitive")
                else "none"
            ),
            "search_id": search_id,
            "directory": step_dir,
            "selection_status": decision.get("_selection_status"),
            "selected_candidate": (
                self._parallelworld_candidate_summary(decision["_selected_candidate"])
                if isinstance(decision.get("_selected_candidate"), dict) else None
            ),
            "error": decision.get("error"),
            "depths": [
                {
                    "depth": level.get("depth"),
                    "verifier": level.get("verifier"),
                    "artifacts": level.get("artifacts", []),
                }
                for level in levels
            ],
        })
        self._state.save(index_path, existing, step=None)

    def _save_parallelworld_viewer(self) -> None:
        """Build a self-contained episode-level HTML timeline."""
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>ParallelWorld timeline</title>",
            "<style>body{font:14px sans-serif;max-width:1400px;margin:24px auto}"
            "section{border:1px solid #ccc;padding:16px;margin:16px 0}"
            ".grid{display:flex;flex-wrap:wrap;gap:12px}"
            "figure{margin:0;width:280px}img{max-width:100%}"
            "video{width:100%}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}"
            "</style></head><body><h1>ParallelWorld timeline</h1>",
        ]
        for search in self._parallelworld_trace:
            parts.append(
                f"<section><h2>Search {escape(str(search['search_id']))} "
                f"selected {escape(str(search.get('selected_candidate')))}</h2>"
            )
            for level in search.get("levels", []):
                parts.append(f"<h3>Depth {escape(str(level.get('depth')))}</h3><div class='grid'>")
                for filename in level.get("artifacts", []):
                    label = escape(filename)
                    if filename.endswith(".mp4"):
                        parts.append(
                            f"<figure><figcaption>{label}</figcaption>"
                            f"<video controls preload='metadata' src='{label}'></video></figure>"
                        )
                    elif filename.endswith(".png"):
                        parts.append(
                            f"<figure><figcaption>{label}</figcaption>"
                            f"<img loading='lazy' src='{label}'></figure>"
                        )
                parts.append("</div>")
                details = escape(json.dumps(level, indent=2, default=str))
                parts.append(f"<details><summary>Decision data</summary><pre>{details}</pre></details>")
            actual = search.get("actual_execution") or {}
            actual_artifacts = actual.get("artifacts") or {}
            if actual_artifacts:
                parts.append("<h3>Actual execution</h3><div class='grid'>")
                for filename in actual_artifacts.values():
                    label = escape(str(filename))
                    if str(filename).endswith(".mp4"):
                        parts.append(
                            f"<figure><figcaption>{label}</figcaption>"
                            f"<video controls preload='metadata' src='{label}'></video></figure>"
                        )
                parts.append("</div>")
                details = escape(json.dumps(actual, indent=2, default=str))
                parts.append(
                    f"<details><summary>Actual action trace</summary><pre>{details}</pre></details>"
                )
            parts.append("</section>")
        parts.append("</body></html>")
        self._state.save(
            "parallelworld_viewer.html",
            "\n".join(parts),
            step=None,
        )

    def _save_actual_execution_artifacts(self, *, frame_start: int) -> None:
        """Persist the complete real execution after a selected route runs."""
        pending = self._pending_parallel_execution
        if pending is None:
            return
        try:
            records = self._primitives.execution_slice(frame_start)
            if not records:
                return
            prefix = f'{pending["step_dir"]}/actual_execution'
            front_frames = [pending["initial_front"]]
            front_frames.extend(
                np.ascontiguousarray(
                    np.asarray(item["front"], dtype=np.uint8)[:, ::-1]
                )
                for item in records
            )
            wrist_frames = [pending["initial_wrist"]]
            wrist_frames.extend(
                np.ascontiguousarray(
                    np.asarray(item["wrist"], dtype=np.uint8)[:, ::-1]
                )
                for item in records if item.get("wrist") is not None
            )
            front_video = f"{prefix}/actual_execution__front.mp4"
            wrist_video = f"{prefix}/actual_execution__wrist.mp4"
            actions_file = f"{prefix}/actual_execution__actions.json"
            metadata_file = f"{prefix}/actual_execution.json"
            self._state.save(front_video, np.asarray(front_frames, dtype=np.uint8), step=None, fps=20)
            if wrist_frames:
                self._state.save(wrist_video, np.asarray(wrist_frames, dtype=np.uint8), step=None, fps=20)
            actions = [
                np.asarray(item["action"], dtype=np.float32).tolist()
                for item in records if item.get("action") is not None
            ]
            self._state.save(actions_file, actions, step=None)
            metadata = {
                "status": "completed",
                "requested_action": pending["requested_action"],
                "selected_candidate": pending["selected_candidate"],
                "search_id": pending["search_id"],
                "real_env_steps": len(records),
                "video_frame_count": len(front_frames),
                "video_fps": 20,
                "initial_frame_included": True,
                "execution_elapsed_s": pending.get("execution_elapsed_s"),
                "parallelworld_timing": pending.get("parallelworld_timing", {}),
                "note": (
                    "This is the complete real environment execution. Candidate "
                    "videos are full closed-loop simulations from restored snapshots. "
                    "Front and wrist frames use the camera-calibration orientation, "
                    "not the horizontally mirrored Pi0 policy orientation."
                ),
                "artifacts": {"front": front_video, "actions": actions_file},
            }
            if wrist_frames:
                metadata["artifacts"]["wrist"] = wrist_video
            self._state.save(metadata_file, metadata, step=None)
            if self._parallelworld_trace:
                trace_entry = self._parallelworld_trace[-1]
                metadata["artifacts"]["metadata"] = metadata_file
                trace_entry["actual_execution"] = metadata
                self._state.save("parallelworld_trace.json", self._parallelworld_trace, step=None)
                self._save_parallelworld_viewer()
        finally:
            self._pending_parallel_execution = None

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        if self._pending_parallel_execution is not None:
            self._pending_parallel_execution["execution_elapsed_s"] = float(elapsed_s)
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        self._save_actual_execution_artifacts(frame_start=frame_start)
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = libero_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.update(result)
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            get_logger("libero_toolkit").warning(
                f"failed to save episode video: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(self._state, recipe_tag)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch tools, running internal ParallelWorld before real actions."""
        if name == "parallelworld_decide" and self._parallelworld_gate is not None:
            try:
                selected_name, selected_args = self._parallelworld_select(
                    name, dict(input_dict)
                )
                executed = super().execute_tool(selected_name, selected_args)
                result = dict(executed.result)
                result["parallelworld_selected_primitive"] = selected_name
                result["parallelworld_selected_arguments"] = selected_args
                return ToolResult(name=name, result=result, call_id=executed.call_id)
            except Exception as exc:
                return ToolResult(name=name, result={"error": str(exc)})
        physical = {
            "move_to", "move_pose", "navigate_to", "move_base", "pi0_pick", "pi0_doubled", "release",
            "set_gripper", "rotate_wrist", "rotate_pitch",
        }
        if self._parallelworld_gate is not None and name in physical:
            try:
                name, input_dict = self._parallelworld_select(name, dict(input_dict))
            except Exception as exc:
                return ToolResult(name=name, result={
                    "error": str(exc), "code": "parallelworld_route_required"
                })
        return super().execute_tool(name, input_dict)
