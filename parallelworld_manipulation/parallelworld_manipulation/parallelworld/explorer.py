"""Natural-language candidate search for a LIBERO I2V world model."""
from __future__ import annotations
import io, json, time
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
from parallelworld_manipulation.tools.toolkit import readonly


_IMAGE_FRAME_CONVENTION = (
    "Coordinate convention: image-left and image-right always mean the viewer's "
    "left and right in Image 1 exactly as displayed. Never interpret left or "
    "right from the robot's own perspective unless an explicit robot or world "
    "coordinate frame is stated."
)


def _calibration_images(primitives: Any, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-calibration RGBs, not the horizontally mirrored Pi0 RGBs."""
    front = None
    wrist = None
    raw_obs = getattr(getattr(primitives, "env", None), "raw_obs", None)
    if callable(raw_obs):
        try:
            raw = raw_obs()
            front = raw.get("agentview_image")
            wrist = raw.get("robot0_eye_in_hand_image")
        except Exception:
            pass

    # LIBERO's policy observations are exact horizontal mirrors of the RGBs
    # aligned with the saved depth maps and camera matrices.  Keep this fallback
    # for lightweight test doubles and older environment servers.
    return (
        (
            np.ascontiguousarray(np.asarray(front, dtype=np.uint8)[::-1])
            if front is not None else
            np.ascontiguousarray(
                np.asarray(obs["main_images"], dtype=np.uint8)[:, ::-1]
            )
        ),
        (
            np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8)[::-1])
            if wrist is not None else
            np.ascontiguousarray(
                np.asarray(obs["wrist_images"], dtype=np.uint8)[:, ::-1]
            )
        ),
    )


def _calibration_depths(primitives: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return compact raw depth views aligned with the planner cameras."""
    raw_obs = getattr(getattr(primitives, "env", None), "raw_obs", None)
    if not callable(raw_obs):
        return None, None
    try:
        raw = raw_obs()
    except Exception:
        return None, None
    def normalize(value: Any) -> np.ndarray | None:
        try:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 2 or not arr.size:
                return None
            finite = np.isfinite(arr)
            if not finite.any():
                return None
            lo, hi = np.percentile(arr[finite], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            return np.asarray(np.clip((arr - lo) / (hi - lo), 0, 1) * 255,
                              dtype=np.uint8)
        except Exception:
            return None
    return (normalize(raw.get("agentview_depth")),
            normalize(raw.get("robot0_eye_in_hand_depth")))


def _jsonable_context(value: Any) -> Any:
    """Convert small numeric metadata trees to JSON without raw image arrays."""
    if isinstance(value, np.ndarray):
        if value.size > 256:
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "summary": "large_numeric_array_omitted"}
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable_context(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_context(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _compact_planner_context(primitives: Any, obs: dict[str, Any],
                             camera_meta: dict[str, Any] | None,
                             *, task: str, depth: int,
                             decision_count: int) -> dict[str, Any]:
    """Serialize planner-visible state without embedding giant raw arrays."""
    states = np.asarray(obs.get("states", []), dtype=np.float32).reshape(-1)
    context: dict[str, Any] = {
        "source": "planner_current_observation",
        "task_language": task,
        "parallelworld_depth": int(depth),
        "observation_keys": sorted(str(k) for k in obs.keys()),
        "decision_count": int(decision_count),
        "tool_state": {
            "eef_xyz": [round(float(x), 5) for x in states[:3]] if len(states) >= 3 else None,
            "state_vector_length": int(len(states)),
            "gripper_opening": round(float(getattr(primitives, "_last_obs_gripper", 0.0)), 5),
            "eef_z": round(float(getattr(primitives, "_last_obs_eef_z", 0.0)), 5),
            "task_descriptions": _jsonable_context(obs.get("task_descriptions", "")),
        },
        "camera_calibration": _jsonable_context(camera_meta),
    }
    trace = getattr(primitives, "_action_trace", [])
    # Do not send the full low-level action history: it grows with every
    # environment step and can make verifier requests excessively large.
    last_action = None
    if trace and trace[-1].get("action") is not None:
        last_action = [round(float(x), 5) for x in np.asarray(
            trace[-1]["action"], dtype=np.float32).reshape(-1)]
    context["execution_summary"] = {
        "recorded_env_steps": int(len(trace)),
        "last_action": last_action,
    }
    return context


def _project_world_xyz(
    xyz: Any,
    camera_meta: dict[str, Any] | None,
    *,
    height: int,
    width: int,
) -> tuple[float, float] | None:
    """Project a world XYZ endpoint to normalized calibration-image coordinates."""
    if not isinstance(xyz, (list, tuple, np.ndarray)) or len(xyz) != 3:
        return None
    if not camera_meta:
        return None
    try:
        intrinsic = np.asarray(camera_meta["intrinsic_K"], dtype=np.float64)
        cam2world = np.asarray(
            camera_meta["extrinsic_cam2world"], dtype=np.float64
        )
        world = np.asarray([*map(float, xyz), 1.0], dtype=np.float64)
        camera = np.linalg.solve(cam2world, world)
        if not np.isfinite(camera).all() or camera[2] <= 0:
            return None
        col = intrinsic[0, 0] * camera[0] / camera[2] + intrinsic[0, 2]
        row = intrinsic[1, 1] * camera[1] / camera[2] + intrinsic[1, 2]
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not (0 <= col < width and 0 <= row < height):
        return None
    return float(col / max(1, width - 1)), float(row / max(1, height - 1))


def _mark_endpoint(
    front: np.ndarray,
    normalized_uv: tuple[float, float] | None,
) -> np.ndarray:
    """Overlay a high-contrast A marker used only as world-model guidance."""
    if normalized_uv is None:
        return front
    image = Image.fromarray(front).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    col = int(round(normalized_uv[0] * (width - 1)))
    row = int(round(normalized_uv[1] * (height - 1)))
    radius = max(6, min(width, height) // 32)
    color = (255, 0, 255)
    draw.ellipse(
        (col - radius, row - radius, col + radius, row + radius),
        outline=color,
        width=max(2, radius // 3),
    )
    draw.line((col - radius, row, col + radius, row), fill=color, width=2)
    draw.line((col, row - radius, col, row + radius), fill=color, width=2)
    draw.text((col + radius + 2, row - radius), "A", fill=color)
    return np.asarray(image, dtype=np.uint8)


def _normalized_target_uv(candidate: dict[str, Any]) -> tuple[float, float] | None:
    value = candidate.get("visual_action", {}).get("target_uv")
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        return None
    try:
        col, row = map(float, value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([col, row]).all() or not (0 <= col <= 1 and 0 <= row <= 1):
        return None
    return col, row

def _sheet(images: list[np.ndarray], labels: list[str]) -> bytes | None:
    if not images: return None
    thumbs = [Image.fromarray(x).convert("RGB").resize((256, 256)) for x in images]
    cols = min(4, len(thumbs)); rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 256, rows * 286), "white"); draw = ImageDraw.Draw(canvas)
    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        x, y = (i % cols) * 256, (i // cols) * 286
        canvas.paste(thumb, (x, y + 30)); draw.text((x + 5, y + 7), label[:36], fill="black")
    buf = io.BytesIO(); canvas.save(buf, format="PNG"); return buf.getvalue()


def _verifier_frame_indices(frame_count: int, stride: int = 10) -> list[int]:
    """Sample a rollout at ``stride`` and always retain its terminal frame."""
    if frame_count <= 0:
        return []
    indices = list(range(0, int(frame_count), int(stride)))
    final_index = int(frame_count) - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


class ParallelWorldExplorer:
    """Evaluate planner-proposed primitives through text-conditioned I2V."""
    N_MAX, K_MAX, M_MAX, D_MAX = 4, 3, 4, 4
    def __init__(self, *, primitives: Any, client: Any, tts: bool, tts_samples: int,
                 rollout_steps: int, seed: int = 0, verifier: Any | None = None,
                 prompt_enhancer: Any | None = None,
                 task: str = "", **_: Any) -> None:
        self._primitives, self._client = primitives, client
        self._tts, self._tts_samples = bool(tts), max(2, int(tts_samples))
        self._horizon, self._seed = max(1, int(rollout_steps)), int(seed)
        self._verifier, self._prompt_enhancer, self._task = verifier, prompt_enhancer, task
        self._imagined_states = {}; self._search_counter = 0
        self._last_search_history = []; self._last_rollout_videos = []
        self._last_real_images = None
        self._last_timing = {}
        self._last_search_timing = {}

    def _start_images(
        self,
        candidate: dict[str, Any],
        *,
        depth: int,
        parent_id: str | None,
        real_front: np.ndarray,
        real_wrist: np.ndarray,
    ) -> tuple[str | None, np.ndarray, np.ndarray]:
        """Resolve the exact state from which one candidate must branch."""
        candidate_parent = (
            candidate.get("parent_id")
            or candidate.get("_parent_id")
            or parent_id
        )
        if depth <= 1:
            return None, real_front, real_wrist
        if candidate_parent is None:
            raise ValueError(
                f"candidate {candidate.get('id')!r} at depth {depth} has no parent_id"
            )
        candidate_parent = str(candidate_parent)
        imagined = self._imagined_states.get(candidate_parent)
        if imagined is None:
            raise ValueError(
                f"imagined state for parent {candidate_parent!r} is unavailable"
            )
        return candidate_parent, imagined[0], imagined[1]

    def _current_task(self, obs: dict[str, Any]) -> str:
        """Return the real environment instruction, not the suite/task placeholder."""
        # The environment RPC is authoritative.  ``task_descriptions`` is
        # temporarily replaced by low-level VLA instructions during a skill.
        task = ""
        env = getattr(self._primitives, "env", None)
        get_task_language = getattr(env, "get_task_language", None)
        if callable(get_task_language):
            try:
                task = str(get_task_language() or "").strip()
            except Exception:
                task = ""
        if task:
            return task
        value: Any = obs.get("task_descriptions")
        if isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            value = next(
                (item for item in value if item is not None and str(item).strip()),
                "",
            )
        task = str(value or "").strip()
        return task or self._task

    def _enhance_group(
        self,
        *,
        candidates: list[dict[str, Any]],
        task: str,
        start_states: dict[str, tuple[str | None, np.ndarray, np.ndarray]],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
        """Enhance one depth as a contrastive group using each branch's own image."""
        from .prompt_enhancer import contains_control_syntax, semantic_fallback

        raw_prompts = {
            str(candidate["id"]): json.dumps(
                {"task": task, "candidate": candidate}, ensure_ascii=True
            )
            for candidate in candidates
        }
        valid_candidates = [
            candidate for candidate in candidates if str(candidate["id"]) in start_states
        ]
        fallback_prompts = {
            str(candidate["id"]): semantic_fallback(candidate)
            for candidate in candidates
        }
        prompts = dict(fallback_prompts)
        metadata: dict[str, dict[str, Any]] = {
            candidate_id: {
                "source": "not_run",
                "fallback_reason": "candidate start state is unavailable",
            }
            for candidate_id in raw_prompts
            if candidate_id not in start_states
        }
        if self._prompt_enhancer is None or not valid_candidates:
            for candidate in valid_candidates:
                candidate_id = str(candidate["id"])
                metadata[candidate_id] = {
                    "source": "semantic_fallback",
                    "fallback_reason": "prompt enhancer is not configured",
                }
            return raw_prompts, prompts, metadata

        enhance_batch = getattr(self._prompt_enhancer, "enhance_batch", None)
        if callable(enhance_batch):
            records = enhance_batch(
                candidates=valid_candidates,
                task=task,
                fronts={
                    str(candidate["id"]): start_states[str(candidate["id"])][1].copy()
                    for candidate in valid_candidates
                },
                rollout_steps=self._horizon,
            )
            for candidate in valid_candidates:
                candidate_id = str(candidate["id"])
                record = records.get(candidate_id, {}) if isinstance(records, dict) else {}
                if isinstance(record, str):
                    prompt, detail = record, {"source": "llm"}
                elif isinstance(record, dict):
                    prompt = str(record.get("instruction") or "").strip()
                    detail = dict(record)
                    detail.pop("instruction", None)
                else:
                    prompt, detail = "", {}
                if not prompt:
                    prompt = fallback_prompts[candidate_id]
                    detail = {
                        **detail,
                        "source": "semantic_fallback",
                        "fallback_reason": (
                            detail.get("fallback_reason")
                            or "enhancer returned no instruction for this candidate"
                        ),
                    }
                prompts[candidate_id] = prompt
                metadata[candidate_id] = detail
            return raw_prompts, prompts, metadata

        # Compatibility for simple/custom enhancers that predate grouped input.
        for candidate in valid_candidates:
            candidate_id = str(candidate["id"])
            try:
                enhanced = self._prompt_enhancer.enhance(
                    candidate=candidate, task=task
                )
                if not enhanced or contains_control_syntax(enhanced):
                    raise ValueError("legacy enhancer returned an unsafe instruction")
                prompts[candidate_id] = enhanced
                metadata[candidate_id] = {
                    "source": "legacy_enhancer",
                    "fallback_reason": None,
                }
            except Exception as exc:
                metadata[candidate_id] = {
                    "source": "semantic_fallback",
                    "fallback_reason": str(exc),
                }
        return raw_prompts, prompts, metadata

    @readonly
    def decide(self, *, candidates: list[dict[str, Any]], depth: int = 1,
               parent_id: str | None = None, candidate_limit: int | None = None) -> dict[str, Any]:
        if not candidates: raise ValueError("candidates must contain at least one branch")
        candidates = list(candidates)[:candidate_limit or self.N_MAX]
        ids = [str(c.get("id", "")).strip() for c in candidates]
        if any(not x for x in ids) or len(set(ids)) != len(ids): raise ValueError("candidate ids must be unique")
        obs = dict(self._primitives._last_obs)
        real_front, real_wrist = _calibration_images(self._primitives, obs)
        camera_meta = None
        get_camera_meta = getattr(
            getattr(self._primitives, "env", None), "get_camera_meta", None
        )
        if callable(get_camera_meta):
            try:
                camera_meta = get_camera_meta(
                    camera_name="agentview",
                    height=int(real_front.shape[0]),
                    width=int(real_front.shape[1]),
                )
            except Exception:
                camera_meta = None
        task = self._current_task(obs)
        self._last_real_images = (real_front.copy(), real_wrist.copy())
        samples = self._tts_samples if self._tts else 1
        decide_started = time.perf_counter()
        branches, previews, labels, videos, verifier_frames = [], [], [], [], []
        candidate_timings = []

        # Resolve every branch before enhancement so the group call sees the
        # exact real/imagined front frame from which each rollout will begin.
        start_states: dict[str, tuple[str | None, np.ndarray, np.ndarray]] = {}
        spatial_guidance: dict[str, dict[str, Any]] = {}
        start_errors: dict[str, Exception] = {}
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            try:
                resolved_parent, front, wrist = self._start_images(
                    candidate,
                    depth=depth,
                    parent_id=parent_id,
                    real_front=real_front,
                    real_wrist=real_wrist,
                )
                normalized_uv = _normalized_target_uv(candidate)
                guidance_meaning = "intended target center"
                guidance_source = "visual_action.target_uv"
                if normalized_uv is None:
                    normalized_uv = _project_world_xyz(
                        candidate.get("arguments", {}).get("xyz"),
                        camera_meta,
                        height=int(front.shape[0]),
                        width=int(front.shape[1]),
                    )
                    guidance_meaning = "projected terminal end-effector waypoint"
                    guidance_source = "arguments.xyz projection"
                start_states[candidate_id] = (
                    resolved_parent,
                    _mark_endpoint(front, normalized_uv),
                    wrist,
                )
                spatial_guidance[candidate_id] = {
                    "image_frame": "agentview calibration frame",
                    "normalized_uv": (
                        [round(normalized_uv[0], 4), round(normalized_uv[1], 4)]
                        if normalized_uv is not None else None
                    ),
                    "marker": "A" if normalized_uv is not None else None,
                    "meaning": guidance_meaning if normalized_uv is not None else None,
                    "source": guidance_source if normalized_uv is not None else None,
                }
            except Exception as exc:
                start_errors[candidate_id] = exc

        prompt_started = time.perf_counter()
        try:
            raw_prompts, prompts, enhancement_metadata = self._enhance_group(
                candidates=candidates, task=task, start_states=start_states
            )
        except Exception as exc:
            from .prompt_enhancer import semantic_fallback

            raw_prompts = {
                str(candidate["id"]): json.dumps(
                    {"task": task, "candidate": candidate}, ensure_ascii=True
                )
                for candidate in candidates
            }
            prompts = {
                str(candidate["id"]): semantic_fallback(candidate)
                for candidate in candidates
            }
            enhancement_metadata = {
                str(candidate["id"]): {
                    "source": "semantic_fallback",
                    "fallback_reason": f"group enhancement failed: {exc}",
                }
                for candidate in candidates
            }
        prompt_elapsed = time.perf_counter() - prompt_started
        for i, candidate in enumerate(candidates):
            candidate_started = time.perf_counter()
            candidate_id = ids[i]
            raw_prompt = raw_prompts[candidate_id]
            prompt = prompts[candidate_id]
            guidance = spatial_guidance.get(candidate_id, {})
            normalized_uv = guidance.get("normalized_uv")
            if normalized_uv is not None:
                marker_meaning = guidance["meaning"]
                prompt = (
                    f"{prompt}\n\n{_IMAGE_FRAME_CONVENTION} "
                    f"The magenta marker A in Image 1 indicates the {marker_meaning} "
                    "for this candidate at "
                    f"normalized image coordinates (u={normalized_uv[0]:.4f}, "
                    f"v={normalized_uv[1]:.4f}), with origin at the top-left, "
                    "+u right, and +v down. Use marker A as spatial guidance, "
                    "but remove the marker and its label from the generated image."
                )
            else:
                prompt = f"{prompt}\n\n{_IMAGE_FRAME_CONVENTION}"
            enhancement = enhancement_metadata.get(candidate_id, {})
            try:
                if candidate_id in start_errors:
                    raise start_errors[candidate_id]
                resolved_parent_id, front, wrist = start_states[candidate_id]
                rollout_started = time.perf_counter()
                result = self._client.rollout(front=front, wrist=wrist, instruction=prompt, horizon=self._horizon,
                    samples=samples, stochastic=self._tts, seed=self._seed + i * samples)
                rollout_elapsed = time.perf_counter() - rollout_started
                fs = np.asarray(result["front"])
                # Front is the actual imagined state.  Keep a wrist value only as
                # an opaque compatibility input for joint multi-view backends;
                # front-only backends return no wrist and simply carry the input.
                generated_wrist = result.get("wrist")
                next_wrist = (
                    np.asarray(generated_wrist)[0, -1].copy()
                    if generated_wrist is not None
                    else wrist.copy()
                )
                self._imagined_states[ids[i]] = (fs[0, -1].copy(), next_wrist)
                candidate_elapsed = time.perf_counter() - candidate_started
                candidate_timing = {"candidate_id": ids[i], "prompt_enhancement_s": round(prompt_elapsed, 4),
                    "world_model_rollout_s": round(rollout_elapsed, 4),
                    "candidate_total_s": round(candidate_elapsed, 4)}
                candidate_timings.append(candidate_timing)
                branches.append({"id": ids[i], "parent_id": resolved_parent_id,
                    "primitive": candidate.get("primitive"),
                    "arguments": candidate.get("arguments", {}), "rationale": candidate.get("rationale", ""),
                    "candidate_prompt": raw_prompt, "prompt": prompt, "enhanced_prompt": prompt,
                    "world_model_instruction": prompt, "prompt_enhancement": enhancement,
                    "spatial_guidance": guidance,
                    "timing": candidate_timing,
                    "accepted": True, "video_horizon": int(fs.shape[1]),
                    "same_action_consistency": self._consistency(fs[:, -1])})
                previews.append(fs[0, -1]); labels.append(ids[i])
                videos.append({"id": ids[i], "parent_id": resolved_parent_id,
                               "initial_front": front.copy(),
                               "initial_wrist": wrist.copy(),
                               "front": fs[0].astype(np.uint8),
                               **({"wrist": np.asarray(generated_wrist)[0].astype(np.uint8)}
                                  if generated_wrist is not None else {}),
                               "candidate_prompt": raw_prompt, "enhanced_prompt": prompt,
                               "world_model_instruction": prompt,
                               "spatial_guidance": guidance,
                               "prompt_enhancement": enhancement,
                               "timing": candidate_timing})
                # The GPT Image backend predicts front views only. Preserve the
                # real wrist observation alongside every candidate so the verifier
                # can judge gripper contact and held-object state instead of making
                # a front-only decision.
                verifier_frames.append({"candidate_id": ids[i], "frame_index": 0, "view": "wrist_input", "image": wrist.copy()})
                for frame_index in _verifier_frame_indices(fs.shape[1]):
                    verifier_frames.append({
                        "candidate_id": ids[i], "frame_index": frame_index,
                        "view": "front", "image": fs[0, frame_index],
                    })
            except Exception as exc:
                candidate_elapsed = time.perf_counter() - candidate_started
                candidate_timing = {"candidate_id": ids[i], "prompt_enhancement_s": round(prompt_elapsed, 4),
                    "candidate_total_s": round(candidate_elapsed, 4), "failed": True}
                candidate_timings.append(candidate_timing)
                branches.append({"id": ids[i],
                    "parent_id": candidate.get("parent_id") or candidate.get("_parent_id") or parent_id,
                    "primitive": candidate.get("primitive"),
                    "arguments": candidate.get("arguments", {}), "candidate_prompt": raw_prompt,
                    "prompt": prompt, "enhanced_prompt": prompt, "world_model_instruction": prompt,
                    "prompt_enhancement": enhancement, "spatial_guidance": guidance,
                    "timing": candidate_timing,
                    "accepted": False, "world_model_error": str(exc)})
        valid = [b for b in branches if "world_model_error" not in b]; decision = None
        verifier_elapsed = 0.0
        if self._verifier and valid:
            try:
                verifier_started = time.perf_counter()
                decision = self._verifier.decide(
                    depth=depth, candidates=valid, frames=verifier_frames,
                    task_language=task,
                )
                verifier_elapsed = time.perf_counter() - verifier_started
            except Exception as exc: decision = {"keep": [], "dominant_id": None, "continue_search": False, "reason": f"verifier_failed: {exc}"}
            keep = set(decision.get("keep", []))
            for branch in branches: branch["accepted"] = branch["id"] in keep
        total_elapsed = time.perf_counter() - decide_started
        self._last_timing = {"depth": int(depth), "candidate_count": len(candidates),
            "valid_candidate_count": len(valid), "candidate_timings": candidate_timings,
            "prompt_enhancement_batch_s": round(prompt_elapsed, 4),
            "verifier_s": round(verifier_elapsed, 4), "total_s": round(total_elapsed, 4)}
        self._last_rollout_videos = videos
        return {"depth": int(depth), "parent_id": parent_id, "branches": branches,
            "fallback_candidates": candidates, "verifier": decision,
            "_timing": self._last_timing,
            "tts": {"enabled": self._tts, "samples_per_candidate": samples},
            "selection_policy": "Verifier selects one planner candidate; only that original primitive executes.",
            "_videos": videos, "_image_cam_bytes": _sheet(previews, labels),
            "_image_wrist_bytes": None}

    def _build_prompts(self, candidate: dict[str, Any]) -> tuple[str, str]:
        from .prompt_enhancer import semantic_fallback

        raw = json.dumps({"task": self._task, "candidate": candidate}, ensure_ascii=True)
        if self._prompt_enhancer is None:
            return raw, semantic_fallback(candidate)
        try:
            prompt = self._prompt_enhancer.enhance(candidate=candidate, task=self._task)
        except Exception:
            prompt = semantic_fallback(candidate)
        return raw, prompt

    def _prompt(self, candidate: dict[str, Any]) -> str:
        return self._build_prompts(candidate)[1]

    def search(self, candidates: list[dict[str, Any]], *, next_candidate_generator: Any | None = None) -> dict[str, Any]:
        self._search_counter += 1
        search_started = time.perf_counter()
        self._last_search_history = []
        self._imagined_states = {}
        current = list(candidates)[:self.N_MAX]
        previous = dict(current[0]) if current else None
        for depth in range(1, self.D_MAX + 1):
            parent_id = None
            if depth > 1 and current:
                parent_id = current[0].get("parent_id") or current[0].get("_parent_id")
            result = self.decide(candidates=current, depth=depth, parent_id=parent_id)
            self._last_search_history.append({
                "search_id": self._search_counter, "depth": depth,
                "front": result.get("_image_cam_bytes"), "wrist": result.get("_image_wrist_bytes"),
                "real_front": self._last_real_images[0] if depth == 1 and self._last_real_images else None,
                "real_wrist": self._last_real_images[1] if depth == 1 and self._last_real_images else None,
                "videos": self._last_rollout_videos,
                "summary": result,
            })
            decision = result.get("verifier") or {}
            kept = {str(x) for x in decision.get("keep", [])}
            selected = next(
                (
                    candidate
                    for candidate in current
                    if str(candidate.get("id"))
                    == str(decision.get("dominant_id"))
                ),
                None,
            )
            if selected is not None:
                previous = dict(selected)
            if not decision.get("continue_search") or depth >= self.D_MAX:
                self._last_search_timing = {"search_id": self._search_counter,
                    "depth_count": depth, "depth_timings": [x["summary"].get("_timing", {}) for x in self._last_search_history],
                    "total_s": round(time.perf_counter() - search_started, 4)}
                result["_selected_candidate"] = previous; result["_selection_status"] = "normal"
                result["_search_timing"] = self._last_search_timing; return result
            if next_candidate_generator is None: break
            if selected is not None:
                kept.add(str(selected["id"]))
            retained = [c for c in current if str(c.get("id")) in kept][:self.K_MAX]
            children_by_parent = []
            for parent in retained:
                parent_id_value = str(parent["id"])
                imagined = self._imagined_states.get(parent_id_value)
                if imagined is None:
                    raise RuntimeError(
                        f"retained route {parent_id_value!r} has no imagined state"
                    )
                planner_parent = dict(parent)
                planner_parent["_imagined_front"] = imagined[0].copy()
                parent_children = []
                for child in (next_candidate_generator(planner_parent, self.M_MAX) or [])[:self.M_MAX]:
                    child = dict(child)
                    child["parent_id"] = parent_id_value
                    child["id"] = f"{parent_id_value}::{child.get('id', len(parent_children))}"
                    parent_children.append(child)
                children_by_parent.append(parent_children)
            # Interleave children so the global per-depth budget does not let
            # the first retained parent starve every other branch.
            expanded = [
                children[index]
                for index in range(self.M_MAX)
                for children in children_by_parent
                if index < len(children)
            ][:self.N_MAX]
            if not expanded: break
            current = expanded
        self._last_search_timing = {"search_id": self._search_counter,
            "depth_count": len(self._last_search_history),
            "depth_timings": [x["summary"].get("_timing", {}) for x in self._last_search_history],
            "total_s": round(time.perf_counter() - search_started, 4)}
        return {"error": "parallel search did not select a route", "_selected_candidate": previous,
                "_selection_status": "fallback_previous_top1", "_search_timing": self._last_search_timing}

    @staticmethod
    def _consistency(samples: np.ndarray) -> float | None:
        if len(samples) < 2: return None
        vals = [1.0 - float(np.mean(np.abs(samples[i].astype(float)-samples[j].astype(float))) / 255.0)
                for i in range(len(samples)) for j in range(i + 1, len(samples))]
        return round(float(np.mean(vals)), 6)
