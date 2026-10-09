from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import omnigibson as og

import baseline_active as base
import branching as branch_base
import answer_evidence


PIPELINE_VARIANT = "parallelworld"


_PREFERRED_CAMERA_ACTION_ORDER = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "turn_up",
    "turn_down",
)
BRANCH_CAMERA_ACTIONS = tuple(
    action for action in _PREFERRED_CAMERA_ACTION_ORDER if action in base.CAMERA_ACTIONS
) + tuple(sorted(base.CAMERA_ACTIONS.difference(_PREFERRED_CAMERA_ACTION_ORDER)))
HEAVY_ACTION_FRAME_THRESHOLD = 8
HEAVY_ACTION_FRAME_STEP_WINDOW = 2
EXPLICIT_OBJECT_STATE_TASKS = {
    "action",
    "counting",
    "deformable",
    "size",
    "stacking",
    "storage",
    "transparent",
}
OBJECT_STATE_FREE_TASKS = {
    "angle_confusion",
    "cognitivemap",
    "distance",
    "line",
    "mirror",
    "multiagent",
    "occlusion",
    "slope",
    "touching",
    "triangle",
    "unobserved_changes",
}


def _ordered_unique(actions) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_action in actions:
        action = base.normalize_text(raw_action).lower()
        if not action or action in seen:
            continue
        seen.add(action)
        output.append(action)
    return tuple(output)


def _generic_object_actions(task_module, payload: dict[str, Any], camera_info: dict[str, Any]) -> list[str]:
    names = [
        base.normalize_text(display_name).lower()
        for display_name in base.object_display_names(task_module, payload, camera_info).values()
        if base.normalize_text(display_name)
    ]
    actions = [f"pick up {name}" for name in names]
    actions.extend(
        f"place {name} on top of {target}"
        for name in names
        for target in names
        if name != target
    )
    actions.extend(f"put back {name}" for name in names)
    return actions


def _task_branch_actions(task_module, payload: dict[str, Any], camera_info: dict[str, Any]) -> tuple[str, ...]:
    """Return every executable non-stop action exposed by the original task pipeline."""
    task_name = base.normalize_text(getattr(task_module, "TASK_NAME", ""))
    actions: list[str] = list(BRANCH_CAMERA_ACTIONS)

    if task_name in {"action", "counting"}:
        actions.extend(_generic_object_actions(task_module, payload, camera_info))
    elif task_name == "size":
        actions.extend(
            [
                f"pick_up_obj_near_{task_module.ref_label(payload, 1).lower()}",
                f"pick_up_obj_near_{task_module.ref_label(payload, 2).lower()}",
            ]
        )
    elif task_name == "transparent":
        actions.extend(["pickup", "pour"])
    elif task_name == "slope":
        actions.append("put_on_slope")
    elif task_name == "pour":
        actions.extend(
            [
                "fill left",
                "fill right",
                "pour left into right",
                "pour right into left",
            ]
        )
    elif task_name == "stacking":
        names = [
            task_module.display_category(category).lower()
            for category in task_module.object_categories(payload)
            if base.normalize_text(category)
        ]
        actions.extend(f"pick up {name}" for name in names)
        actions.extend(
            f"place {name} on top of {target}"
            for name in names
            for target in names
            if name != target
        )
        actions.extend(f"place {name} on the floor" for name in names)
    elif task_name == "storage":
        item = task_module.normalize_label(task_module.containee(payload).get("cat")).lower()
        containers = [name.lower() for name in task_module.container_display_names(payload).values()]
        actions.append(f"pick up {item}")
        actions.extend(f"put {item} inside {container}" for container in containers)
    else:
        # This also keeps future task modules aligned when they expose literal
        # physical actions through one of the conventions used in tasks/.
        for attr_name in ("VALID_ACTIONS", "CAMERA_ACTIONS", "VALID_CAMERA_ACTIONS"):
            raw_actions = getattr(task_module, attr_name, ()) or ()
            if isinstance(raw_actions, set):
                raw_actions = sorted(raw_actions)
            for action in raw_actions:
                normalized = base.normalize_text(action).lower()
                if normalized not in base.CAMERA_ACTIONS and normalized != "stop":
                    actions.append(normalized)

    return _ordered_unique(actions)


def _clone_branch_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_branch_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_branch_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_branch_value(item) for item in value)
    if isinstance(value, set):
        return {_clone_branch_value(item) for item in value}
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, random.Random):
        cloned = random.Random()
        cloned.setstate(value.getstate())
        return cloned
    if type(value).__module__.startswith("torch") and hasattr(value, "clone"):
        try:
            return value.clone()
        except Exception:
            pass
    # OmniGibson objects, robots, and systems are live handles and must remain
    # shared; their mutable simulator state is restored separately below.
    return value


def _clone_task_state(task_state: dict[str, Any] | None) -> dict[str, Any]:
    return _clone_branch_value(task_state or {})


def _cleanup_transient_branch_constraints(task_module, task_state: dict[str, Any] | None) -> None:
    state = task_state if isinstance(task_state, dict) else {}
    try:
        base.detach_held_object(state)
    except Exception:
        pass
    task_name = base.normalize_text(getattr(task_module, "TASK_NAME", ""))
    try:
        if task_name == "stacking" and hasattr(task_module, "detach_grasp"):
            task_module.detach_grasp()
        elif task_name == "storage" and hasattr(task_module, "_detach_grasp"):
            task_module._detach_grasp(state)
    except Exception:
        pass


def _rehydrate_branch_constraints(task_module, env, task_state: dict[str, Any]) -> None:
    task_name = base.normalize_text(getattr(task_module, "TASK_NAME", ""))
    if task_name == "stacking" and task_state.get("held_obj_name"):
        robot = env.robots[0] if getattr(env, "robots", None) else None
        obj = (task_state.get("objs_by_name") or {}).get(task_state["held_obj_name"])
        if robot is not None and obj is not None and hasattr(task_module, "attach_current_grasp"):
            task_module.attach_current_grasp(robot, obj)
    elif task_name == "storage" and task_state.pop("ag_joint_prim_path", None):
        obj = env.scene.object_registry("name", "obj_containee")
        if obj is not None and hasattr(task_module, "_attach_current_grasp"):
            base.set_visual_only(obj, True)
            task_module._attach_current_grasp(env, obj, task_state)
    elif task_state.pop("physical_action_joint_prim_path", None) and task_state.get("held_obj_name"):
        obj = env.scene.object_registry("name", task_state["held_obj_name"])
        if obj is not None:
            base.attach_object_to_robot(env, obj, task_state)


def _capture_task_object_states(task_module, env, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    task_name = base.normalize_text(getattr(task_module, "TASK_NAME", ""))
    if task_name not in EXPLICIT_OBJECT_STATE_TASKS:
        return {}

    states: dict[str, dict[str, Any]] = {}
    for spec in task_module.build_env_objects(payload):
        object_name = base.normalize_text(spec.get("name"))
        if not object_name:
            continue
        obj = env.scene.object_registry("name", object_name)
        if obj is None:
            continue
        position, orientation, visual_only = base.object_pose(obj)
        states[object_name] = {
            "position": position,
            "orientation": orientation,
            "visual_only": visual_only,
        }
    return states


def _restore_task_object_states(env, object_states: dict[str, dict[str, Any]]) -> None:
    for object_name, state in object_states.items():
        obj = env.scene.object_registry("name", object_name)
        if obj is None:
            continue
        target_position = np.asarray(state["position"], dtype=float).reshape(-1)
        target_orientation = np.asarray(state["orientation"], dtype=float).reshape(-1)
        current_position, current_orientation, current_visual_only = base.object_pose(obj)
        same_position = np.allclose(current_position, target_position, atol=1e-5, rtol=0.0)
        same_orientation = min(
            np.linalg.norm(current_orientation - target_orientation),
            np.linalg.norm(current_orientation + target_orientation),
        ) <= 1e-5
        target_visual_only = bool(state.get("visual_only", current_visual_only))
        if same_position and same_orientation:
            if current_visual_only != target_visual_only:
                base.set_visual_only(obj, target_visual_only)
            obj.keep_still()
            continue
        base.set_object_pose(
            obj,
            target_position,
            target_orientation,
            target_visual_only,
        )


def _set_node_runtime_state(
    node,
    sim_state: Any,
    task_state: dict[str, Any],
    object_states: dict[str, dict[str, Any]] | None = None,
    requires_sim_restore: bool = True,
) -> None:
    node.requires_sim_restore = bool(requires_sim_restore)
    if node.requires_sim_restore:
        # OmniGibson includes tensor views in non-serialized states (for example,
        # rigid-body velocities), so retain an independent snapshot per branch.
        node.sim_state = _clone_branch_value(sim_state)
    else:
        node.sim_state = None
    node.task_state = _clone_task_state(task_state)
    node.object_states = _clone_branch_value(object_states or {})


def _restore_node_runtime_state(task_module, env, node, active_task_state: dict[str, Any] | None) -> dict[str, Any]:
    if not getattr(node, "requires_sim_restore", True):
        return _clone_task_state(getattr(node, "task_state", {}))
    sim_state = getattr(node, "sim_state", None)
    if sim_state is None:
        raise RuntimeError(f"Branch {node.branch_id!r} has no simulator state to restore")
    _cleanup_transient_branch_constraints(task_module, active_task_state)
    og.sim.load_state(sim_state, serialized=False)
    _restore_task_object_states(env, getattr(node, "object_states", {}))
    og.sim.step()
    restored_task_state = _clone_task_state(getattr(node, "task_state", {}))
    _rehydrate_branch_constraints(task_module, env, restored_task_state)
    return restored_task_state


def _node_extra_image_paths(node) -> list[Path]:
    output: list[Path] = []
    for raw_path in getattr(node, "extra_image_paths", []) or []:
        path = Path(raw_path)
        if path.exists():
            output.append(path)
    return output


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    return repr(value)


VERIFIER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_branch_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
    "required": ["selected_branch_ids", "reasoning"],
}


VERIFIER_JSON_OUTPUT_CONTRACT = """Return EXACTLY one JSON object and nothing else:
{
  "selected_branch_ids": ["<branch_id_1>", "...", "<branch_id_N>"],
  "reasoning": "<brief evidence-focused explanation>"
}
Select between 1 and all supplied candidate routes. You decide N from the
visual context: keep only routes that add useful, non-duplicated evidence.
If a fixed Top-K is explicitly configured, select exactly that many (or all
available candidates when fewer are supplied)."""


GENERIC_VERIFIER_PROMPT = """You are a verifier model for an embodied visual exploration task.
Your job is NOT to answer the task yet.
Select a context-dependent number of candidate branch routes whose views
provide the strongest visual evidence for the final answer. Keep between 1
and all supplied candidates; avoid retaining routes that add no new evidence.

This is verifier step {step}. Parent routes from the previous step:
{parent_count}.
Rendered candidate routes this step: {frontier_total}.
Candidate routes supplied in this prompt: {verifier_frontier_size}.
Candidate truncation applied before verification: {yes/no}.

Only choose branch ids that appear in [VERIFIER CANDIDATE ...] labels.
Prefer routes that improve task-relevant visibility, reduce ambiguity, and
make the key spatial or physical relation easier to inspect.
Avoid near-duplicate routes unless each duplicate provides materially useful
new evidence."""


def _format_verifier_prompt_template(
    template: str,
    task_context: str,
    config,
    step: int,
    parent_count: int,
    frontier_total: int,
    verifier_frontier_size: int,
    verifier_truncated: bool,
) -> str:
    values = {
        "verifier_top_k": (
            str(config.verifier_top_k)
            if getattr(config, "verifier_top_k", None) is not None
            else "a context-dependent number"
        ),
        "step": step,
        "parent_count": parent_count,
        "frontier_total": frontier_total,
        "verifier_frontier_size": verifier_frontier_size,
        "yes/no": "yes" if verifier_truncated else "no",
        "task_context": task_context,
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--verifier-provider", choices=["gemini", "gpt", "qwen", "glm"], default="qwen")
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--verifier-api-key", default=None)
    parser.add_argument("--start-question-index", type=int, default=0)
    parser.add_argument("--end-question-index", type=int, default=None)
    parser.add_argument("--batch-range-start-index", type=int, default=None)
    parser.add_argument("--batch-range-end-index", type=int, default=None)
    parser.add_argument("--batch-progress-file", default=None)
    parser.add_argument(
        "--run-all-questions",
        action="store_true",
        help="Run all questions from a summary metadata file in one Python process and only shut down OmniGibson once at the end.",
    )
    parser.add_argument(
        "--fixed-verifier-top-k",
        type=int,
        default=None,
        help="Force the verifier to retain this many routes at every exploration step.",
    )
    parser.add_argument(
        "--verifier-max-images",
        type=int,
        default=0,
        help="Optional cap on how many frontier images are sent to the verifier per step. <=0 means unlimited.",
    )
    parser.add_argument(
        "--final-max-images",
        type=int,
        default=0,
        help="Optional cap on how many verifier-selected images are sent to the final answer model. <=0 means unlimited.",
    )
    parser.add_argument(
        "--answer-evidence-mode",
        choices=[answer_evidence.ALL, answer_evidence.TOP1_ROUTE],
        default=answer_evidence.TOP1_ROUTE,
        help="Send all verifier-selected images or only the final rank-1 branch's ancestor route to the answer model.",
    )
    parser.add_argument("--answer-review", type=int, choices=[0, 1], default=0)
    extra_args, remaining = parser.parse_known_args(argv)
    config = base.parse_args(remaining)
    setattr(config, "verifier_provider", extra_args.verifier_provider)
    setattr(config, "verifier_model", extra_args.verifier_model)
    setattr(config, "verifier_api_key", extra_args.verifier_api_key)
    setattr(config, "start_question_index", max(0, int(extra_args.start_question_index)))
    setattr(
        config,
        "end_question_index",
        None if extra_args.end_question_index is None else max(0, int(extra_args.end_question_index)),
    )
    setattr(
        config,
        "batch_range_start_index",
        None
        if extra_args.batch_range_start_index is None
        else max(0, int(extra_args.batch_range_start_index)),
    )
    setattr(
        config,
        "batch_range_end_index",
        None
        if extra_args.batch_range_end_index is None
        else max(0, int(extra_args.batch_range_end_index)),
    )
    setattr(
        config,
        "batch_progress_file",
        Path(extra_args.batch_progress_file).expanduser().resolve() if extra_args.batch_progress_file else None,
    )
    setattr(config, "run_all_questions", bool(extra_args.run_all_questions))
    fixed_top_k = extra_args.fixed_verifier_top_k
    if fixed_top_k is not None:
        fixed_top_k = max(1, int(fixed_top_k))
    setattr(config, "fixed_verifier_top_k", fixed_top_k)
    setattr(config, "verifier_max_images", int(extra_args.verifier_max_images))
    setattr(config, "final_max_images", int(extra_args.final_max_images))
    setattr(config, "answer_evidence_mode", extra_args.answer_evidence_mode)
    setattr(config, "answer_review", bool(extra_args.answer_review))
    return config


def _metadata_question_count(metadata_path: Path) -> int:
    if metadata_path.suffix.lower() == ".jsonl":
        raise ValueError(
            "questions.jsonl is an HF export artifact, not a runner input. "
            "Use dataset/json_clean/*.json or dataset/json_clean/<Big Task>.json."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if base.is_hf_question_row(metadata):
        return 1
    if isinstance(metadata, dict) and isinstance(metadata.get("json_paths"), list):
        return len(metadata["json_paths"])
    if isinstance(metadata, list):
        return len(metadata)
    return 1


def _resolve_model_names(config, task_module) -> tuple[str, str]:
    if config.provider == "gpt":
        default_model = "gpt-5"
    elif config.provider == "qwen":
        default_model = "qwen"
    elif config.provider == "glm":
        default_model = "glm-4.5v"
    else:
        default_model = getattr(task_module, "DEFAULT_MODEL", "gemini-2.5-flash")
    model_name = config.model or default_model
    verifier_model_name = config.verifier_model or model_name
    return model_name, verifier_model_name


def _build_model_clients(config, task_module):
    model_name, verifier_model_name = _resolve_model_names(config, task_module)
    answer_client = base.build_model_client(config.provider, config.api_key, model_name)
    if (
        config.verifier_provider == config.provider
        and verifier_model_name == model_name
        and (config.verifier_api_key or config.api_key) == config.api_key
    ):
        verifier_client = answer_client
    else:
        verifier_client = base.build_model_client(
            config.verifier_provider,
            config.verifier_api_key or config.api_key,
            verifier_model_name,
        )
    return answer_client, verifier_client, model_name, verifier_model_name


def _reset_og_runtime(keep_app_alive: bool) -> None:
    sim = getattr(og, "sim", None)

    try:
        if sim is not None and hasattr(sim, "stop"):
            sim.stop()
    except Exception:
        pass

    try:
        if sim is not None and hasattr(sim, "_partial_clear"):
            sim._partial_clear()
    except Exception:
        pass

    # Some failures happen during scene loading before the normal simulator clear path
    # completes, so we also clear singleton caches explicitly.
    try:
        from omnigibson.utils.python_utils import clear as clear_python_utils

        clear_python_utils()
    except Exception:
        pass

    try:
        from omnigibson.utils.usd_utils import clear as clear_usd_utils

        clear_usd_utils()
    except Exception:
        pass

    try:
        import omnigibson.lazy as lazy

        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        if stage is not None:
            lazy.isaacsim.core.utils.stage.close_stage()
        lazy.isaacsim.core.api.SimulationContext.clear_instance()
    except Exception:
        pass

    try:
        og.sim = None
    except Exception:
        pass

    if not keep_app_alive:
        try:
            if getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass


def _cleanup_env(task_module, env, payload: dict[str, Any], task_state: dict[str, Any], shutdown_app: bool) -> None:
    try:
        base.task_cleanup_runtime(task_module, env, payload, task_state)
    except Exception:
        pass
    try:
        if getattr(og, "sim", None) is not None:
            og.sim.stop()
    except Exception:
        pass
    try:
        if env is not None and hasattr(env, "close"):
            env.close()
    except Exception:
        pass
    _reset_og_runtime(keep_app_alive=not shutdown_app)


def _write_batch_progress(
    progress_path: Path | None,
    *,
    status: str,
    total_questions: int,
    start_question_index: int,
    end_question_index: int,
    active_question_index: int | None,
    next_question_index: int,
    success_count: int,
    fail_count: int,
    failed_indices: list[int],
    error: str | None = None,
) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "total_questions": int(total_questions),
        "start_question_index": int(start_question_index),
        "end_question_index": int(end_question_index),
        "range_total_questions": max(0, int(end_question_index) - int(start_question_index)),
        "active_question_index": None if active_question_index is None else int(active_question_index),
        "next_question_index": int(next_question_index),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "failed_indices": sorted({int(index) for index in failed_indices}),
    }
    if error:
        payload["error"] = error
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_batch_progress(progress_path: Path | None) -> dict[str, Any]:
    if progress_path is None or not progress_path.is_file():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _existing_result_state_between(
    config,
    task_module,
    start_index: int,
    end_index: int,
) -> tuple[int, list[int]]:
    results_root = config.results_root / PIPELINE_VARIANT
    completed = 0
    failed_indices: list[int] = []
    for question_index in range(max(0, start_index), max(0, end_index)):
        try:
            selected = base.select_question(config.metadata, question_index, config.json_root)
            output_path = base.output_path_for(
                task_module,
                selected.payload,
                selected.source_path,
                results_root,
            )
        except Exception:
            failed_indices.append(question_index)
            continue
        if output_path.is_file():
            completed += 1
        else:
            failed_indices.append(question_index)
    return completed, failed_indices


def _initial_batch_counts(
    config,
    task_module,
    total_questions: int,
    range_start_question_index: int,
    start_question_index: int,
    end_question_index: int,
) -> tuple[int, int, list[int]]:
    previous = _read_batch_progress(getattr(config, "batch_progress_file", None))
    try:
        previous_total = int(previous.get("total_questions", -1))
        previous_range_start = int(previous.get("start_question_index", -1))
        previous_range_end = int(previous.get("end_question_index", -1))
        previous_next = int(previous.get("next_question_index", -1))
        previous_success = int(previous.get("success_count", 0))
        previous_fail = int(previous.get("fail_count", 0))
    except (TypeError, ValueError):
        previous_total = previous_range_start = previous_range_end = previous_next = -1
        previous_success = previous_fail = 0

    raw_failed_indices = previous.get("failed_indices", [])
    if isinstance(raw_failed_indices, list):
        try:
            previous_failed_indices = sorted(
                {
                    int(index)
                    for index in raw_failed_indices
                    if range_start_question_index <= int(index) < start_question_index
                }
            )
        except (TypeError, ValueError):
            previous_failed_indices = []
    else:
        previous_failed_indices = []

    if (
        previous_total == total_questions
        and previous_range_start == range_start_question_index
        and previous_range_end == end_question_index
        and previous_next == start_question_index
        and previous_success >= 0
        and previous_fail >= 0
        and previous_success + previous_fail == start_question_index - range_start_question_index
        and len(previous_failed_indices) == previous_fail
    ):
        return previous_success, previous_fail, previous_failed_indices

    # Older runs reset counters after every process restart. Reconstruct their
    # cumulative counts from result files for indices already passed.
    success_count, failed_indices = _existing_result_state_between(
        config,
        task_module,
        range_start_question_index,
        start_question_index,
    )
    return success_count, len(failed_indices), failed_indices


def _verifier_history_summary(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for item in history:
        lines.append(
            f"Step {item['step']}: frontier={item.get('frontier_size', 0)} "
            f"verifier_views={item.get('verifier_frontier_size', 0)} "
            f"requested_top_k={item.get('requested_top_k', 0)} "
            f"selected={item.get('selected_count', 0)} "
            f"truncated={bool(item.get('verifier_truncated'))} "
            f"selected_branch_ids={item.get('selected_branch_ids', [])} "
            f"checkpoint_answer={item.get('checkpoint_answer', '')} "
            f"checkpoint_confidence={float(item.get('checkpoint_confidence', 0.0)):.2f} "
            f"checkpoint_conclusive={bool(item.get('checkpoint_conclusive', False))} "
            f"verifier_reasoning={item.get('verifier_reasoning', '')}"
        )
    return "\n".join(lines)


def _final_history_summary(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for item in history:
        lines.append(
            f"Step {item['step']}: selected_branch_ids={item.get('selected_branch_ids', [])} "
            f"checkpoint_answer={item.get('checkpoint_answer', '')} "
            f"checkpoint_confidence={float(item.get('checkpoint_confidence', 0.0)):.2f} "
            f"checkpoint_conclusive={bool(item.get('checkpoint_conclusive', False))}"
        )
    return "\n".join(lines)


def _collect_verifier_contents(
    frontier: list[branch_base.BranchNode],
    history: list[dict[str, Any]],
    prompt: str,
    reference_image_paths: list[Path] | None = None,
    frontier_total: int | None = None,
    frontier_truncated: bool = False,
) -> list[Any]:
    contents: list[Any] = []
    # The current frontier already carries complete branch lineage through
    # each candidate's branch id, parent id, and depth. Previous selected
    # routes and checkpoint answers are intentionally omitted to keep the
    # verifier focused on the current visual evidence and avoid anchoring.
    for index, reference in enumerate(reference_image_paths or [], start=1):
        if reference.exists():
            label = "[QUESTION REFERENCE IMAGE - dataset render]" if len(reference_image_paths or []) == 1 else f"[QUESTION REFERENCE IMAGE {index}]"
            contents.append(label)
            contents.append(reference)
    if frontier_total is not None:
        contents.append(
            "Verifier candidate summary: "
            f"total_views={frontier_total}, supplied_views={len(frontier)}, truncated={frontier_truncated}."
        )
    for candidate_index, node in enumerate(frontier, start=1):
        if node.image_path is None or not node.image_path.exists():
            continue
        contents.append(
            f"[VERIFIER CANDIDATE index={candidate_index} depth={node.depth} "
            f"branch={node.branch_id} action={node.action} "
            f"parent={node.parent_branch_id or 'root'}]"
        )
        contents.append(node.image_path)
        for extra_index, extra_path in enumerate(_node_extra_image_paths(node), start=1):
            contents.append(
                f"[VERIFIER ACTION FRAME depth={node.depth} branch={node.branch_id} "
                f"action={node.action} frame={extra_index}]"
            )
            contents.append(extra_path)
    contents.append(prompt)
    return contents


def _collect_final_contents(
    selected_groups: list[list[branch_base.BranchNode]],
    history: list[dict[str, Any]],
    prompt: str,
    reference_image_paths: list[Path] | None = None,
) -> list[Any]:
    contents: list[Any] = []
    history_summary = _final_history_summary(history)
    if history_summary:
        contents.append("Verifier-selected route history:\n" + history_summary)
    for index, reference in enumerate(reference_image_paths or [], start=1):
        if reference.exists():
            label = "[QUESTION REFERENCE IMAGE - dataset render]" if len(reference_image_paths or []) == 1 else f"[QUESTION REFERENCE IMAGE {index}]"
            contents.append(label)
            contents.append(reference)
    for group in selected_groups:
        for rank, node in enumerate(group, start=1):
            if node.image_path is None or not node.image_path.exists():
                continue
            contents.append(
                f"[FINAL EVIDENCE step={node.depth} rank={rank} depth={node.depth} "
                f"branch={node.branch_id} action={node.action}]"
            )
            contents.append(node.image_path)
            for extra_index, extra_path in enumerate(_node_extra_image_paths(node), start=1):
                contents.append(
                    f"[FINAL ACTION FRAME step={node.depth} rank={rank} depth={node.depth} "
                    f"branch={node.branch_id} action={node.action} frame={extra_index}]"
                )
                contents.append(extra_path)
    contents.append(prompt)
    return contents


def _build_default_verifier_prompt(
    task_module,
    payload: dict[str, Any],
    config,
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    step: int,
    parent_count: int,
    frontier_total: int,
    verifier_frontier_size: int,
    verifier_truncated: bool,
) -> str:
    task_context = base.task_build_system_prompt(
        task_module,
        payload,
        config.threshold,
        config.min_steps,
        camera_info,
        task_state,
    )
    template = "\n\n".join(
        [
            GENERIC_VERIFIER_PROMPT,
            VERIFIER_JSON_OUTPUT_CONTRACT,
            "Task context:\n{task_context}",
        ]
    )
    return _format_verifier_prompt_template(
        template,
        task_context,
        config,
        step,
        parent_count,
        frontier_total,
        verifier_frontier_size,
        verifier_truncated,
    )


def _build_verifier_prompt(
    task_module,
    payload: dict[str, Any],
    config,
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    step: int,
    parent_count: int,
    frontier_total: int,
    verifier_frontier_size: int,
    verifier_truncated: bool,
) -> str:
    hook = getattr(task_module, "build_verifier_prompt", None)
    prompt = None
    if hook is not None:
        candidate_prompt = base.call_with_supported_args(
            hook,
            payload,
            config.threshold,
            config.min_steps,
            camera_info,
            step=step,
            parent_count=parent_count,
            frontier_total=frontier_total,
            verifier_frontier_size=verifier_frontier_size,
            verifier_truncated=verifier_truncated,
            verifier_top_k=getattr(config, "verifier_top_k", None),
            task_state=task_state,
        )
        if isinstance(candidate_prompt, str) and candidate_prompt.strip():
            prompt = candidate_prompt
    if prompt is None:
        prompt = _build_default_verifier_prompt(
            task_module,
            payload,
            config,
            camera_info,
            task_state,
            step,
            parent_count,
            frontier_total,
            verifier_frontier_size,
            verifier_truncated,
        )
    sections = [prompt]
    sections.extend(
        [
            "",
            "Mandatory selection consistency rules:",
            "  - Return the exact branch= value from each selected [VERIFIER CANDIDATE ...] label; do not return only numeric indexes.",
            "  - Every evidence claim in reasoning must come from a selected branch.",
            "  - If reasoning cites a [VERIFIER ACTION FRAME ...] or any post-action outcome, select that frame's corresponding branch.",
            "  - Do not cite useful evidence from a route while leaving that route unselected.",
        ]
    )
    return "\n".join(sections)


def _build_final_answer_prompt(
    task_module,
    payload: dict[str, Any],
    config,
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    selected_groups: list[list[branch_base.BranchNode]],
    current_step: int,
    action_frame_windowed: bool,
    force_choice: bool,
) -> str:
    if force_choice:
        base_prompt = base.task_build_force_choice_prompt(task_module, payload, camera_info, task_state)
        checkpoint_instruction = (
            "  - This is the final checkpoint because the exploration budget is exhausted. "
            "Return the required conclusive task answer."
        )
    else:
        base_prompt = base.task_build_system_prompt(
            task_module,
            payload,
            config.threshold,
            config.min_steps,
            camera_info,
            task_state,
        )
        checkpoint_instruction = (
            "  - This is an interim checkpoint, not the final decision. Exploration is still available. "
            "Report an answer and confidence calibrated to the selected evidence."
        )
    total_images = sum(
        1 + len(_node_extra_image_paths(node))
        for group in selected_groups
        for node in group
        if node.image_path is not None and node.image_path.exists()
    )
    selected_counts = [len(group) for group in selected_groups]
    return "\n".join(
        [
            base_prompt,
            "",
            "Pipeline override for verifier-guided selection:",
            f"  - This is answer checkpoint {current_step} of at most {config.max_steps} exploration steps.",
            checkpoint_instruction,
            f"  - Actual retained-route counts by supplied step: {selected_counts}.",
            f"  - Selected evidence groups supplied in this request: {len(selected_groups)}.",
            f"  - Answer evidence mode: {answer_evidence.mode(config)}.",
            f"  - Older selected image groups omitted because of heavy action-frame evidence: {'yes' if action_frame_windowed else 'no'}.",
            f"  - Total answer evidence images supplied: {total_images}.",
            f"  - Report calibrated confidence; only confidence > {config.threshold:.2f} with a conclusive answer stops exploration.",
            "  - Use all provided evidence together to produce the final answer.",
            "  - Base the answer only on the supplied FINAL EVIDENCE and FINAL ACTION FRAME images; unselected verifier candidates are not evidence.",
            "  - Ignore route selection and focus only on the task answer now.",
            "  - Do not select an action in this checkpoint; report only the task answer, confidence, and reasoning.",
        ]
    )



def _expand_frontier(
    task_module,
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    frontier: list[branch_base.BranchNode],
    step: int,
    step_image_dir: Path,
    task_state: dict[str, Any],
    branch_actions: tuple[str, ...],
) -> list[branch_base.BranchNode]:
    step_dir = step_image_dir / f"step_{step:03d}"
    expanded: list[branch_base.BranchNode] = []
    active_task_state = task_state
    for parent in frontier:
        for action in branch_actions:
            branch_task_state = _restore_node_runtime_state(task_module, env, parent, active_task_state)
            branch_id = action if parent.branch_id == "root" else f"{parent.branch_id}/{action}"
            image_path = step_dir / f"{branch_base._safe_branch_filename(branch_id)}.png"
            action_frame_dir = step_dir / "_action_frames" / branch_base._safe_branch_filename(branch_id)
            next_pos, next_quat, action_info = base.execute_action(
                task_module,
                env,
                payload,
                camera_info,
                parent.position,
                parent.quaternion_xyzw,
                action,
                task_state=branch_task_state,
                step=step,
                step_image_dir=action_frame_dir,
            )
            rendered_path = base.task_capture_image(
                task_module,
                env,
                payload,
                camera_info,
                next_pos,
                next_quat,
                image_path,
                branch_task_state,
            )
            node = branch_base.BranchNode(
                branch_id=branch_id,
                parent_branch_id=parent.branch_id,
                depth=step,
                action=action,
                position=next_pos,
                quaternion_xyzw=next_quat,
                image_path=Path(rendered_path),
            )
            node.parent_node = parent
            node.action_info = _json_safe(action_info)
            node.extra_image_paths = [
                str(path)
                for path in (action_info.get("extra_image_paths", []) if isinstance(action_info, dict) else [])
                if Path(path).exists()
            ]
            expanded.append(node)
            active_task_state = branch_task_state
    # The candidate states are intentionally not retained. Remove any grasp
    # constraint left by the final candidate before selected branches are
    # replayed from their parent snapshots.
    _cleanup_transient_branch_constraints(task_module, active_task_state)
    return expanded


def _materialize_selected_nodes(
    task_module,
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    selected_nodes: list[branch_base.BranchNode],
    step: int,
    step_image_dir: Path,
    active_task_state: dict[str, Any],
) -> dict[str, Any]:
    """Replay only verifier-selected children and retain their branch states."""
    step_dir = step_image_dir / f"step_{step:03d}"
    current_task_state = active_task_state
    consumed_parents: dict[int, Any] = {}
    for node in selected_nodes:
        parent = getattr(node, "parent_node", None)
        if parent is None:
            raise RuntimeError(f"Selected branch {node.branch_id!r} lost its parent state")
        consumed_parents[id(parent)] = parent
        branch_task_state = _restore_node_runtime_state(task_module, env, parent, current_task_state)
        action_frame_dir = step_dir / "_action_frames" / branch_base._safe_branch_filename(node.branch_id)
        next_pos, next_quat, action_info = base.execute_action(
            task_module,
            env,
            payload,
            camera_info,
            parent.position,
            parent.quaternion_xyzw,
            node.action,
            task_state=branch_task_state,
            step=step,
            step_image_dir=action_frame_dir,
        )
        node.position = next_pos
        node.quaternion_xyzw = next_quat
        node.action_info = _json_safe(action_info)
        # Keep the retained branch image aligned with the replayed simulator state.
        node.image_path = Path(
            base.task_capture_image(
                task_module,
                env,
                payload,
                camera_info,
                next_pos,
                next_quat,
                node.image_path,
                branch_task_state,
            )
        )
        replay_extra_paths = [
            str(path)
            for path in (action_info.get("extra_image_paths", []) if isinstance(action_info, dict) else [])
            if Path(path).exists()
        ]
        if replay_extra_paths:
            node.extra_image_paths = replay_extra_paths
        _set_node_runtime_state(
            node,
            og.sim.dump_state(serialized=False) if getattr(parent, "requires_sim_restore", True) else None,
            branch_task_state,
            _capture_task_object_states(task_module, env, payload)
            if getattr(parent, "requires_sim_restore", True)
            else {},
            requires_sim_restore=getattr(parent, "requires_sim_restore", True),
        )
        # Keep the lightweight parent pointer for answer-evidence route
        # reconstruction. Simulator/object/task state is cleared below, so
        # retaining this lineage does not keep the heavy runtime snapshot.
        current_task_state = branch_task_state
    # Historical selected nodes are retained for their images and lightweight
    # lineage, not their full simulator snapshots. Only the new frontier needs
    # runtime state.
    for parent in consumed_parents.values():
        parent.sim_state = None
        parent.task_state = {}
    return current_task_state


def _is_conclusive_answer(answer: Any) -> bool:
    if isinstance(answer, (dict, list)):
        return bool(answer)
    normalized = " ".join(str(answer or "").strip().lower().replace("_", " ").replace("-", " ").split())
    return normalized not in {"", "not sure", "unsure", "unknown"}


def _is_task_conclusive_answer(
    task_module,
    answer: Any,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
) -> bool:
    hook = getattr(task_module, "is_conclusive_answer", None)
    if hook is not None:
        try:
            return bool(
                base.call_with_supported_args(
                    hook,
                    answer,
                    payload,
                    camera_info,
                    task_state=task_state,
                )
            )
        except Exception:
            return False
    return _is_conclusive_answer(answer)


def _normalize_line_review_answer(task_module, value: Any) -> str:
    normalizer = getattr(task_module, "normalize_answer", None)
    if normalizer is None:
        return "not sure"
    try:
        return str(normalizer(value))
    except Exception:
        return "not sure"


def _review_line_checkpoint(
    task_module,
    answer_client,
    payload: dict[str, Any],
    config,
    selected_groups: list[list[branch_base.BranchNode]],
    history: list[dict[str, Any]],
    reference_image_paths: list[Path],
    checkpoint_answer: dict[str, Any],
    force_choice: bool,
) -> dict[str, Any]:
    original_answer = _normalize_line_review_answer(task_module, checkpoint_answer.get("answer"))
    original_reasoning = base.normalize_text(checkpoint_answer.get("reasoning"))
    try:
        object_names = ", ".join(task_module.object_labels(payload))
    except Exception:
        object_names = "the three queried objects"
    allowed_answers = "yes or no" if force_choice else "yes, no, or not sure"
    review_prompt = "\n".join(
        [
            "You are reviewing a Linear Alignment answer using exactly the supplied visual evidence.",
            f"Objects: {object_names}.",
            "First make your own spatial judgment about whether the three object centers form a straight line on the floor.",
            "Then classify which answer the original reasoning semantically supports. Do not compare wording or writing style.",
            "Treat perspective-only apparent alignment as insufficient evidence and use all supplied views.",
            f"Original answer: {original_answer}",
            f"Original reasoning: {original_reasoning}",
            f"The independent_answer must be one of: {allowed_answers}.",
            "original_reasoning_implies must be yes, no, or unclear.",
            "original_reasoning_consistency must be consistent, contradictory, or unclear.",
            "Output exactly one JSON object in this field order:",
            '{"independent_reasoning":"<brief visual judgment>","independent_answer":"<yes/no/not sure>",'
            '"original_reasoning_implies":"<yes/no/unclear>",'
            '"original_reasoning_consistency":"<consistent/contradictory/unclear>",'
            '"confidence":<float 0.0-1.0>}',
        ]
    )
    review_record: dict[str, Any] = {
        "enabled": True,
        "original_answer": original_answer,
        "original_reasoning": original_reasoning,
        "adjudication_triggered": False,
        "decision": "keep_original",
    }
    try:
        review_parsed, review_raw, review_finish_reason = answer_client.generate_json(
            contents=_collect_final_contents(
                selected_groups,
                history,
                review_prompt,
                reference_image_paths=reference_image_paths,
            ),
            system_instruction="Return exactly one valid JSON object and nothing else.",
            response_schema=None,
            max_output_tokens=max(320, min(config.max_new_tokens, 640)),
            temperature=0.0,
            top_p=1.0,
            fallback={
                "independent_reasoning": "review parse fallback",
                "independent_answer": "not sure",
                "original_reasoning_implies": "unclear",
                "original_reasoning_consistency": "unclear",
                "confidence": 0.0,
            },
        )
        independent_answer = _normalize_line_review_answer(task_module, review_parsed.get("independent_answer"))
        implied_raw = base.normalize_text(review_parsed.get("original_reasoning_implies")).lower()
        implied_answer = implied_raw if implied_raw in {"yes", "no"} else "unclear"
        consistency = base.normalize_text(review_parsed.get("original_reasoning_consistency")).lower()
        if consistency not in {"consistent", "contradictory", "unclear"}:
            consistency = "unclear"
        try:
            review_confidence = max(0.0, min(1.0, float(review_parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            review_confidence = 0.0
        review_record.update(
            {
                "independent_reasoning": base.normalize_text(review_parsed.get("independent_reasoning")),
                "independent_answer": independent_answer,
                "original_reasoning_implies": implied_answer,
                "original_reasoning_consistency": consistency,
                "confidence": review_confidence,
                "raw_output": review_raw,
                "finish_reason": review_finish_reason,
            }
        )
    except Exception as exc:
        review_record.update(
            {
                "decision": "keep_original_after_review_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        checkpoint_answer["answer_review"] = review_record
        return checkpoint_answer

    independent_answer = review_record["independent_answer"]
    implied_answer = review_record["original_reasoning_implies"]
    consistency = review_record["original_reasoning_consistency"]
    answer_disagreement = independent_answer != original_answer
    reasoning_disagreement = implied_answer in {"yes", "no"} and implied_answer != original_answer
    needs_adjudication = answer_disagreement or reasoning_disagreement or consistency == "contradictory"

    if not needs_adjudication:
        if consistency == "unclear" or implied_answer == "unclear":
            reviewer_confidence = float(review_record.get("confidence", 0.0))
            if reviewer_confidence > 0.0:
                checkpoint_answer["confidence"] = min(float(checkpoint_answer.get("confidence", 0.0)), reviewer_confidence)
            review_record["decision"] = "keep_original_with_conservative_confidence"
        checkpoint_answer["answer_review"] = review_record
        return checkpoint_answer

    review_record["adjudication_triggered"] = True
    adjudication_prompt = "\n".join(
        [
            "You are the final adjudicator for a disputed Linear Alignment checkpoint.",
            "Re-examine the supplied images. Decide whether the three object centers form a straight line on the floor.",
            "Do not use class frequency, scene identity, or any information outside the supplied evidence.",
            f"Original answer: {original_answer}",
            f"Original reasoning: {original_reasoning}",
            f"Reviewer independent answer: {independent_answer}",
            f"Reviewer independent reasoning: {review_record.get('independent_reasoning', '')}",
            f"Reviewer says original reasoning implies: {implied_answer}",
            f"Return {allowed_answers}.",
            "Output exactly one JSON object with reasoning before answer:",
            '{"reasoning":"<brief evidence-based adjudication>","answer":"<yes/no/not sure>","confidence":<float 0.0-1.0>}',
        ]
    )
    try:
        adjudicated_parsed, adjudicated_raw, adjudicated_finish_reason = answer_client.generate_json(
            contents=_collect_final_contents(
                selected_groups,
                history,
                adjudication_prompt,
                reference_image_paths=reference_image_paths,
            ),
            system_instruction="Return exactly one valid JSON object and nothing else.",
            response_schema=None,
            max_output_tokens=max(256, min(config.max_new_tokens, 512)),
            temperature=0.0,
            top_p=1.0,
            fallback={
                "reasoning": "adjudication parse fallback",
                "answer": original_answer,
                "confidence": checkpoint_answer.get("confidence", 0.0),
            },
        )
        adjudicated = base.task_parse_model_output(task_module, adjudicated_parsed, payload, {})
        adjudicated_answer = _normalize_line_review_answer(task_module, adjudicated.get("answer"))
        if force_choice and adjudicated_answer == "not sure":
            adjudicated_answer = original_answer
        checkpoint_answer.update(
            {
                "answer": adjudicated_answer,
                "confidence": max(0.0, min(1.0, float(adjudicated.get("confidence", 0.0)))),
                "reasoning": base.normalize_text(adjudicated.get("reasoning")) or original_reasoning,
            }
        )
        review_record.update(
            {
                "decision": "use_adjudicated_answer",
                "adjudicated_answer": adjudicated_answer,
                "adjudicated_reasoning": checkpoint_answer["reasoning"],
                "adjudicated_confidence": checkpoint_answer["confidence"],
                "adjudicator_raw_output": adjudicated_raw,
                "adjudicator_finish_reason": adjudicated_finish_reason,
            }
        )
    except Exception as exc:
        review_record.update(
            {
                "decision": "keep_original_after_adjudication_error",
                "adjudication_error": f"{type(exc).__name__}: {exc}",
            }
        )
    checkpoint_answer["answer_review"] = review_record
    return checkpoint_answer


def _request_checkpoint_answer(
    task_module,
    answer_client,
    payload: dict[str, Any],
    config,
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    selected_groups: list[list[branch_base.BranchNode]],
    history: list[dict[str, Any]],
    reference_image_paths: list[Path],
    step: int,
) -> tuple[dict[str, Any], str, str | None, bool]:
    selected_answer_groups, answer_history = answer_evidence.select(config, selected_groups, history)
    if answer_evidence.mode(config) == answer_evidence.TOP1_ROUTE:
        sampled_groups = selected_answer_groups
        action_frame_windowed = False
        images_truncated = False
    else:
        evidence_groups, action_frame_windowed = _limit_heavy_action_frame_groups(selected_answer_groups)
        sampled_groups, max_images_truncated = _sample_final_groups(evidence_groups, config.final_max_images)
        images_truncated = action_frame_windowed or max_images_truncated
    setattr(
        config,
        "_answer_evidence_branch_ids",
        [node.branch_id for group in sampled_groups for node in group],
    )
    setattr(
        config,
        "_answer_evidence_image_count",
        sum(
            1 + len(_node_extra_image_paths(node))
            for group in sampled_groups
            for node in group
            if node.image_path is not None and node.image_path.exists()
        ),
    )
    task_name = base.normalize_text(getattr(task_module, "TASK_NAME", "")).lower()
    force_choice = task_name == "occlusion" or step >= config.max_steps
    prompt = _build_final_answer_prompt(
        task_module,
        payload,
        config,
        camera_info,
        task_state,
        sampled_groups,
        step,
        action_frame_windowed,
        force_choice=force_choice,
    )
    parsed, raw_text, finish_reason = answer_client.generate_json(
        contents=_collect_final_contents(
            sampled_groups,
            answer_history,
            prompt,
            reference_image_paths=reference_image_paths,
        ),
        system_instruction="Return exactly one valid JSON object and nothing else.",
        response_schema=None,
        max_output_tokens=max(256, min(config.max_new_tokens, 512)),
        temperature=0.0,
        top_p=1.0,
        fallback={"answer": "not sure", "confidence": 0.0, "reasoning": "checkpoint parse fallback"},
    )
    parsed = base.task_parse_model_output(task_module, parsed, payload, task_state)
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    answer = parsed.get("answer", "not sure")
    checkpoint_answer = {
        "answer": answer,
        "answer_step": step,
        "confidence": confidence,
        "reasoning": base.normalize_text(parsed.get("reasoning")) or "no reasoning provided",
        "steps": step,
        "stopped_by": "checkpoint",
    }
    if bool(getattr(config, "answer_review", False)) and task_name == "line":
        checkpoint_answer = _review_line_checkpoint(
            task_module,
            answer_client,
            payload,
            config,
            sampled_groups,
            answer_history,
            reference_image_paths,
            checkpoint_answer,
            force_choice,
        )
    return checkpoint_answer, raw_text, finish_reason, images_truncated


def _normalize_selected_branch_ids(parsed: dict[str, Any]) -> list[str]:
    raw = parsed.get("selected_branch_ids")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split("|") if part.strip()]
    return []


def _select_verifier_frontier(
    frontier: list[branch_base.BranchNode],
    max_images: int,
) -> tuple[list[branch_base.BranchNode], bool]:
    return branch_base._sample_frontier(frontier, max_images)


def _choose_selected_nodes(
    frontier: list[branch_base.BranchNode],
    parsed: dict[str, Any],
    top_k: int | None = None,
) -> list[branch_base.BranchNode]:
    by_id = {node.branch_id: node for node in frontier}
    chosen: list[branch_base.BranchNode] = []
    seen: set[str] = set()
    for branch_id in _normalize_selected_branch_ids(parsed):
        node = by_id.get(branch_id)
        if node is None and branch_id.isdigit():
            candidate_index = int(branch_id) - 1
            if 0 <= candidate_index < len(frontier):
                node = frontier[candidate_index]
        if node is None or node.branch_id in seen:
            continue
        chosen.append(node)
        seen.add(node.branch_id)
        if top_k is not None and len(chosen) >= top_k:
            return chosen
    if chosen:
        return chosen if top_k is None else chosen[:top_k]
    for node in frontier:
        if node.branch_id in seen:
            continue
        chosen.append(node)
        seen.add(node.branch_id)
        if top_k is not None and len(chosen) >= top_k:
            break
    return chosen


def _flatten_selected_groups(
    selected_groups: list[list[branch_base.BranchNode]],
) -> list[branch_base.BranchNode]:
    flat: list[branch_base.BranchNode] = []
    for group in selected_groups:
        flat.extend(group)
    return flat


def _sample_final_groups(
    selected_groups: list[list[branch_base.BranchNode]],
    final_max_images: int,
) -> tuple[list[list[branch_base.BranchNode]], bool]:
    if final_max_images <= 0:
        return [list(group) for group in selected_groups], False
    total_images = sum(len(group) for group in selected_groups)
    if total_images <= final_max_images:
        return [list(group) for group in selected_groups], False

    remaining = final_max_images
    sampled_groups: list[list[branch_base.BranchNode]] = []
    group_count = len(selected_groups)
    for index, group in enumerate(selected_groups):
        groups_left = group_count - index
        keep = max(1, remaining // groups_left)
        keep = min(keep, len(group))
        sampled, _ = branch_base._sample_frontier(group, keep)
        sampled_groups.append(sampled)
        remaining -= len(sampled)
    return sampled_groups, True


def _limit_heavy_action_frame_groups(
    selected_groups: list[list[branch_base.BranchNode]],
) -> tuple[list[list[branch_base.BranchNode]], bool]:
    extra_frame_count = sum(
        len(_node_extra_image_paths(node))
        for group in selected_groups
        for node in group
    )
    if (
        extra_frame_count < HEAVY_ACTION_FRAME_THRESHOLD
        or len(selected_groups) <= HEAVY_ACTION_FRAME_STEP_WINDOW
    ):
        return [list(group) for group in selected_groups], False
    return [list(group) for group in selected_groups[-HEAVY_ACTION_FRAME_STEP_WINDOW:]], True


def _archive_selected_nodes(
    selected_nodes: list[branch_base.BranchNode],
    selected_image_dir: Path,
    step: int,
) -> list[str]:
    archived_paths: list[str] = []
    step_dir = selected_image_dir / f"step_{step:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for rank, node in enumerate(selected_nodes, start=1):
        source = node.image_path
        if source is None or not source.exists():
            continue
        filename = f"rank_{rank:02d}__{branch_base._safe_branch_filename(node.branch_id)}{source.suffix or '.png'}"
        destination = step_dir / filename
        shutil.copy2(source, destination)
        archived_paths.append(str(destination))
        for extra_index, extra_path in enumerate(_node_extra_image_paths(node), start=1):
            extra_name = (
                f"rank_{rank:02d}__{branch_base._safe_branch_filename(node.branch_id)}"
                f"__frame_{extra_index:03d}{extra_path.suffix or '.png'}"
            )
            extra_destination = step_dir / extra_name
            shutil.copy2(extra_path, extra_destination)
            archived_paths.append(str(extra_destination))
    return archived_paths


def _emit_runtime_stage(config, stage: str, **details: Any) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "question_index": getattr(config, "question_index", None),
                "pipeline_variant": PIPELINE_VARIANT,
                **_json_safe(details),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_one(
    config,
    *,
    answer_client=None,
    verifier_client=None,
    shutdown_app: bool = True,
) -> dict[str, Any]:
    task_module = base.load_task_module(config.task)
    selected_question = base.select_question(config.metadata, config.question_index, config.json_root)
    question_json = selected_question.source_path
    source_label = selected_question.source_label
    payload = selected_question.payload

    results_root = config.results_root / PIPELINE_VARIANT
    step_root = config.step_image_root / PIPELINE_VARIANT
    output_path = base.output_path_for(task_module, payload, question_json, results_root)
    step_image_dir = base.step_image_dir_for(task_module, payload, question_json, step_root)
    _emit_runtime_stage(
        config,
        "run_one_selected",
        source_json=source_label,
        output_path=str(output_path),
        step_image_dir=str(step_image_dir),
    )
    if output_path.exists() and not config.overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    scene_name, room_name = task_module.scene_room(payload)
    task_state = base.task_preprocess(task_module, payload, question_json, config)
    _emit_runtime_stage(
        config,
        "run_one_preprocessed",
        skip_reason=task_state.get("skip_reason"),
    )
    if task_state.get("skip_reason"):
        return {
            "source_json": source_label,
            "source_metadata": str(config.metadata),
            "question_index": config.question_index,
            "task": getattr(task_module, "TASK_NAME", config.task),
            "scene": payload.get("scene"),
            "room": payload.get("room"),
            "question_id": task_module.question_id(payload, question_json),
            "skipped": True,
            "skip_reason": task_state["skip_reason"],
            "correct": None,
            "pipeline_variant": PIPELINE_VARIANT,
        }

    pos, quat, camera_info = task_module.initial_camera(payload)
    objects = task_module.build_env_objects(payload)
    env = None
    history: list[dict[str, Any]] = []
    selected_groups: list[list[branch_base.BranchNode]] = []
    final_answer: dict[str, Any] | None = None
    final_raw_text = ""
    final_finish_reason: str | None = None
    final_truncated = False
    env_postprocess: dict[str, Any] = {}
    active_task_state = task_state
    selected_image_dir = step_image_dir / "_verifier_selected"
    frontier = [
        branch_base.BranchNode(
            branch_id="root",
            parent_branch_id=None,
            depth=0,
            action="root",
            position=np.array(pos, dtype=float),
            quaternion_xyzw=np.array(quat, dtype=float),
        )
    ]

    model_name, verifier_model_name = _resolve_model_names(config, task_module)
    if answer_client is None or verifier_client is None:
        answer_client, verifier_client, model_name, verifier_model_name = _build_model_clients(config, task_module)

    try:
        full_scene = bool(getattr(task_module, "FULL_SCENE", False))
        task_name = base.normalize_text(getattr(task_module, "TASK_NAME", config.task))
        desired_object_states = None
        if task_name in OBJECT_STATE_FREE_TASKS:
            desired_object_states = False
        elif task_name in EXPLICIT_OBJECT_STATE_TASKS:
            desired_object_states = True
        if (
            desired_object_states is not None
            and bool(base.gm.ENABLE_OBJECT_STATES) != desired_object_states
        ):
            with base.gm.unlocked():
                base.gm.ENABLE_OBJECT_STATES = desired_object_states
        try:
            if getattr(og, "sim", None) is not None and not og.sim.is_stopped():
                og.sim.stop()
        except Exception:
            pass
        _emit_runtime_stage(
            config,
            "run_one_environment_start",
            scene=scene_name,
            room=room_name,
            object_states_enabled=bool(base.gm.ENABLE_OBJECT_STATES),
        )
        env = og.Environment(
            configs=base.task_build_env_config(
                task_module,
                scene_name,
                room_name,
                config.robot,
                objects,
                full_scene=full_scene,
            )
        )
        _emit_runtime_stage(config, "run_one_environment_ready")
        if not getattr(task_module, "DISABLE_VIEWER_CAMERA_MODALITIES", False):
            for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
                try:
                    og.sim._viewer_camera.add_modality(modality)
                except Exception:
                    pass
        for _ in range(base.task_initial_settle_steps(task_module)):
            og.sim.step()
        _emit_runtime_stage(config, "run_one_initial_settle_done")
        env_postprocess = base.task_postprocess_env(task_module, env, payload, camera_info, task_state)
        _emit_runtime_stage(config, "run_one_postprocess_done")
        camera_override = env_postprocess.get("camera_override") if isinstance(env_postprocess, dict) else None
        if isinstance(camera_override, dict) and camera_override.get("position") and camera_override.get("quaternion_xyzw"):
            pos = np.array(camera_override["position"], dtype=float)
            quat = np.array(camera_override["quaternion_xyzw"], dtype=float)
            camera_info = {**camera_info, "postprocess_camera_override": camera_override}
            frontier = [
                branch_base.BranchNode(
                    branch_id="root",
                    parent_branch_id=None,
                    depth=0,
                    action="root",
                    position=np.array(pos, dtype=float),
                    quaternion_xyzw=np.array(quat, dtype=float),
                )
            ]
        reference_image_paths = base.task_reference_image_paths(task_module, payload, task_state)
        branch_actions = _task_branch_actions(task_module, payload, camera_info)
        if not branch_actions:
            raise RuntimeError(f"Task {task_module.TASK_NAME!r} exposes no executable branch actions")
        requires_sim_restore = any(action not in base.CAMERA_ACTIONS for action in branch_actions)
        _set_node_runtime_state(
            frontier[0],
            og.sim.dump_state(serialized=False) if requires_sim_restore else None,
            task_state,
            _capture_task_object_states(task_module, env, payload) if requires_sim_restore else {},
            requires_sim_restore=requires_sim_restore,
        )
        _emit_runtime_stage(config, "run_one_root_state_ready", branch_action_count=len(branch_actions))

        for step in range(1, config.max_steps + 1):
            _emit_runtime_stage(config, "run_one_step_start", step=step, parent_count=len(frontier))
            parent_count = len(frontier)
            fixed_top_k = getattr(config, "fixed_verifier_top_k", None)
            frontier = _expand_frontier(
                task_module,
                env,
                payload,
                camera_info,
                frontier,
                step,
                step_image_dir,
                active_task_state,
                branch_actions,
            )
            rendered_frontier_size = len(frontier)
            verifier_frontier, verifier_truncated = _select_verifier_frontier(frontier, config.verifier_max_images)
            selection_config = copy.copy(config)
            selection_config.verifier_top_k = fixed_top_k
            verifier_prompt = _build_verifier_prompt(
                task_module,
                payload,
                selection_config,
                camera_info,
                task_state,
                step,
                parent_count,
                len(frontier),
                len(verifier_frontier),
                verifier_truncated,
            )
            verifier_parsed, verifier_raw_text, verifier_finish_reason = verifier_client.generate_json(
                contents=_collect_verifier_contents(
                    verifier_frontier,
                    history,
                    verifier_prompt,
                    reference_image_paths=reference_image_paths,
                    frontier_total=len(frontier),
                    frontier_truncated=verifier_truncated,
                ),
                system_instruction="Return exactly one valid JSON object and nothing else.",
                response_schema=VERIFIER_RESPONSE_SCHEMA,
                max_output_tokens=max(256, min(config.max_new_tokens, 512)),
                temperature=0.0,
                top_p=1.0,
                fallback={
                    "selected_branch_ids": [
                        node.branch_id
                        for node in verifier_frontier[: fixed_top_k or 1]
                    ],
                    "reasoning": "verifier fallback",
                },
            )
            selected_nodes = _choose_selected_nodes(verifier_frontier, verifier_parsed, fixed_top_k)
            active_task_state = _materialize_selected_nodes(
                task_module,
                env,
                payload,
                camera_info,
                selected_nodes,
                step,
                step_image_dir,
                active_task_state,
            )
            selected_groups.append(selected_nodes)
            frontier = selected_nodes
            archived_selected_paths = _archive_selected_nodes(selected_nodes, selected_image_dir, step)

            history.append(
                {
                    "step": step,
                    "frontier_size": len(frontier),
                    "rendered_frontier_size": rendered_frontier_size,
                    "verifier_frontier_size": len(verifier_frontier),
                    "verifier_truncated": verifier_truncated,
                    "requested_top_k": fixed_top_k,
                    "selected_top_k": len(selected_nodes),
                    "selected_count": len(selected_nodes),
                    "selected_branch_ids": [node.branch_id for node in selected_nodes],
                    "selected_image_paths": [str(node.image_path) for node in selected_nodes if node.image_path is not None],
                    "selected_extra_image_paths": [
                        str(path)
                        for node in selected_nodes
                        for path in _node_extra_image_paths(node)
                    ],
                    "selected_action_info": [getattr(node, "action_info", {}) for node in selected_nodes],
                    "selected_archive_paths": archived_selected_paths,
                    "verifier_reasoning": base.normalize_text(verifier_parsed.get("reasoning")) or "no verifier reasoning provided",
                    "verifier_raw_output": verifier_raw_text,
                    "verifier_finish_reason": verifier_finish_reason,
                }
            )
            checkpoint_answer, final_raw_text, final_finish_reason, final_truncated = _request_checkpoint_answer(
                task_module,
                answer_client,
                payload,
                config,
                camera_info,
                task_state,
                selected_groups,
                history,
                reference_image_paths,
                step,
            )
            history[-1]["checkpoint_answer"] = checkpoint_answer["answer"]
            history[-1]["checkpoint_confidence"] = checkpoint_answer["confidence"]
            history[-1]["checkpoint_reasoning"] = checkpoint_answer["reasoning"]
            if checkpoint_answer.get("answer_review") is not None:
                history[-1]["answer_review"] = checkpoint_answer["answer_review"]
            history[-1]["answer_evidence_mode"] = answer_evidence.mode(config)
            history[-1]["answer_evidence_branch_ids"] = list(
                getattr(config, "_answer_evidence_branch_ids", []) or []
            )
            history[-1]["answer_evidence_image_count"] = int(
                getattr(config, "_answer_evidence_image_count", 0) or 0
            )
            checkpoint_conclusive = _is_task_conclusive_answer(
                task_module,
                checkpoint_answer["answer"],
                payload,
                camera_info,
                task_state,
            )
            history[-1]["checkpoint_conclusive"] = checkpoint_conclusive

            confidence_reached = (
                checkpoint_answer["confidence"] > config.threshold
                and checkpoint_conclusive
            )
            if confidence_reached:
                checkpoint_answer["stopped_by"] = "confidence_threshold"
                final_answer = checkpoint_answer
                break
            if step == config.max_steps:
                checkpoint_answer["stopped_by"] = "max_steps_verifier_final"
                final_answer = checkpoint_answer
                break

        if final_answer is None:
            raise RuntimeError("Verifier exploration finished without producing a checkpoint answer")

        if frontier:
            scoring_task_state = _restore_node_runtime_state(
                task_module,
                env,
                frontier[0],
                active_task_state,
            )
            active_task_state = scoring_task_state
        else:
            scoring_task_state = task_state
        score = base.task_score(task_module, payload, final_answer, camera_info, scoring_task_state)
        result = {
            "result_path": str(output_path),
            "source_json": source_label,
            "source_metadata": str(config.metadata),
            "question_index": config.question_index,
            "task": task_module.TASK_NAME,
            "scene": scene_name,
            "room": room_name,
            "question_id": task_module.question_id(payload, question_json),
            "provider": config.provider,
            "model": model_name,
            "verifier_provider": config.verifier_provider,
            "verifier_model": verifier_model_name,
            "pipeline_variant": PIPELINE_VARIANT,
            "branch_actions": list(branch_actions),
            "task_action_space": [*branch_actions, "stop"],
            "control_actions": ["stop"],
            "verifier_top_k_mode": (
                {"mode": "fixed", "fixed_top_k": config.fixed_verifier_top_k}
                if config.fixed_verifier_top_k is not None
                else {"mode": "verifier_selected"}
            ),
            "max_steps": config.max_steps,
            "unbounded_exploration": False,
            "min_steps": config.min_steps,
            "confidence_threshold": config.threshold,
            "verifier_max_images": config.verifier_max_images,
            "final_max_images": config.final_max_images,
            "answer_evidence_mode": answer_evidence.mode(config),
            "answer_evidence_branch_ids": list(getattr(config, "_answer_evidence_branch_ids", []) or []),
            "answer_evidence_image_count": int(getattr(config, "_answer_evidence_image_count", 0) or 0),
            "answer_review_enabled": bool(getattr(config, "answer_review", False)) and task_name == "line",
            "heavy_action_frame_threshold": HEAVY_ACTION_FRAME_THRESHOLD,
            "heavy_action_frame_step_window": HEAVY_ACTION_FRAME_STEP_WINDOW,
            "final_images_truncated": final_truncated,
            "final_answer": final_answer,
            "history": history,
            "step_image_dir": str(step_image_dir),
            "selected_step_image_dir": str(selected_image_dir),
            "camera_info": camera_info,
            "env_postprocess": env_postprocess,
            "final_raw_output": final_raw_text,
            "final_finish_reason": final_finish_reason,
            **score,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_runtime_stage(config, "run_one_result_write", output_path=str(output_path))
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    except BaseException as exc:
        _emit_runtime_stage(
            config,
            "run_one_exception_before_cleanup",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        _cleanup_env(
            task_module,
            env,
            payload if "payload" in locals() else {},
            active_task_state if "active_task_state" in locals() else task_state if "task_state" in locals() else {},
            shutdown_app=shutdown_app,
        )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    if getattr(config, "run_all_questions", False):
        total_questions = 0
        start_question_index = max(0, int(getattr(config, "start_question_index", 0)))
        range_start_question_index = start_question_index
        range_end_question_index = start_question_index
        process_end_question_index = start_question_index
        success_count = 0
        fail_count = 0
        failed_indices: list[int] = []
        try:
            total_questions = _metadata_question_count(config.metadata)
            task_module = base.load_task_module(config.task)
            answer_client, verifier_client, _, _ = _build_model_clients(config, task_module)
            requested_start = min(max(0, int(getattr(config, "start_question_index", 0))), total_questions)
            configured_range_start = getattr(config, "batch_range_start_index", None)
            range_start_question_index = min(
                requested_start if configured_range_start is None else max(0, int(configured_range_start)),
                total_questions,
            )
            configured_end = getattr(config, "end_question_index", None)
            configured_range_end = getattr(config, "batch_range_end_index", None)
            if configured_range_end is None:
                range_end_question_index = total_questions if configured_end is None else min(
                    max(range_start_question_index, int(configured_end)),
                    total_questions,
                )
            else:
                range_end_question_index = min(
                    max(range_start_question_index, int(configured_range_end)),
                    total_questions,
                )
            process_end_question_index = (
                range_end_question_index
                if configured_end is None
                else min(max(range_start_question_index, int(configured_end)), range_end_question_index)
            )
            start_question_index = min(
                max(range_start_question_index, requested_start),
                process_end_question_index,
            )
            success_count, fail_count, failed_indices = _initial_batch_counts(
                config,
                task_module,
                total_questions,
                range_start_question_index,
                start_question_index,
                range_end_question_index,
            )
            _write_batch_progress(
                getattr(config, "batch_progress_file", None),
                status="running",
                total_questions=total_questions,
                start_question_index=range_start_question_index,
                end_question_index=range_end_question_index,
                active_question_index=None,
                next_question_index=start_question_index,
                success_count=success_count,
                fail_count=fail_count,
                failed_indices=failed_indices,
            )
            for question_index in range(start_question_index, process_end_question_index):
                _write_batch_progress(
                    getattr(config, "batch_progress_file", None),
                    status="running",
                    total_questions=total_questions,
                    start_question_index=range_start_question_index,
                    end_question_index=range_end_question_index,
                    active_question_index=question_index,
                    next_question_index=question_index,
                    success_count=success_count,
                    fail_count=fail_count,
                    failed_indices=failed_indices,
                )
                item_config = copy.copy(config)
                item_config.question_index = question_index
                print(
                    json.dumps(
                        {
                            "stage": "question_start",
                            "question_index": question_index,
                            "total_questions": total_questions,
                            "range_start_question_index": range_start_question_index,
                            "range_end_question_index": range_end_question_index,
                            "pipeline_variant": PIPELINE_VARIANT,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                try:
                    result = run_one(
                        item_config,
                        answer_client=answer_client,
                        verifier_client=verifier_client,
                        shutdown_app=False,
                    )
                    success_count += 1
                    failed_indices = [index for index in failed_indices if index != question_index]
                    print(
                        json.dumps(
                            {
                                "stage": "question_done",
                                "question_index": question_index,
                                "source_json": result.get("source_json"),
                                "pipeline_variant": result.get("pipeline_variant"),
                                "skipped": result.get("skipped", False),
                                "skip_reason": result.get("skip_reason"),
                                "final_answer": result.get("final_answer"),
                                "correct": result.get("correct"),
                                "step_image_dir": result.get("step_image_dir"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    _write_batch_progress(
                        getattr(config, "batch_progress_file", None),
                        status="running",
                        total_questions=total_questions,
                        start_question_index=range_start_question_index,
                        end_question_index=range_end_question_index,
                        active_question_index=question_index,
                        next_question_index=question_index + 1,
                        success_count=success_count,
                        fail_count=fail_count,
                        failed_indices=failed_indices,
                    )
                except (Exception, SystemExit) as exc:
                    restart_question_index = question_index + 1
                    fail_count += 1
                    if question_index not in failed_indices:
                        failed_indices.append(question_index)
                        failed_indices.sort()
                    print(
                        json.dumps(
                            {
                                "stage": "question_failed",
                                "question_index": question_index,
                                "pipeline_variant": PIPELINE_VARIANT,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    _write_batch_progress(
                        getattr(config, "batch_progress_file", None),
                        status="running",
                        total_questions=total_questions,
                        start_question_index=range_start_question_index,
                        end_question_index=range_end_question_index,
                        active_question_index=question_index,
                        next_question_index=restart_question_index,
                        success_count=success_count,
                        fail_count=fail_count,
                        failed_indices=failed_indices,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "batch_restart_required",
                                "failed_question_index": question_index,
                                "next_question_index": restart_question_index,
                                "reason": (
                                    "Restarting Isaac/OmniGibson after a per-question failure to prevent scene-state contamination."
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(3)
            _cleanup_env(task_module, None, {}, {}, shutdown_app=True)
            batch_completed = process_end_question_index >= range_end_question_index
            _write_batch_progress(
                getattr(config, "batch_progress_file", None),
                status="completed" if batch_completed else "running",
                total_questions=total_questions,
                start_question_index=range_start_question_index,
                end_question_index=range_end_question_index,
                active_question_index=None,
                next_question_index=process_end_question_index,
                success_count=success_count,
                fail_count=fail_count,
                failed_indices=failed_indices,
            )
            print(
                json.dumps(
                    {
                        "stage": "batch_summary",
                        "pipeline_variant": PIPELINE_VARIANT,
                        "total_questions": total_questions,
                        "range_start_question_index": range_start_question_index,
                        "range_end_question_index": range_end_question_index,
                        "process_end_question_index": process_end_question_index,
                        "range_total_questions": range_end_question_index - range_start_question_index,
                        "batch_completed": batch_completed,
                        "success": success_count,
                        "failed": fail_count,
                        "failed_indices": failed_indices,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (Exception, SystemExit) as exc:
            print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            _write_batch_progress(
                getattr(config, "batch_progress_file", None),
                status="fatal",
                total_questions=total_questions,
                start_question_index=range_start_question_index,
                end_question_index=range_end_question_index,
                active_question_index=None,
                next_question_index=start_question_index,
                success_count=success_count,
                fail_count=fail_count,
                failed_indices=failed_indices,
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                _cleanup_env(base.load_task_module(config.task), None, {}, {}, shutdown_app=True)
            except Exception:
                pass
            return 2

    try:
        task_module = base.load_task_module(config.task)
        answer_client, verifier_client, _, _ = _build_model_clients(config, task_module)
        result = run_one(
            config,
            answer_client=answer_client,
            verifier_client=verifier_client,
            shutdown_app=True,
        )
    except (Exception, SystemExit) as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "source_json": result.get("source_json"),
                "pipeline_variant": result.get("pipeline_variant"),
                "skipped": result.get("skipped", False),
                "skip_reason": result.get("skip_reason"),
                "final_answer": result.get("final_answer"),
                "correct": result.get("correct"),
                "step_image_dir": result.get("step_image_dir"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
