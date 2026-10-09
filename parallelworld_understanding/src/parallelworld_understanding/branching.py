from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import omnigibson as og

import baseline_active as base


BRANCH_CAMERA_ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
)


@dataclass
class BranchNode:
    branch_id: str
    parent_branch_id: str | None
    depth: int
    action: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    image_path: Path | None = None


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-branch-images",
        type=int,
        default=0,
        help="Optional cap on how many frontier images are sent to the model per step. <=0 means unlimited.",
    )
    extra_args, remaining = parser.parse_known_args(argv)
    config = base.parse_args(remaining)
    setattr(config, "max_branch_images", int(extra_args.max_branch_images))
    return config


def _safe_branch_filename(branch_id: str, max_length: int = 120) -> str:
    """Return a stable, filesystem-safe name without losing branch identity."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", branch_id).strip("._") or "branch"
    if len(safe_name) <= max_length:
        return safe_name

    digest = hashlib.sha256(branch_id.encode("utf-8")).hexdigest()[:16]
    readable_budget = max_length - len(digest) - 4
    head_length = readable_budget // 2
    tail_length = readable_budget - head_length
    return f"{safe_name[:head_length]}__{safe_name[-tail_length:]}__{digest}"


def _sample_frontier(frontier: list[BranchNode], max_branch_images: int) -> tuple[list[BranchNode], bool]:
    if max_branch_images <= 0 or len(frontier) <= max_branch_images:
        return list(frontier), False
    if max_branch_images == 1:
        return [frontier[-1]], True
    if not frontier:
        return [], False
    last_index = len(frontier) - 1
    chosen_indexes = []
    for i in range(max_branch_images):
        index = round(i * last_index / (max_branch_images - 1))
        if not chosen_indexes or index != chosen_indexes[-1]:
            chosen_indexes.append(index)
    sampled = [frontier[index] for index in chosen_indexes]
    return sampled, True


def _history_summary(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for item in history:
        lines.append(
            f"Step {item['step']}: explored={item.get('frontier_size', 0)} "
            f"model_views={item.get('prompt_frontier_size', 0)} "
            f"truncated={bool(item.get('frontier_truncated'))} "
            f"answer={item.get('answer')} conf={float(item.get('confidence', 0.0)):.2f} "
            f"action_signal={item.get('action')} reasoning={item.get('reasoning')}"
        )
    return "\n".join(lines)


def _collect_branch_contents(
    frontier: list[BranchNode],
    history: list[dict[str, Any]],
    prompt: str,
    reference_image_paths: list[Path] | None = None,
    frontier_total: int | None = None,
    frontier_truncated: bool = False,
) -> list[Any]:
    contents: list[Any] = []
    summary = _history_summary(history)
    if summary:
        contents.append("Branch exploration history so far:\n" + summary)
    for index, reference in enumerate(reference_image_paths or [], start=1):
        if reference.exists():
            label = "[QUESTION REFERENCE IMAGE - dataset render]" if len(reference_image_paths or []) == 1 else f"[QUESTION REFERENCE IMAGE {index}]"
            contents.append(label)
            contents.append(reference)
    if frontier_total is not None:
        contents.append(
            "Current branch frontier summary: "
            f"total_views={frontier_total}, supplied_views={len(frontier)}, truncated={frontier_truncated}."
        )
    for node in frontier:
        if node.image_path is None or not node.image_path.exists():
            continue
        contents.append(
            f"[BRANCH VIEW depth={node.depth} branch={node.branch_id} action={node.action} "
            f"parent={node.parent_branch_id or 'root'}]"
        )
        contents.append(node.image_path)
    contents.append(prompt)
    return contents


def _build_branching_prompt(
    task_module,
    payload: dict[str, Any],
    config,
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    step: int,
    parent_count: int,
    frontier_total: int,
    prompt_frontier_size: int,
    frontier_truncated: bool,
) -> str:
    base_prompt = base.task_build_system_prompt(
        task_module,
        payload,
        config.threshold,
        config.min_steps,
        camera_info,
        task_state,
    )
    extra = [
        "",
        "Pipeline override for this run:",
        f"  - This is exhaustive branch step {step}.",
        f"  - The system expanded all six translational actions from every previously reachable camera pose.",
        f"  - Parent positions at previous depth: {parent_count}.",
        f"  - Rendered frontier views at this depth: {frontier_total}.",
        f"  - Frontier views supplied to you in this prompt: {prompt_frontier_size}.",
        f"  - Frontier truncation applied: {'yes' if frontier_truncated else 'no'}.",
        "  - Use the `action` field only as a control signal:",
        "    - return `stop` if the frontier evidence is sufficient to answer now;",
        "    - otherwise return any non-stop movement action to request another exhaustive six-way expansion round.",
        "  - The views labeled [BRANCH VIEW ...] are peer frontier views at the same depth.",
        "  - There is no single privileged CURRENT view in this pipeline.",
    ]
    return base_prompt + "\n" + "\n".join(extra)


def _build_branching_force_choice_prompt(
    task_module,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    task_state: dict[str, Any],
    frontier_total: int,
    prompt_frontier_size: int,
    frontier_truncated: bool,
) -> str:
    base_prompt = base.task_build_force_choice_prompt(task_module, payload, camera_info, task_state)
    extra = [
        "",
        "Pipeline override for final choice:",
        f"  - Rendered frontier views at the final depth: {frontier_total}.",
        f"  - Frontier views supplied in this prompt: {prompt_frontier_size}.",
        f"  - Frontier truncation applied: {'yes' if frontier_truncated else 'no'}.",
        "  - The branch views are peer observations from exhaustive six-way expansion.",
        "  - Ignore action selection and output only the final answer JSON requested above.",
    ]
    return base_prompt + "\n" + "\n".join(extra)


def _expand_frontier(
    task_module,
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    frontier: list[BranchNode],
    step: int,
    step_image_dir: Path,
    task_state: dict[str, Any],
) -> list[BranchNode]:
    step_dir = step_image_dir / f"step_{step:03d}"
    expanded: list[BranchNode] = []
    for parent in frontier:
        for action in BRANCH_CAMERA_ACTIONS:
            next_pos, next_quat = base.apply_camera_action(parent.position, parent.quaternion_xyzw, action)
            branch_id = action if parent.branch_id == "root" else f"{parent.branch_id}/{action}"
            image_path = step_dir / f"{_safe_branch_filename(branch_id)}.png"
            rendered_path = base.task_capture_image(
                task_module,
                env,
                payload,
                camera_info,
                next_pos,
                next_quat,
                image_path,
                task_state,
            )
            expanded.append(
                BranchNode(
                    branch_id=branch_id,
                    parent_branch_id=parent.branch_id,
                    depth=step,
                    action=action,
                    position=next_pos,
                    quaternion_xyzw=next_quat,
                    image_path=Path(rendered_path),
                )
            )
    return expanded


def _force_final_choice(
    task_module,
    model_client,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    history: list[dict[str, Any]],
    frontier: list[BranchNode],
    config,
    task_state: dict[str, Any],
    reference_image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    prompt_frontier, frontier_truncated = _sample_frontier(frontier, int(getattr(config, "max_branch_images", 0)))
    prompt = _build_branching_force_choice_prompt(
        task_module,
        payload,
        camera_info,
        task_state,
        frontier_total=len(frontier),
        prompt_frontier_size=len(prompt_frontier),
        frontier_truncated=frontier_truncated,
    )
    parsed, raw_text, _ = model_client.generate_json(
        contents=_collect_branch_contents(
            prompt_frontier,
            history,
            prompt,
            reference_image_paths=reference_image_paths,
            frontier_total=len(frontier),
            frontier_truncated=frontier_truncated,
        ),
        system_instruction="Return exactly one valid JSON object and nothing else.",
        response_schema=None,
        max_output_tokens=max(256, min(config.max_new_tokens, 512)),
        temperature=0.0,
        top_p=1.0,
        fallback={"answer": "not sure", "confidence": 0.0, "reasoning": "forced choice parse fallback"},
    )
    parsed = base.task_parse_model_output(task_module, parsed, payload, task_state)
    return {
        "answer": parsed.get("answer", "not sure"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": base.normalize_text(parsed.get("reasoning")) or "forced choice fallback",
        "raw_output": raw_text,
        "prompt_frontier_size": len(prompt_frontier),
        "frontier_total": len(frontier),
        "frontier_truncated": frontier_truncated,
    }


def run_one(config) -> dict[str, Any]:
    task_module = base.load_task_module(config.task)
    selected_question = base.select_question(config.metadata, config.question_index, config.json_root)
    question_json = selected_question.source_path
    source_label = selected_question.source_label
    payload = selected_question.payload

    results_root = config.results_root / "branching"
    step_root = config.step_image_root / "branching"
    output_path = base.output_path_for(task_module, payload, question_json, results_root)
    step_image_dir = base.step_image_dir_for(task_module, payload, question_json, step_root)
    if output_path.exists() and not config.overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    scene_name, room_name = task_module.scene_room(payload)
    task_state = base.task_preprocess(task_module, payload, question_json, config)
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
            "pipeline_variant": "branching",
        }

    pos, quat, camera_info = task_module.initial_camera(payload)
    objects = task_module.build_env_objects(payload)
    env = None
    history: list[dict[str, Any]] = []
    final_answer: dict[str, Any] | None = None
    env_postprocess: dict[str, Any] = {}
    frontier = [
        BranchNode(
            branch_id="root",
            parent_branch_id=None,
            depth=0,
            action="root",
            position=np.array(pos, dtype=float),
            quaternion_xyzw=np.array(quat, dtype=float),
        )
    ]

    if config.provider == "gpt":
        default_model = "gpt-5"
    elif config.provider == "qwen":
        default_model = "qwen"
    else:
        default_model = getattr(task_module, "DEFAULT_MODEL", "gemini-2.5-flash")
    model_name = config.model or default_model
    model_client = base.build_model_client(config.provider, config.api_key, model_name)

    try:
        full_scene = bool(getattr(task_module, "FULL_SCENE", False))
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
        if not getattr(task_module, "DISABLE_VIEWER_CAMERA_MODALITIES", False):
            for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
                try:
                    og.sim._viewer_camera.add_modality(modality)
                except Exception:
                    pass
        for _ in range(base.task_initial_settle_steps(task_module)):
            og.sim.step()
        env_postprocess = base.task_postprocess_env(task_module, env, payload, camera_info, task_state)
        camera_override = env_postprocess.get("camera_override") if isinstance(env_postprocess, dict) else None
        if isinstance(camera_override, dict) and camera_override.get("position") and camera_override.get("quaternion_xyzw"):
            pos = np.array(camera_override["position"], dtype=float)
            quat = np.array(camera_override["quaternion_xyzw"], dtype=float)
            camera_info = {**camera_info, "postprocess_camera_override": camera_override}
            frontier = [
                BranchNode(
                    branch_id="root",
                    parent_branch_id=None,
                    depth=0,
                    action="root",
                    position=np.array(pos, dtype=float),
                    quaternion_xyzw=np.array(quat, dtype=float),
                )
            ]
        reference_image_paths = base.task_reference_image_paths(task_module, payload, task_state)

        for step in range(1, config.max_steps + 1):
            parent_count = len(frontier)
            frontier = _expand_frontier(
                task_module,
                env,
                payload,
                camera_info,
                frontier,
                step,
                step_image_dir,
                task_state,
            )
            prompt_frontier, frontier_truncated = _sample_frontier(frontier, int(getattr(config, "max_branch_images", 0)))
            prompt = _build_branching_prompt(
                task_module,
                payload,
                config,
                camera_info,
                task_state,
                step=step,
                parent_count=parent_count,
                frontier_total=len(frontier),
                prompt_frontier_size=len(prompt_frontier),
                frontier_truncated=frontier_truncated,
            )
            parsed, raw_text, finish_reason = model_client.generate_json(
                contents=_collect_branch_contents(
                    prompt_frontier,
                    history,
                    prompt,
                    reference_image_paths=reference_image_paths,
                    frontier_total=len(frontier),
                    frontier_truncated=frontier_truncated,
                ),
                system_instruction="Return exactly one valid JSON object and nothing else.",
                response_schema=getattr(task_module, "ACTION_RESPONSE_SCHEMA", None),
                max_output_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                fallback={"action": "move_forward", "answer": "not sure", "confidence": 0.0, "reasoning": "parse fallback"},
            )
            parsed = base.task_parse_model_output(task_module, parsed, payload, task_state)
            action = parsed["action"]
            answer = parsed["answer"]
            confidence = float(parsed["confidence"])
            reasoning = parsed["reasoning"]

            history.append(
                {
                    "step": step,
                    "action": action,
                    "answer": answer,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "frontier_size": len(frontier),
                    "prompt_frontier_size": len(prompt_frontier),
                    "frontier_truncated": frontier_truncated,
                    "parent_count": parent_count,
                    "branch_ids": [node.branch_id for node in prompt_frontier],
                    "image_paths": [str(node.image_path) for node in prompt_frontier if node.image_path is not None],
                    "raw_output": raw_text,
                    "finish_reason": finish_reason,
                }
            )

            should_stop, stop_reason = base.task_should_stop(task_module, parsed, history, step, config)
            if should_stop:
                resolved_answer, answer_step = base.task_resolve_final_answer(task_module, history)
                if base.task_needs_force_final_choice(task_module, resolved_answer, stop_reason):
                    forced = _force_final_choice(
                        task_module,
                        model_client,
                        payload,
                        camera_info,
                        history,
                        frontier,
                        config,
                        task_state,
                        reference_image_paths=reference_image_paths,
                    )
                    resolved_answer = forced["answer"]
                    answer_step = step
                    confidence = forced["confidence"]
                    reasoning = forced["reasoning"]
                    history.append(
                        {
                            "step": step,
                            "action": "force_final_choice",
                            "answer": resolved_answer,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "frontier_size": forced["frontier_total"],
                            "prompt_frontier_size": forced["prompt_frontier_size"],
                            "frontier_truncated": forced["frontier_truncated"],
                            "raw_output": forced["raw_output"],
                        }
                    )
                    stop_reason = f"{stop_reason}_forced_choice"
                final_answer = {
                    "answer": resolved_answer,
                    "answer_step": answer_step,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "steps": step,
                    "stopped_by": stop_reason,
                }
                break

        final_answer = final_answer or {
            "answer": "not sure",
            "answer_step": -1,
            "confidence": 0.0,
            "reasoning": "no answer",
            "steps": len(history),
            "stopped_by": "empty",
        }
        score = base.task_score(task_module, payload, final_answer, camera_info, task_state)
        result = {
            "source_json": source_label,
            "source_metadata": str(config.metadata),
            "question_index": config.question_index,
            "task": task_module.TASK_NAME,
            "scene": scene_name,
            "room": room_name,
            "question_id": task_module.question_id(payload, question_json),
            "provider": config.provider,
            "model": model_name,
            "pipeline_variant": "branching",
            "branch_actions": list(BRANCH_CAMERA_ACTIONS),
            "max_branch_images": int(getattr(config, "max_branch_images", 0)),
            "final_answer": final_answer,
            "history": history,
            "step_image_dir": str(step_image_dir),
            "camera_info": camera_info,
            "env_postprocess": env_postprocess,
            **score,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        handled_cleanup = False
        try:
            handled_cleanup = base.task_cleanup_runtime(
                task_module,
                env,
                payload if "payload" in locals() else {},
                task_state if "task_state" in locals() else {},
            )
        except Exception:
            handled_cleanup = False
        try:
            if not active_exception and not handled_cleanup and getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        result = run_one(config)
    except Exception as exc:
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
