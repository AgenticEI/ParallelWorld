#!/usr/bin/env python3
"""Regenerate and stage Unobserved Change questions with their render assets.

The process imports OmniGibson once, reuses one environment for all requested
rooms in a scene, and recreates environments only when the scene changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEAN_ROOT = REPO_ROOT / "dataset/json_clean/Temporal Understanding/Unobserved Change"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".runtime/unobserved_regen"
DEFAULT_STAGE_ROOT = REPO_ROOT / ".runtime/unobserved_fixed/source_json"
DEFAULT_FIXED_ROOT = REPO_ROOT / ".runtime/unobserved_fixed"
DEFAULT_KEYS_JSON = REPO_ROOT / "src/dataset_generation/task_confusing_relation/keys.json"
BEHAVIOR_ROOT = REPO_ROOT.parent / "BEHAVIOR-1K"
DEFAULT_OBJECT_INVENTORY = BEHAVIOR_ROOT / "bddl3/bddl/generated_data/object_inventory.json"
DEFAULT_AVG_CATEGORY_SPECS = BEHAVIOR_ROOT / "OmniGibson/omnigibson/configs/avg_category_specs.json"
DEFAULT_OBJECT_DATASET_ROOT = BEHAVIOR_ROOT / "datasets/behavior-1k-assets/objects"


@dataclass(frozen=True)
class SelectedQuestion:
    source: Path
    scene: str
    room: str
    task_type: str
    q_idx: int
    floor_name: str | None
    expected: dict[str, Any]


def _read_selected_questions(clean_root: Path) -> list[SelectedQuestion]:
    selected = []
    for source in sorted(clean_root.rglob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        metadata = json.loads(row.get("metadata_json") or "{}")
        question_data = metadata.get("question_data") or {}
        task_type = str(metadata.get("task_type") or question_data.get("task_type") or "")
        q_idx = int(question_data.get("candidate_index", metadata.get("question_index", 0)))
        if not task_type:
            raise ValueError(f"Missing task type in {source}")
        selected.append(
            SelectedQuestion(
                source=source,
                scene=str(row["scene"]),
                room=str(row["room"]),
                task_type=task_type,
                q_idx=q_idx,
                floor_name=metadata.get("floor_name"),
                expected=question_data,
            )
        )
    if not selected:
        raise ValueError(f"No selected question JSONs found below {clean_root}")
    return selected


def _phase_content(content: Any) -> Any:
    if not isinstance(content, dict):
        return None
    return {
        key: content.get(key)
        for key in ("category", "display_name", "model", "bbox_size_m", "sampling_source")
    }


def _semantic_question(question_data: dict[str, Any]) -> dict[str, Any]:
    boxes = []
    for box in question_data.get("boxes") or []:
        container = box.get("container") or {}
        boxes.append(
            {
                "box_index": int(box.get("box_index", len(boxes))),
                "position_label": box.get("position_label"),
                "change_type": box.get("change_type"),
                "phase1_content": _phase_content(box.get("phase1_content")),
                "phase2_content": _phase_content(box.get("phase2_content")),
                "container_category": container.get("category"),
                "container_model": container.get("model"),
            }
        )
    return {
        "task_type": question_data.get("task_type"),
        "question": question_data.get("question"),
        "options": question_data.get("options"),
        "answer": question_data.get("answer"),
        "candidate_index": int(question_data.get("candidate_index", 0)),
        "box_count": int(question_data.get("box_count", len(boxes))),
        "boxes": boxes,
        "question_metadata": question_data.get("question_metadata") or {},
    }


def _describe_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    expected_semantic = _semantic_question(expected)
    actual_semantic = _semantic_question(actual)
    if expected_semantic == actual_semantic:
        return ""
    differences = []
    for key in expected_semantic:
        if expected_semantic[key] != actual_semantic.get(key):
            differences.append(
                f"{key}: expected={expected_semantic[key]!r}, actual={actual_semantic.get(key)!r}"
            )
    return "; ".join(differences)


def _declared_image_paths(payload: dict[str, Any]) -> list[Path]:
    return [Path(value) for value in payload.get("image_paths") or [] if isinstance(value, str)]


def _generated_json(output_root: Path, item: SelectedQuestion) -> Path:
    return (
        output_root
        / item.scene
        / item.room
        / "question_jsons"
        / item.task_type
        / f"q_{item.q_idx:03d}.json"
    )


def _stage_path(stage_root: Path, item: SelectedQuestion) -> Path:
    return (
        stage_root
        / "Temporal Understanding"
        / "Unobserved Change"
        / item.scene
        / item.room
        / f"{item.task_type}__q_{item.q_idx:03d}.json"
    )


def _validate_and_stage(output_root: Path, stage_root: Path, item: SelectedQuestion) -> None:
    generated_path = _generated_json(output_root, item)
    if not generated_path.is_file():
        raise FileNotFoundError(f"Generated question is missing: {generated_path}")
    payload = json.loads(generated_path.read_text(encoding="utf-8"))
    difference = _describe_difference(item.expected, payload.get("question_data") or {})
    if difference:
        raise ValueError(f"Generated question does not match {item.source}: {difference}")
    missing = [path for path in _declared_image_paths(payload) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generated question has {len(missing)} missing images; first: {missing[0]}")
    target = _stage_path(stage_root, item)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_path, target)


def _already_staged(stage_root: Path, item: SelectedQuestion) -> bool:
    target = _stage_path(stage_root, item)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if _describe_difference(item.expected, payload.get("question_data") or {}):
            return False
        return bool(_declared_image_paths(payload)) and all(path.is_file() for path in _declared_image_paths(payload))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _export_fixed_dataset(stage_root: Path, fixed_root: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "hf_dataset/scripts/export_hf_questions.py"),
        "--json-root",
        str(stage_root),
        "--output",
        str(fixed_root / "hf_dataset/data/questions.jsonl"),
        "--clean-root",
        str(fixed_root / "json_clean"),
        "--hf-root",
        str(fixed_root / "hf_dataset"),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _clear_omnigibson_texture_cache() -> None:
    appdata = os.environ.get("OMNIGIBSON_APPDATA_PATH")
    if not appdata:
        return
    texture_cache = Path(appdata).resolve() / "global/cache/texturecache"
    if not texture_cache.is_dir():
        return
    removed_bytes = sum(path.stat().st_size for path in texture_cache.rglob("*") if path.is_file())
    shutil.rmtree(texture_cache)
    texture_cache.mkdir(parents=True, exist_ok=True)
    print(f"Cleared OmniGibson texture cache ({removed_bytes / 1024**3:.2f} GiB)", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--keys-json", type=Path, default=DEFAULT_KEYS_JSON)
    parser.add_argument("--object-inventory", type=Path, default=DEFAULT_OBJECT_INVENTORY)
    parser.add_argument("--avg-category-specs", type=Path, default=DEFAULT_AVG_CATEGORY_SPECS)
    parser.add_argument("--object-dataset-root", type=Path, default=DEFAULT_OBJECT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene")
    parser.add_argument("--room")
    parser.add_argument("--disable-runtime-physics", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = _read_selected_questions(args.clean_root)
    if args.scene:
        selected = [item for item in selected if item.scene == args.scene]
    if args.room:
        selected = [item for item in selected if item.room == args.room]
    if not selected:
        raise SystemExit("The requested scene/room does not contain selected questions.")

    pending = [item for item in selected if not _already_staged(args.stage_root, item)]
    print(
        f"Manifest: selected={len(selected)} staged={len(selected) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        _clear_omnigibson_texture_cache()
        # Importing the generator starts the OmniGibson application. Keep this
        # import inside main so completed/resumed runs do not start it at all.
        generator_dir = Path(__file__).resolve().parent
        counting_dir = generator_dir.parent / "task_counting"
        for path in (generator_dir, counting_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import batch_unobserved_changes as generator

        generator.bcm.DIRECT_PLACEMENT_MODE = bool(args.disable_runtime_physics)
        for resource in (args.object_inventory, args.avg_category_specs):
            if not resource.is_file():
                raise FileNotFoundError(f"Required generator resource is missing: {resource}")
        if not args.object_dataset_root.is_dir():
            raise FileNotFoundError(f"Required object dataset is missing: {args.object_dataset_root}")
        generator.bcm.DEFAULT_OBJECT_INVENTORY = str(args.object_inventory)
        generator.bcm.DEFAULT_AVG_CATEGORY_SPECS = str(args.avg_category_specs)
        generator.bcm.DEFAULT_OBJECT_DATASET_ROOT = args.object_dataset_root
        generator.bcm._load_object_inventory_payload.cache_clear()
        generator.bcm._bounding_box_sizes_by_model.cache_clear()
        candidates = generator.bcm._build_count_target_candidates(str(args.keys_json))
        if not candidates:
            raise ValueError("No content candidates passed the local model and bbox filters.")
        print(f"Loaded {len(candidates)} deterministic content candidates", flush=True)
        by_scene: dict[str, list[SelectedQuestion]] = defaultdict(list)
        for item in pending:
            by_scene[item.scene].append(item)

        failures = []
        for scene_name, scene_items in sorted(by_scene.items()):
            rooms = sorted({item.room for item in scene_items})
            print(f"Loading scene {scene_name} once for {len(rooms)} room(s)", flush=True)
            env = None
            try:
                config = generator.bcm._build_config(
                    scene_name=scene_name,
                    robot="R1",
                    load_full_scene=True,
                    room_names=rooms,
                )
                env = generator.bcm.og.Environment(configs=config)
                generator.bcm._set_viewer_camera_fov()
                scene = env.scene
                wall_records = generator.bcm._collect_wall_records(scene)
                wall_bboxes = [wall.bbox_world_xy for wall in wall_records if wall.is_structural_wall]
                by_room: dict[str, list[SelectedQuestion]] = defaultdict(list)
                for item in scene_items:
                    by_room[item.room].append(item)
                for room_name, room_items in sorted(by_room.items()):
                    task_types = sorted({item.task_type for item in room_items})
                    question_count = max(item.q_idx for item in room_items) + 1
                    floor_names = {item.floor_name for item in room_items if item.floor_name}
                    if len(floor_names) > 1:
                        raise ValueError(f"Conflicting stored floors for {scene_name}/{room_name}: {floor_names}")
                    existing_questions = {
                        (item.task_type, item.q_idx): item.expected
                        for item in room_items
                    }
                    print(
                        f"Generating {scene_name}/{room_name}: task_types={task_types} q_per_task={question_count}",
                        flush=True,
                    )
                    try:
                        generator._process_room(
                            env=env,
                            scene=scene,
                            scene_name=scene_name,
                            room_name=room_name,
                            floor_name=next(iter(floor_names), None),
                            output_root=str(args.output_root),
                            seed=args.seed,
                            questions_per_task=question_count,
                            task_types=tuple(task_types),
                            count_target_candidates=candidates,
                            structural_wall_bboxes=wall_bboxes,
                            agent_position=None,
                            skip_render=False,
                            existing_questions=existing_questions,
                        )
                        for item in room_items:
                            _validate_and_stage(args.output_root, args.stage_root, item)
                        print(f"Staged {len(room_items)} selected question(s) for {scene_name}/{room_name}", flush=True)
                    except Exception as exc:
                        failures.append(f"{scene_name}/{room_name}: {exc}")
                        print(f"FAILED {scene_name}/{room_name}: {exc}", file=sys.stderr, flush=True)
                        traceback.print_exc()
            except Exception as exc:
                failures.append(f"{scene_name}: {exc}")
                print(f"FAILED scene {scene_name}: {exc}", file=sys.stderr, flush=True)
                traceback.print_exc()
            finally:
                if env is not None:
                    try:
                        generator.bcm.og.clear()
                    except Exception:
                        traceback.print_exc()
                _clear_omnigibson_texture_cache()
            if failures:
                break

        if failures:
            print("Generation stopped after a mismatch or render failure:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1

    if not args.no_export:
        _export_fixed_dataset(args.stage_root, args.fixed_root)
    print(f"Complete: staged={len(selected)} fixed_root={args.fixed_root}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim can segfault while Python unloads its extension modules even
    # after og.clear() and a successful export. All owned state is flushed.
    os._exit(exit_code)
