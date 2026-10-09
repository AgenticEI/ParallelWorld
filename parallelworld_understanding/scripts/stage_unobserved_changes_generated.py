#!/usr/bin/env python3
"""Stage generated Unobserved Change questions in the evaluator layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BIG_TASK = "Temporal Understanding"
SMALL_TASK = "Unobserved Change"
READY_MARKER = ".ready.json"


def decode_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata_json")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata if isinstance(metadata, dict) else payload


def image_paths(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "image_path" and isinstance(item, str) and item.strip():
                yield item
            else:
                yield from image_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from image_paths(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_json_root.resolve()
    generated_root = args.generated_root.resolve()
    output_root = args.output_root.resolve()
    source_task_root = source_root / BIG_TASK / SMALL_TASK
    metadata_source = source_root / f"{BIG_TASK}.json"

    source_paths = sorted(source_task_root.rglob("*.json"))
    if not source_paths:
        raise RuntimeError(f"No source questions found under {source_task_root}")

    staged: list[tuple[Path, dict[str, Any]]] = []
    missing_questions: list[str] = []
    missing_images: list[str] = []
    for source_path in source_paths:
        outer = json.loads(source_path.read_text(encoding="utf-8"))
        metadata = decode_metadata(outer)
        scene = str(outer.get("scene") or metadata.get("scene") or "").strip()
        room = str(outer.get("room") or metadata.get("room") or "").strip()
        task_type = str(outer.get("task_type") or metadata.get("task_type") or "").strip()
        question_index = outer.get("question_index", metadata.get("question_index"))
        if question_index is None:
            stem = source_path.stem
            if "__q_" not in stem:
                raise ValueError(f"Cannot determine question index: {source_path}")
            question_index = int(stem.rsplit("__q_", 1)[1])
        if not scene or not room or not task_type:
            raise ValueError(f"Incomplete question metadata: {source_path}")

        generated_path = (
            generated_root
            / scene
            / room
            / "question_jsons"
            / task_type
            / f"q_{int(question_index):03d}.json"
        )
        if not generated_path.is_file():
            missing_questions.append(str(generated_path))
            continue

        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        for raw_image_path in sorted(set(image_paths(generated))):
            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                image_path = (generated_path.parent / image_path).resolve()
            if not image_path.is_file():
                missing_images.append(f"{generated_path}: {image_path}")

        relative_path = source_path.relative_to(source_root)
        staged.append((relative_path, generated))

    if missing_questions or missing_images:
        details = []
        if missing_questions:
            details.append(
                f"missing generated questions ({len(missing_questions)}):\n  "
                + "\n  ".join(missing_questions[:20])
            )
        if missing_images:
            details.append(
                f"missing referenced images ({len(missing_images)}):\n  "
                + "\n  ".join(missing_images[:20])
            )
        raise RuntimeError("\n".join(details))

    output_root.mkdir(parents=True, exist_ok=True)
    for relative_path, generated in staged:
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    suite_metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    suite_paths = suite_metadata.get("json_paths")
    if not isinstance(suite_paths, list):
        raise ValueError(f"Metadata has no json_paths list: {metadata_source}")
    selected_paths = [
        path
        for path in suite_paths
        if len(Path(str(path)).parts) >= 2
        and Path(str(path)).parts[0] == BIG_TASK
        and Path(str(path)).parts[1] == SMALL_TASK
    ]
    if len(selected_paths) != len(staged):
        raise RuntimeError(
            f"Metadata selects {len(selected_paths)} questions, but staged {len(staged)}"
        )
    suite_metadata["json_paths"] = selected_paths
    suite_metadata["count"] = len(selected_paths)
    suite_metadata["total_target"] = len(selected_paths)
    suite_metadata["available_json_count"] = len(selected_paths)
    suite_metadata["valid_json_count"] = len(selected_paths)
    suite_metadata["available_per_task_type"] = {SMALL_TASK: len(selected_paths)}
    suite_metadata["sampled_per_task_type"] = {SMALL_TASK: len(selected_paths)}
    (output_root / metadata_source.name).write_text(
        json.dumps(suite_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    marker = {
        "question_count": len(staged),
        "source_json_root": str(source_root),
        "generated_root": str(generated_root),
    }
    (output_root / READY_MARKER).write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Staged and validated {len(staged)} questions at {output_root}")


if __name__ == "__main__":
    main()
