#!/usr/bin/env python3
"""Prepare ESI runner inputs without editing the source or copying image assets.

Accepts the upstream json_clean directory or its fixed-column questions.jsonl
export. JSONL has no original question filename field: rows are retained intact
and written as <big_task>/<small_task>/<scene>/<room>/q_<id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def component(value: Any, field: str) -> str:
    """Reject path traversal and components invalid on supported platforms."""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"Invalid {field}: expected a string or integer")
    value = str(value)
    if (
        not value or value in {".", ".."}
        or re.search(r'[\\/<>:"|?*\x00-\x1f]', value)
        or value.endswith((" ", "."))
        or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", value)
    ):
        raise ValueError(f"Invalid path component in {field}: {value!r}")
    return value


def load_questions(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    questions: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    if source.is_dir():
        for question in sorted(source.rglob("*.json")):
            if not within(question.resolve(), source):
                raise ValueError(f"Question symlink leaves the input directory: {question}")
            relative = question.relative_to(source)
            if len(relative.parts) == 1:
                payload = read_json(question)
                if isinstance(payload, dict) and isinstance(payload.get("json_paths"), list):
                    summaries[question.stem] = payload
                elif isinstance(payload, list) and all(isinstance(item, str) for item in payload):
                    summaries[question.stem] = {"json_paths": payload}
                continue
            # Upstream question layout: category/subcategory/scene/room/file.json.
            # Asset metadata and unrelated auxiliary JSON are not runner questions.
            if len(relative.parts) < 5 or relative.parts[0] == "assets":
                continue
            for part in relative.parts:
                component(part, "question path")
            payload = read_json(question)
            if not isinstance(payload, dict):
                raise ValueError(f"Question JSON must contain an object: {question}")
            questions[relative.as_posix()] = payload
        source_format = "json_directory"
    else:
        if source.suffix.lower() != ".jsonl":
            raise ValueError("--source must be a json_clean directory or questions.jsonl")
        with source.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_number} must contain an object")
                required = ("id", "big_task", "small_task", "scene", "room", "metadata_json")
                if any(key not in row for key in required):
                    raise ValueError(f"JSONL line {line_number} lacks an upstream question field")
                metadata = row["metadata_json"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if not isinstance(metadata, dict):
                    raise ValueError(f"JSONL metadata_json must encode an object at line {line_number}")
                parts = [component(row[key], key) for key in required[:5]]
                relative = "/".join([parts[1], parts[2], parts[3], parts[4], f"q_{parts[0]}.json"])
                if relative in questions:
                    raise ValueError(f"Duplicate JSONL question path at line {line_number}: {relative}")
                questions[relative] = row
        source_format = "questions_jsonl"
    if not questions:
        raise ValueError(f"No question files found in {source}")
    return questions, summaries, source_format


class Relocator:
    def __init__(self, source: Path, output: Path, questions: set[str], path_maps: list[str]):
        self.source = source
        self.output = output
        self.questions = questions
        self.counts: Counter[str] = Counter()
        mappings: dict[str, str] = {}
        for item in path_maps:
            old, sep, new = item.partition("=")
            old = old.replace("\\", "/").rstrip("/")
            new = new.replace("\\", "/").rstrip("/")
            if not sep or not old or not new:
                raise ValueError("Each --path-map must contain nonempty OLD=NEW prefixes")
            if old in mappings and mappings[old] != new:
                raise ValueError(f"Conflicting --path-map prefixes: {old}")
            mappings[old] = new
        self.path_maps = sorted(mappings.items(), key=lambda item: (-len(item[0]), item[0]))

    def question_relative(self, raw: str, *, required: bool = False) -> str | None:
        normalized = raw.replace("\\", "/")
        candidate: str | None = None
        if normalized in self.questions:
            candidate = normalized
        source_prefix = self.source.as_posix().rstrip("/") + "/"
        if self.source.is_dir() and normalized.startswith(source_prefix):
            candidate = normalized[len(source_prefix):]
        # Historical hosts may have a different prefix; the canonical suffix
        # must exactly name a question present in this input, never a new path.
        absolute = normalized.startswith("/") or bool(re.match(r"^[A-Za-z]:/", normalized))
        if absolute and "/json_clean/" in normalized:
            candidate = normalized.split("/json_clean/", 1)[1]
        if candidate is not None:
            if candidate not in self.questions:
                raise ValueError(f"Question reference is absent from the input: {raw}")
            return candidate
        if required:
            raise ValueError(f"Cannot resolve a category-summary question reference: {raw}")
        return None

    def transform(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.transform(item) for item in value]
        if not isinstance(value, str):
            return value
        # The upstream exporter uses encoded JSON strings in metadata_json,
        # options_json, and image_paths_json. Preserve their string type.
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                changed = self.transform(decoded)
                if changed != decoded:
                    self.counts["encoded_json_fields_updated"] += 1
                    return json.dumps(changed, ensure_ascii=False, separators=(",", ":"))
                return value
        normalized = value.replace("\\", "/")
        absolute = normalized.startswith("/") or bool(re.match(r"^[A-Za-z]:/", normalized))
        if absolute and normalized.lower().endswith(".json"):
            relative = self.question_relative(value)
            if relative is not None:
                self.counts["relocated_question_references"] += 1
                return (self.output / relative).as_posix()
        for old, new in self.path_maps:
            if normalized == old or normalized.startswith(old + "/"):
                self.counts["mapped_path_values"] += 1
                return new + normalized[len(old):]
        return value


def prepare(source: Path, output: Path, path_maps: list[str] | None = None) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    requested_output = output.expanduser().absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError("--output already exists; choose a new directory")
    output = requested_output.resolve()
    protected_root = source if source.is_dir() else source.parent
    if within(output, protected_root):
        raise ValueError("--output must be outside the input directory")
    if output.exists() or output.is_symlink():
        raise ValueError("--output already exists; choose a new directory")
    questions, summaries, source_format = load_questions(source)
    relocator = Relocator(source, output, set(questions), path_maps or [])
    for category, summary in summaries.items():
        for reference in summary["json_paths"]:
            if not isinstance(reference, str):
                raise ValueError(f"Invalid json_paths entry in {category}.json")
            relative = relocator.question_relative(reference, required=True)
            if relative.split("/", 1)[0] != category:
                raise ValueError(f"Category summary references another category: {category}.json")

    transformed = {relative: relocator.transform(payload) for relative, payload in questions.items()}
    grouped: dict[str, list[str]] = defaultdict(list)
    for relative in questions:
        grouped[relative.split("/", 1)[0]].append(relative)
    for category in summaries:
        if category not in grouped:
            raise ValueError(f"Summary has no question files: {category}.json")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent)).resolve()
    try:
        for relative, payload in transformed.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for category, paths in sorted(grouped.items()):
            counts = Counter(path.split("/")[1] for path in paths)
            summary = dict(relocator.transform(summaries.get(category, {})))
            summary.update(
                root_dir=".", count=len(paths), total_target=len(paths),
                available_json_count=len(paths), valid_json_count=len(paths),
                available_per_task_type=dict(sorted(counts.items())),
                sampled_per_task_type=dict(sorted(counts.items())),
                per_type_targets=dict(sorted(counts.items())), json_paths=sorted(paths),
            )
            if "per_type_target" in summary:
                summary["per_type_target"] = max(counts.values())
            (stage / f"{category}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if output.exists() or output.is_symlink():
            raise ValueError("--output appeared during preparation; refusing to replace it")
        os.rename(stage, output)
    finally:
        # Only remove our own unpublished temporary directory.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.prepare-"):
            shutil.rmtree(stage)
    return {
        "source_format": source_format, "question_files": len(questions),
        "category_summaries": len(grouped), "output": output.as_posix(),
        **dict(relocator.counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Upstream json_clean directory or questions.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New json_clean directory, outside the input")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Explicit asset path prefix replacement; repeatable")
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output, args.path_map)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
