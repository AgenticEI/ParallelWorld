#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


TASKS = (
    ("Perceptual Grounding", "View Hallucination"),
    ("Perceptual Grounding", "Partial Occlusion"),
    ("Perceptual Grounding", "Material Transparency"),
    ("Physical Structure", "Rigid Containment"),
    ("Physical Structure", "Liquid Volume"),
    ("Physical Structure", "Deformable"),
    ("Physical Dynamics", "Inclined Plane"),
    ("Physical Dynamics", "Stacking & Stability"),
    ("Specular Reflection", "Reflection Authoring"),
    ("Specular Reflection", "Spatial Relations"),
    ("Specular Reflection", "Correspondence"),
    ("Spatial Relations", "Linear Alignment"),
    ("Spatial Relations", "Geometric Configuration"),
    ("Spatial Relations", "Physical Contact"),
    ("Metric Comparison", "Dimensional Size"),
    ("Metric Comparison", "Spatial Distance"),
    ("Cognitive Mapping", "Connectivity"),
    ("Cognitive Mapping", "Traversable Passage"),
    ("Cognitive Mapping", "Regional Boundary"),
    ("Cognitive Mapping", "Long-Term Navigation"),
    ("Enumerative Perception", "Counting w Occlusion"),
    ("Enumerative Perception", "Spatial Segmentation"),
    ("Enumerative Perception", "Merged Observation"),
    ("Enumerative Perception", "Category Ambiguity"),
    ("Enumerative Perception", "Structural Enclosure"),
    ("Enumerative Perception", "Illumination Variability"),
    ("Temporal Understanding", "Unobserved Change"),
    ("Temporal Understanding", "Agent Observation"),
    ("Action Sequencing", "Action Order Inference"),
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dataset_totals(dataset_root: Path) -> dict[tuple[str, str], int]:
    totals: dict[tuple[str, str], int] = {}
    by_category: dict[str, list[Any]] = {}
    for category, subcategory in TASKS:
        if category not in by_category:
            data = _read_json(dataset_root / f"{category}.json")
            paths = data.get("json_paths") if isinstance(data, dict) else data
            by_category[category] = paths if isinstance(paths, list) else []
        totals[(category, subcategory)] = sum(
            1
            for item in by_category[category]
            if len(Path(str(item)).parts) >= 2
            and Path(str(item)).parts[1] == subcategory
        )
    return totals


def _metric_files(metrics_root: Path) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    metrics: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    if not metrics_root.is_dir():
        return metrics
    for path in metrics_root.rglob("*.json"):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        category = data.get("big_task")
        subcategory = data.get("small_task")
        if not isinstance(category, str) or not isinstance(subcategory, str):
            continue
        key = (category, subcategory)
        previous = metrics.get(key)
        if previous is None or path.stat().st_mtime_ns >= previous[0].stat().st_mtime_ns:
            metrics[key] = (path, data)
    return metrics


def _is_complete(metrics_root: Path, metric_path: Path, metric: dict[str, Any], total: int) -> bool:
    try:
        relative_path = metric_path.relative_to(metrics_root)
    except ValueError:
        relative_path = Path(metric_path.name)
    progress_path = metrics_root.parent / "progress" / relative_path
    progress = _read_json(progress_path)
    if isinstance(progress, dict):
        next_index = progress.get("next_question_index")
        if isinstance(next_index, int) and next_index >= total:
            return True
        active_index = progress.get("active_question_index")
        progress_indices = [
            value for value in (next_index, active_index) if isinstance(value, int)
        ]
        furthest_index = max(progress_indices, default=-1)
        completed = metric.get("completed_results")
        missing = metric.get("missing_count")
        fully_accounted = (
            isinstance(completed, int)
            and isinstance(missing, int)
            and completed + missing >= total
        )
        try:
            metrics_are_current = metric_path.stat().st_mtime_ns >= progress_path.stat().st_mtime_ns
        except OSError:
            metrics_are_current = False
        if furthest_index >= total - 1 and fully_accounted and metrics_are_current:
            return True
    completed = metric.get("completed_results")
    return isinstance(completed, int) and completed >= total


def _format_result(
    key: tuple[str, str],
    total: int,
    metrics_root: Path,
    metrics: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
) -> str:
    entry = metrics.get(key)
    if entry is None or total <= 0:
        return "—"
    metric_path, metric = entry
    if not _is_complete(metrics_root, metric_path, metric, total):
        return "—"
    correct = metric.get("correct_count")
    if not isinstance(correct, int):
        return "—"
    completed = metric.get("completed_results")
    if not isinstance(completed, int) or completed <= 0:
        return "N/A (0/0)"
    accuracy = 100.0 * correct / completed
    return f"{accuracy:.2f}% ({correct}/{completed})"


def _render(
    dataset_root: Path,
    active_metrics_root: Path,
    passive_metrics_root: Path,
    parallelworld_metrics_root: Path,
) -> str:
    totals = _dataset_totals(dataset_root)
    active_metrics = _metric_files(active_metrics_root)
    passive_metrics = _metric_files(passive_metrics_root)
    parallelworld_metrics = _metric_files(parallelworld_metrics_root)
    lines = [
        "# ESI-Bench Results Summary",
        "",
        "> Results are populated when a subcategory completes; incomplete entries are shown as `—`.",
        "> Accuracy is `correct_count / completed_results`.",
        f"> Updated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "| Category | Subcategory | Questions N | Active | Passive | parallelworld |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in TASKS:
        category, subcategory = key
        total = totals.get(key, 0)
        active = _format_result(key, total, active_metrics_root, active_metrics)
        passive = _format_result(key, total, passive_metrics_root, passive_metrics)
        parallelworld = _format_result(key, total, parallelworld_metrics_root, parallelworld_metrics)
        lines.append(
            f"| {category} | {subcategory} | {total} | {active} | "
            f"{passive} | {parallelworld} |"
        )
    lines.append("")
    return "\n".join(lines)


def update_summary(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        content = _render(
            args.dataset_root.resolve(),
            args.active_metrics_root.resolve(),
            args.passive_metrics_root.resolve(),
            args.parallelworld_metrics_root.resolve(),
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output)
        output.chmod(0o644)


def _metrics_signature(roots: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                stat = path.stat()
            except OSError:
                continue
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(signature))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Update the ESI-Bench Markdown summary.")
    parser.add_argument("--dataset-root", type=Path, default=project_root / "dataset/json_clean")
    parser.add_argument(
        "--active-metrics-root",
        type=Path,
        default=project_root / "outputs/baseline_active/metrics",
    )
    parser.add_argument(
        "--passive-metrics-root",
        type=Path,
        default=project_root / "outputs/baseline_passive/metrics",
    )
    parser.add_argument(
        "--parallelworld-metrics-root",
        type=Path,
        default=project_root / "outputs/parallelworld_top1_route/metrics",
    )
    parser.add_argument("--output", type=Path, default=project_root / "RESULTS_SUMMARY.md")
    parser.add_argument("--watch", action="store_true", help="Refresh whenever a selected metrics tree changes.")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch polling interval in seconds.")
    args = parser.parse_args()

    update_summary(args)
    if not args.watch:
        return
    roots = (
        args.active_metrics_root.resolve(),
        args.passive_metrics_root.resolve(),
        args.parallelworld_metrics_root.resolve(),
    )
    signature = _metrics_signature(roots)
    while True:
        time.sleep(max(1.0, args.interval))
        current = _metrics_signature(roots)
        if current != signature:
            update_summary(args)
            signature = current


if __name__ == "__main__":
    main()
