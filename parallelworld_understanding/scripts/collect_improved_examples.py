#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "outputs" / "baseline_active"
DEFAULT_VERIFY_DIRS = [
    PROJECT_ROOT / "outputs" / "parallelworld",
    PROJECT_ROOT / "outputs" / "parallelworld_top1_route",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "improved_examples"


@dataclass
class RecordEntry:
    run_name: str
    big_task: str
    small_task: str
    metrics_path: Path
    question_rel: str
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect tasks where the baseline run is wrong but a verify run is correct, "
            "then build a browsable folder with summaries and step links."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline output directory.",
    )
    parser.add_argument(
        "--verify-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_VERIFY_DIRS,
        help="Verify output directories to compare against the baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination folder for the browsable comparison bundle.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question_path(raw_path: str) -> str:
    path = str(raw_path).strip()
    markers = [
        "dataset/json_clean/",
        "ESI-Bench/dataset/json_clean/",
    ]
    for marker in markers:
        if marker in path:
            return path.split(marker, 1)[1]
    return Path(path).as_posix()


def resolve_repo_path(raw_path: str) -> Path | None:
    path = Path(str(raw_path).strip())
    if not str(path):
        return None
    try:
        if path.exists():
            return path.resolve()
    except PermissionError:
        pass

    text = str(path)
    if "ESI-Bench/" in text:
        suffix = text.split("ESI-Bench/", 1)[1]
        candidate = PROJECT_ROOT / suffix
        if candidate.exists():
            return candidate.resolve()

    if "dataset/json_clean/" in text:
        suffix = text.split("dataset/json_clean/", 1)[1]
        candidate = PROJECT_ROOT / "dataset" / "json_clean" / suffix
        if candidate.exists():
            return candidate.resolve()

    return None


def load_metrics(run_dir: Path) -> dict[str, RecordEntry]:
    metrics_root = run_dir / "metrics"
    if not metrics_root.exists():
        raise FileNotFoundError(f"Missing metrics directory: {metrics_root}")

    entries: dict[str, RecordEntry] = {}
    for metrics_path in sorted(metrics_root.rglob("*.json")):
        payload = read_json(metrics_path)
        big_task = str(payload.get("big_task", "")).strip()
        small_task = str(payload.get("small_task", "")).strip()
        for record in payload.get("records", []):
            question_rel = normalize_question_path(record.get("question_json", ""))
            entries[question_rel] = RecordEntry(
                run_name=run_dir.name,
                big_task=big_task,
                small_task=small_task,
                metrics_path=metrics_path.resolve(),
                question_rel=question_rel,
                record=record,
            )
    return entries


def safe_part(value: str) -> str:
    cleaned = value.strip().replace(" ", "_").replace("&", "and")
    return cleaned.replace("/", "_").replace("\\", "_")


def sample_output_dir(root: Path, verify_name: str, question_rel: str) -> Path:
    rel_path = Path(question_rel)
    if len(rel_path.parts) >= 5:
        big_task, small_task = rel_path.parts[0], rel_path.parts[1]
        remainder = rel_path.parts[2:]
    else:
        big_task = "unknown_big_task"
        small_task = "unknown_small_task"
        remainder = rel_path.parts
    return root / safe_part(verify_name) / safe_part(big_task) / safe_part(small_task) / Path(*remainder).with_suffix("")


def format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.2f}"
    return "n/a"


def render_baseline_history(history: list[dict[str, Any]]) -> str:
    lines = ["# Baseline Steps", ""]
    if not history:
        lines.append("No history found.")
        return "\n".join(lines) + "\n"

    for step in history:
        lines.append(f"## Step {step.get('step', '?')}")
        lines.append(f"- action: {step.get('action', 'n/a')}")
        lines.append(f"- answer: {step.get('answer', 'n/a')}")
        lines.append(f"- confidence: {format_float(step.get('confidence'))}")
        lines.append(f"- image: {step.get('image', 'n/a')}")
        reasoning = str(step.get("reasoning", "")).strip() or "n/a"
        lines.append(f"- reasoning: {reasoning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_verify_history(history: list[dict[str, Any]]) -> str:
    lines = ["# Verify Steps", ""]
    if not history:
        lines.append("No history found.")
        return "\n".join(lines) + "\n"

    for step in history:
        step_id = step.get("step", "?")
        lines.append(f"## Step {step_id}")
        lines.append(f"- requested_top_k: {step.get('requested_top_k', 'n/a')}")
        lines.append(f"- selected_count: {step.get('selected_count', 'n/a')}")
        selected_branch_ids = step.get("selected_branch_ids") or []
        if selected_branch_ids:
            lines.append(f"- selected_branch_ids: {', '.join(selected_branch_ids)}")
        checkpoint_answer = step.get("checkpoint_answer")
        if checkpoint_answer is not None:
            lines.append(f"- checkpoint_answer: {checkpoint_answer}")
        checkpoint_confidence = step.get("checkpoint_confidence")
        if checkpoint_confidence is not None:
            lines.append(f"- checkpoint_confidence: {format_float(checkpoint_confidence)}")
        verifier_reasoning = str(step.get("verifier_reasoning", "")).strip()
        if verifier_reasoning:
            lines.append(f"- verifier_reasoning: {verifier_reasoning}")
        checkpoint_reasoning = str(step.get("checkpoint_reasoning", "")).strip()
        if checkpoint_reasoning:
            lines.append(f"- checkpoint_reasoning: {checkpoint_reasoning}")
        evidence_branch_ids = step.get("answer_evidence_branch_ids") or []
        if evidence_branch_ids:
            lines.append(f"- answer_evidence_branch_ids: {', '.join(evidence_branch_ids)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def ensure_symlink(link_path: Path, target_path: Path | None) -> None:
    if target_path is None or not target_path.exists():
        return
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    relative_target = os.path.relpath(target_path, link_path.parent)
    link_path.symlink_to(relative_target)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def prediction_from_result(result: dict[str, Any], record: dict[str, Any]) -> Any:
    final_answer = result.get("final_answer")
    if isinstance(final_answer, dict) and "answer" in final_answer:
        return final_answer.get("answer")
    return record.get("prediction")


def create_sample_bundle(
    output_dir: Path,
    baseline_entry: RecordEntry,
    verify_entry: RecordEntry,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_result_path = resolve_repo_path(str(baseline_entry.record.get("result_json", "")))
    verify_result_path = resolve_repo_path(str(verify_entry.record.get("result_json", "")))

    baseline_result = read_json(baseline_result_path) if baseline_result_path else {}
    verify_result = read_json(verify_result_path) if verify_result_path else {}

    question_path = resolve_repo_path(str(verify_result.get("source_json") or baseline_entry.record.get("question_json", "")))
    baseline_step_dir = resolve_repo_path(str(baseline_result.get("step_image_dir", "")))
    verify_step_dir = resolve_repo_path(str(verify_result.get("step_image_dir", "")))
    verify_selected_dir = resolve_repo_path(str(verify_result.get("selected_step_image_dir", "")))

    summary = {
        "question_rel": baseline_entry.question_rel,
        "big_task": baseline_entry.big_task,
        "small_task": baseline_entry.small_task,
        "ground_truth": baseline_entry.record.get("ground_truth"),
        "baseline_prediction": baseline_entry.record.get("prediction"),
        "baseline_prediction_from_result": prediction_from_result(baseline_result, baseline_entry.record),
        "verify_prediction": verify_entry.record.get("prediction"),
        "verify_prediction_from_result": prediction_from_result(verify_result, verify_entry.record),
        "baseline_result_json": str(baseline_result_path) if baseline_result_path else "",
        "verify_result_json": str(verify_result_path) if verify_result_path else "",
        "question_json": str(question_path) if question_path else "",
        "baseline_step_dir": str(baseline_step_dir) if baseline_step_dir else "",
        "verify_step_dir": str(verify_step_dir) if verify_step_dir else "",
        "verify_selected_step_dir": str(verify_selected_dir) if verify_selected_dir else "",
        "verify_run": verify_entry.run_name,
    }

    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    summary_md = [
        f"# {baseline_entry.question_rel}",
        "",
        f"- verify_run: {verify_entry.run_name}",
        f"- big_task: {baseline_entry.big_task}",
        f"- small_task: {baseline_entry.small_task}",
        f"- ground_truth: {summary['ground_truth']}",
        f"- baseline_prediction: {summary['baseline_prediction']}",
        f"- verify_prediction: {summary['verify_prediction']}",
        f"- question_json: {summary['question_json'] or 'missing'}",
        f"- baseline_result_json: {summary['baseline_result_json'] or 'missing'}",
        f"- verify_result_json: {summary['verify_result_json'] or 'missing'}",
        f"- baseline_step_dir: {summary['baseline_step_dir'] or 'missing'}",
        f"- verify_step_dir: {summary['verify_step_dir'] or 'missing'}",
        f"- verify_selected_step_dir: {summary['verify_selected_step_dir'] or 'missing'}",
        "",
        "This folder is generated for tasks where the baseline answer is wrong and the verify answer is correct.",
        "",
    ]
    write_text(output_dir / "summary.md", "\n".join(summary_md))

    write_text(output_dir / "baseline_steps.md", render_baseline_history(baseline_result.get("history") or []))
    write_text(output_dir / "verify_steps.md", render_verify_history(verify_result.get("history") or []))

    ensure_symlink(output_dir / "question.json", question_path)
    ensure_symlink(output_dir / "baseline_result.json", baseline_result_path)
    ensure_symlink(output_dir / "verify_result.json", verify_result_path)
    ensure_symlink(output_dir / "baseline_step_images", baseline_step_dir)
    ensure_symlink(output_dir / "verify_step_images", verify_step_dir)
    ensure_symlink(output_dir / "verify_selected_step_images", verify_selected_dir)

    return summary


def write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "verify_run",
        "question_rel",
        "big_task",
        "small_task",
        "ground_truth",
        "baseline_prediction",
        "verify_prediction",
        "question_json",
        "baseline_result_json",
        "verify_result_json",
        "baseline_step_dir",
        "verify_step_dir",
        "verify_selected_step_dir",
        "bundle_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    baseline_dir = args.baseline.resolve()
    verify_dirs = [path.resolve() for path in args.verify_dirs]
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_entries = load_metrics(baseline_dir)

    manifest_rows: list[dict[str, str]] = []
    verify_counts: list[tuple[str, int]] = []

    for verify_dir in verify_dirs:
        verify_entries = load_metrics(verify_dir)
        corrected = []
        for question_rel, baseline_entry in baseline_entries.items():
            verify_entry = verify_entries.get(question_rel)
            if verify_entry is None:
                continue
            if baseline_entry.record.get("correct") is False and verify_entry.record.get("correct") is True:
                corrected.append((question_rel, baseline_entry, verify_entry))

        corrected.sort(key=lambda item: item[0])
        verify_counts.append((verify_dir.name, len(corrected)))

        for question_rel, baseline_entry, verify_entry in corrected:
            bundle_dir = sample_output_dir(output_root, verify_dir.name, question_rel)
            summary = create_sample_bundle(bundle_dir, baseline_entry, verify_entry)
            manifest_rows.append(
                {
                    "verify_run": verify_dir.name,
                    "question_rel": question_rel,
                    "big_task": baseline_entry.big_task,
                    "small_task": baseline_entry.small_task,
                    "ground_truth": str(summary["ground_truth"]),
                    "baseline_prediction": str(summary["baseline_prediction"]),
                    "verify_prediction": str(summary["verify_prediction"]),
                    "question_json": summary["question_json"],
                    "baseline_result_json": summary["baseline_result_json"],
                    "verify_result_json": summary["verify_result_json"],
                    "baseline_step_dir": summary["baseline_step_dir"],
                    "verify_step_dir": summary["verify_step_dir"],
                    "verify_selected_step_dir": summary["verify_selected_step_dir"],
                    "bundle_dir": str(bundle_dir),
                }
            )

    manifest_rows.sort(key=lambda row: (row["verify_run"], row["question_rel"]))
    write_manifest_csv(output_root / "manifest.csv", manifest_rows)
    write_text(output_root / "manifest.json", json.dumps(manifest_rows, indent=2, ensure_ascii=False) + "\n")

    readme_lines = [
        "# Corrected Task Step Browser",
        "",
        f"- baseline_run: {baseline_dir}",
        f"- total_bundles: {len(manifest_rows)}",
        "",
        "## Verify counts",
        "",
    ]
    for verify_name, count in verify_counts:
        readme_lines.append(f"- {verify_name}: {count}")
    readme_lines.extend(
        [
            "",
            "Each sample folder contains:",
            "",
            "- summary.md and summary.json",
            "- baseline_steps.md and verify_steps.md",
            "- symlinks to the original question/result json files",
            "- symlinks to baseline and verify step image folders",
            "",
            "Use manifest.csv if you want to filter or sort the corrected cases quickly.",
            "",
        ]
    )
    write_text(output_root / "README.md", "\n".join(readme_lines))

    print(f"Output written to: {output_root}")
    for verify_name, count in verify_counts:
        print(f"{verify_name}: {count}")
    print(f"total bundles: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
