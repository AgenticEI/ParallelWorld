from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize verifier accuracy for one metadata task.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pipeline-variant", default="parallelworld")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--big-task", default=None)
    parser.add_argument("--small-task", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_paths(metadata_path: Path) -> list[str]:
    metadata = load_json(metadata_path)
    paths = metadata.get("json_paths") if isinstance(metadata, dict) else metadata
    if not isinstance(paths, list):
        raise ValueError(f"Metadata has no json_paths list: {metadata_path}")
    return [str(path) for path in paths]


def resolve_question_path(raw_path: str, metadata_path: Path, json_root: Path) -> Path:
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [json_root / raw, metadata_path.parent / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def result_source_path(result: dict[str, Any], result_path: Path) -> Path | None:
    source = result.get("source_json")
    if not source:
        return None
    path = Path(str(source)).expanduser()
    if not path.is_absolute():
        path = result_path.parent / path
    return path.resolve()


def collect_results(results_root: Path, pipeline_variant: str) -> dict[Path, tuple[Path, dict[str, Any]]]:
    pipeline_root = results_root / pipeline_variant
    # Verifier variants add a pipeline directory, while the original pipeline
    # writes task-specific directories directly below results_root.
    if not pipeline_root.exists() and pipeline_variant == "baseline_active":
        pipeline_root = results_root
    matched: dict[Path, tuple[Path, dict[str, Any]]] = {}
    if not pipeline_root.exists():
        return matched
    for result_path in pipeline_root.rglob("*_answer.json"):
        try:
            result = load_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        source_path = result_source_path(result, result_path)
        if source_path is None:
            continue
        previous = matched.get(source_path)
        if previous is None or result_path.stat().st_mtime >= previous[0].stat().st_mtime:
            matched[source_path] = (result_path, result)
    return matched


def percentage(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def is_not_sure(answer: Any) -> bool:
    normalized = " ".join(str(answer or "").strip().lower().replace("_", " ").replace("-", " ").split())
    return normalized == "not sure"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    raw_paths = metadata_paths(args.metadata)
    questions = [resolve_question_path(path, args.metadata, args.json_root) for path in raw_paths]
    results = collect_results(args.results_root, args.pipeline_variant)

    correct_indices: list[int] = []
    incorrect_indices: list[int] = []
    unevaluated_indices: list[int] = []
    missing_indices: list[int] = []
    not_sure_indices: list[int] = []
    not_sure_incorrect_indices: list[int] = []
    records: list[dict[str, Any]] = []

    for index, question_path in enumerate(questions):
        item = results.get(question_path)
        if item is None:
            missing_indices.append(index)
            continue

        result_path, result = item
        correct = result.get("correct")
        if correct is True:
            correct_indices.append(index)
        elif correct is False:
            incorrect_indices.append(index)
        else:
            unevaluated_indices.append(index)

        gt_answer = None
        try:
            question = load_json(question_path)
            if isinstance(question, dict):
                gt_answer = question.get("answer")
        except (OSError, json.JSONDecodeError):
            pass
        final_answer = result.get("final_answer")
        prediction = final_answer.get("answer") if isinstance(final_answer, dict) else final_answer
        prediction_is_not_sure = is_not_sure(prediction)
        if prediction_is_not_sure:
            not_sure_indices.append(index)
            if correct is False:
                not_sure_incorrect_indices.append(index)
        records.append(
            {
                "question_index": index,
                "question_json": str(question_path),
                "result_json": str(result_path.resolve()),
                "ground_truth": gt_answer,
                "prediction": prediction,
                "is_not_sure": prediction_is_not_sure,
                "correct": correct,
            }
        )

    total = len(questions)
    completed = len(records)
    evaluated = len(correct_indices) + len(incorrect_indices)
    summary = {
        "big_task": args.big_task,
        "small_task": args.small_task,
        "pipeline_variant": args.pipeline_variant,
        "metadata": str(args.metadata.resolve()),
        "results_root": str(args.results_root.resolve()),
        "total_questions": total,
        "completed_results": completed,
        "evaluated_results": evaluated,
        "correct_count": len(correct_indices),
        "incorrect_count": len(incorrect_indices),
        "unevaluated_count": len(unevaluated_indices),
        "missing_count": len(missing_indices),
        "not_sure_count": len(not_sure_indices),
        "not_sure_incorrect_count": len(not_sure_incorrect_indices),
        "other_incorrect_count": len(incorrect_indices) - len(not_sure_incorrect_indices),
        "not_sure_share_of_incorrect_percent": percentage(len(not_sure_incorrect_indices), len(incorrect_indices)),
        "accuracy_percent": percentage(len(correct_indices), evaluated),
        "coverage_percent": percentage(completed, total),
        "strict_total_accuracy_percent": percentage(len(correct_indices), total),
        "correct_indices": correct_indices,
        "incorrect_indices": incorrect_indices,
        "unevaluated_indices": unevaluated_indices,
        "missing_indices": missing_indices,
        "not_sure_indices": not_sure_indices,
        "not_sure_incorrect_indices": not_sure_incorrect_indices,
        "records": records,
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    accuracy = summary["accuracy_percent"]
    coverage = summary["coverage_percent"]
    strict_accuracy = summary["strict_total_accuracy_percent"]
    accuracy_text = "N/A" if accuracy is None else f"{accuracy:.2f}%"
    coverage_text = "N/A" if coverage is None else f"{coverage:.2f}%"
    strict_accuracy_text = "N/A" if strict_accuracy is None else f"{strict_accuracy:.2f}%"
    print("============================================================")
    print(f"Task accuracy: {summary['big_task']} / {summary['small_task']}")
    print(
        f"correct={summary['correct_count']} incorrect={summary['incorrect_count']} "
        f"evaluated={summary['evaluated_results']} completed={summary['completed_results']} "
        f"missing={summary['missing_count']} total={summary['total_questions']}"
    )
    print(
        f"not_sure={summary['not_sure_count']} "
        f"not_sure_incorrect={summary['not_sure_incorrect_count']} "
        f"other_incorrect={summary['other_incorrect_count']}"
    )
    print(
        f"evaluated_accuracy={accuracy_text} "
        f"strict_total_accuracy={strict_accuracy_text} coverage={coverage_text}"
    )
    print(f"summary_json={args.output}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
