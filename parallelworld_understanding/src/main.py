from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


_CARB_TASK_THREAD_SETTING = "--/plugins/carb.tasking.plugin/threadCount="


def configure_kit_threading() -> None:
    raw_count = os.environ.get("PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS")
    if raw_count is None:
        return
    try:
        thread_count = int(raw_count)
    except ValueError as exc:
        raise ValueError("PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS must be a positive integer") from exc
    if thread_count < 1:
        raise ValueError("PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS must be a positive integer")
    if not any(arg.startswith(_CARB_TASK_THREAD_SETTING) for arg in sys.argv):
        sys.argv.append(f"{_CARB_TASK_THREAD_SETTING}{thread_count}")


def parse_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pipeline-variant",
        choices=[
            "baseline_active",
            "baseline_passive",
            "branching",
            "parallelworld",
        ],
        default="parallelworld",
        help="Select an ESI-Bench evaluation pipeline; parallelworld is the default method.",
    )
    args, remaining = parser.parse_known_args(argv)
    return args.pipeline_variant, remaining


def main(argv: list[str] | None = None) -> int:
    parallelworld_understanding_dir = Path(__file__).resolve().parent / "parallelworld_understanding"
    sys.path.insert(0, str(parallelworld_understanding_dir))

    pipeline_variant, remaining = parse_args(argv)
    configure_kit_threading()
    pipeline_module = importlib.import_module(pipeline_variant)

    return pipeline_module.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
