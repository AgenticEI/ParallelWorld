#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_dataset_file(repo_root: Path) -> None:
    dataset_file = repo_root / "data/questions.jsonl"
    if not dataset_file.exists():
        raise SystemExit(f"Missing {dataset_file}")
    if dataset_file.stat().st_size == 0:
        raise SystemExit(f"{dataset_file} is empty. Regenerate it before uploading.")

    missing_assets: set[str] = set()
    with dataset_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
                image_paths = json.loads(row.get("image_paths_json") or "[]")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise SystemExit(f"Invalid dataset row at {dataset_file}:{line_number}: {exc}") from exc
            if not isinstance(image_paths, list):
                raise SystemExit(f"image_paths_json must contain a list at {dataset_file}:{line_number}")
            for raw_path in image_paths:
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                relative_path = Path(raw_path)
                is_packaged_asset = (
                    not relative_path.is_absolute()
                    and ".." not in relative_path.parts
                    and bool(relative_path.parts)
                    and relative_path.parts[0] == "assets"
                )
                if not is_packaged_asset or not (repo_root / relative_path).is_file():
                    missing_assets.add(raw_path)

    if missing_assets:
        examples = "\n".join(f"  {path}" for path in sorted(missing_assets)[:20])
        raise SystemExit(
            f"Dataset upload aborted: {len(missing_assets)} referenced image assets are missing.\n"
            f"First missing assets:\n{examples}\n"
            "Regenerate the HF export before uploading."
        )


def main() -> int:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description="Upload the Croissant-friendly ESI-Bench table to Hugging Face.")
    parser.add_argument("--repo-id", default="ESI-Bench/esi-bench")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private if it does not exist.")
    parser.add_argument("--commit-message", default="Replace with Croissant-friendly questions table")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_dataset_file(repo_root)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=repo_root,
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "assets/**",
            "data/questions.jsonl",
            "scripts/export_hf_questions.py",
            "scripts/upload_hf_dataset.py",
        ],
        delete_patterns=[
            "assets/**",
            "data/**",
            "dataset/**",
            "scripts/**",
        ],
        commit_message=args.commit_message,
    )
    print(f"Uploaded {repo_root} to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
