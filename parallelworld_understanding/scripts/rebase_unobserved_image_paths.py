#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rebase(value: Any, old_root: str, new_root: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        changed = 0
        output = {}
        for key, item in value.items():
            output[key], item_changed = rebase(item, old_root, new_root)
            changed += item_changed
        return output, changed
    if isinstance(value, list):
        changed = 0
        output = []
        for item in value:
            rebased, item_changed = rebase(item, old_root, new_root)
            output.append(rebased)
            changed += item_changed
        return output, changed
    if isinstance(value, str) and value.startswith(f"{old_root}/"):
        return f"{new_root}/{value[len(old_root) + 1:]}", 1
    return value, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebase generated unobserved-change asset paths.")
    parser.add_argument("json_root", type=Path)
    parser.add_argument("old_root")
    parser.add_argument("new_root", type=Path)
    args = parser.parse_args()

    old_root = args.old_root.rstrip("/")
    new_root = str(args.new_root.expanduser().resolve()).rstrip("/")
    file_count = 0
    path_count = 0
    for path in sorted(args.json_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rebased, changed = rebase(payload, old_root, new_root)
        if not changed:
            continue
        path.write_text(json.dumps(rebased, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        file_count += 1
        path_count += changed

    print(f"Rebased {path_count} path(s) across {file_count} JSON file(s) to {new_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
