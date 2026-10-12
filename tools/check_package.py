"""Check source syntax, configuration files, and local documentation links."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def check(root: Path) -> dict:
    root = root.resolve()
    counts = {'python_files': 0, 'shell_files': 0, 'toml_files': 0,
              'json_files': 0, 'readme_links': 0}
    errors = []
    files = []
    excluded = {'.git', '__pycache__', '.runtime', '.pytest_cache', '.ruff_cache',
                '.venv', '.venv-understanding', '.venv-manipulation', 'third_party', 'checkpoints', 'outputs'}
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(name for name in names if name not in excluded
                          and not name.startswith('eval_runs')
                          and not (Path(directory) / name).is_symlink())
        files.extend(Path(directory) / name for name in filenames)
    for file in sorted(files):
        relative = file.relative_to(root).as_posix()
        try:
            if file.suffix == '.py':
                counts['python_files'] += 1
                ast.parse(file.read_text(encoding='utf-8-sig'), filename=relative)
            elif file.suffix == '.sh':
                counts['shell_files'] += 1
                run = subprocess.run(['bash', '-n', str(file)], capture_output=True, text=True)
                if run.returncode:
                    raise ValueError(run.stderr.strip())
            elif file.suffix == '.toml':
                counts['toml_files'] += 1
                tomllib.loads(file.read_text(encoding='utf-8-sig'))
            elif file.suffix == '.json':
                counts['json_files'] += 1
                json.loads(file.read_text(encoding='utf-8-sig'))
        except Exception as exc:
            errors.append({'file': relative, 'type': 'syntax', 'message': str(exc)})
    for relative in ('README.md', 'parallelworld_understanding/README.md', 'parallelworld_manipulation/README.md'):
        file = root / relative
        if not file.is_file():
            errors.append({'file': relative, 'type': 'missing_readme'})
            continue
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', file.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            counts['readme_links'] += 1
            if not (file.parent / target.split('#')[0]).is_file():
                errors.append({'file': relative, 'type': 'broken_link', 'target': target})
    return {'checks': counts, 'passed': not errors, 'errors': errors,
            'scope': 'Static source and documentation validation; no inference or simulator startup.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.root)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
