"""Run a batch and stop its entire process group on persisted quota errors."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ["PW_OUTPUT_ROOT"]).resolve()
PATTERN = re.compile(r"额度不足|余额不足|insufficient[_ -]?(?:user[_ -]?)?quota|402 Payment Required|quota (?:is )?exceeded", re.I)


def quota_error(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"world_model_error", "fallback_reason", "error", "reason"} and isinstance(child, str) and PATTERN.search(child):
                return child
            found = quota_error(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = quota_error(child)
            if found:
                return found
    return None


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "launcher.out").open("ab") as log:
        child = subprocess.Popen(["bash", "evaluate_manipulation.sh", "gpt_image"], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    (ROOT / "supervisor.pid").write_text(str(os.getpid()))
    seen = {}

    def stop(*_):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(8)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(143)))
    signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(130)))
    while child.poll() is None:
        for path in ROOT.glob("libero_*/*/parallelworld_trace.json"):
            stamp = path.stat().st_mtime_ns
            if seen.get(path) == stamp:
                continue
            try:
                found = quota_error(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
            seen[path] = stamp
            if found:
                (ROOT / "API_QUOTA_EXHAUSTED").write_text(json.dumps({"file": str(path), "error": found, "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False))
                stop()
                return 42
        time.sleep(5)
    stop()
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
