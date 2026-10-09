"""Stop only this launcher's descendants on a fatal image response."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def stop_children(root):
    rows = subprocess.check_output(['ps', '-eo', 'pid,ppid'], text=True).splitlines()[1:]
    parents = {int(row.split()[0]): int(row.split()[1]) for row in rows}
    targets = {root}
    while True:
        new = {p for p, parent in parents.items() if parent in targets} - targets
        if not new:
            break
        targets |= new
    targets -= {root, os.getpid()}
    identities = {}
    for pid in targets:
        try:
            identities[pid] = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    time.sleep(5)
    for pid, identity in identities.items():
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if fields[19] == identity and fields[0] != 'Z':
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError):
            pass


if __name__ == '__main__':
    mode, root = sys.argv[1], int(sys.argv[2])
    if mode == 'stop':
        stop_children(root)
    elif mode == 'watch':
        marker = Path(sys.argv[3])
        while Path(f'/proc/{root}').exists():
            if marker.exists():
                print('FATAL IMAGE RESPONSE: stopping launcher; details in ' + str(marker), flush=True)
                os.kill(root, signal.SIGTERM)
                break
            time.sleep(1)
