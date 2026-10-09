"""Write an parallelworld_understanding metrics table using an atomic file update."""
import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path


def update(metrics_root, output):
    rows = []
    for file in sorted(metrics_root.glob('*/*.json')):
        try:
            data = json.loads(file.read_text())
        except json.JSONDecodeError:
            continue
        accuracy = data.get('accuracy_percent')
        strict = data.get('strict_total_accuracy_percent')
        rows.append([data.get('big_task', file.parent.name), data.get('small_task', file.stem),
                     str(data.get('correct_count', 0)), str(data.get('evaluated_results', 0)),
                     str(data.get('total_questions', 0)), '—' if accuracy is None else f'{accuracy:.2f}%',
                     '—' if strict is None else f'{strict:.2f}%'])
    text = '# parallelworld_understanding evaluation\n\n'
    text += '| Category | Subcategory | Correct | Evaluated | Total | Accuracy | Total-set accuracy |\n'
    text += '|---|---|---:|---:|---:|---:|---:|\n'
    text += ''.join('| '+' | '.join(row)+' |\n' for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name('.'+output.name+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        os.replace(temporary, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)  # Accepted by the suite launcher.
    parser.add_argument('--metrics-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    update(args.metrics_root, args.output)
