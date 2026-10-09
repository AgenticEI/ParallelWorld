"""Validate and summarize completed LIBERO-Pro evaluation outputs."""
import argparse
import json
from pathlib import Path

SUITES = [('libero_spatial_task', 'Spatial T'), ('libero_spatial_swap', 'Spatial S'),
          ('libero_object_task', 'Object T'), ('libero_object_swap', 'Object S'),
          ('libero_goal_task', 'Goal T'), ('libero_goal_swap', 'Goal S'),
          ('libero_10_task', 'LIBERO-10 T'), ('libero_10_swap', 'LIBERO-10 S')]


def final_outcome(trial):
    state = trial / 'states.json'
    if not state.is_file():
        return None
    data = json.loads(state.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('steps', []), list):
        raise ValueError(f'Invalid state trajectory: {state}')
    steps = data.get('steps', [])
    if not steps:
        return None
    if not isinstance(steps[-1], dict):
        raise ValueError(f'Invalid final state: {state}')
    return steps[-1].get('terminated')


def summarize(root):
    """Read one or more batch roots; each suite must have one complete trial grid."""
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    if not roots:
        raise ValueError('At least one results root is required.')
    roots = [Path(path).expanduser().resolve(strict=True) for path in roots]
    if any(not path.is_dir() for path in roots):
        raise ValueError('Each results root must be an existing directory.')
    if len(set(roots)) != len(roots):
        raise ValueError('Duplicate results roots are not allowed.')

    rows = []
    for suite, label in SUITES:
        directories = [path / suite for path in roots if (path / suite).is_dir()]
        if not directories:
            continue
        if len(directories) != 1:
            raise ValueError(f'Suite {suite} appears in multiple results roots; select one evaluation.')
        suite_root = directories[0]
        successes = 0
        for task in range(10):
            for seed in range(1, 11):
                trial = suite_root / f't{task}_s{seed}'
                audit = trial / f'{suite.removeprefix("libero_")}_t{task}_s{seed}.json'
                if not audit.is_file():
                    raise ValueError(f'Incomplete evaluation: missing episode audit {audit}')
                data = json.loads(audit.read_text(encoding='utf-8'))
                if not isinstance(data, dict):
                    raise ValueError(f'Invalid audit object: {audit}')
                if not isinstance(data.get('terminated'), bool):
                    raise ValueError(f'Non-boolean task outcome: {audit}')
                if (type(data.get('task_id')) is not int or type(data.get('seed')) is not int
                    or (data.get('suite'), data.get('task_id'), data.get('seed')) != (suite, task, seed)):
                    raise ValueError(f'Trial identity mismatch: {audit}')
                # This command accepts direct simulator audits only.
                details = data.get('consolidation', {})
                if not isinstance(details, dict):
                    raise ValueError(f'Invalid audit provenance: {audit}')
                if details.get('synthesized') is True or details.get('source_audit_missing') is True:
                    raise ValueError(f'Synthetic audits are not supported; use original evaluation outputs: {audit}')
                state_outcome = final_outcome(trial)
                if not isinstance(state_outcome, bool):
                    raise ValueError(f'Incomplete evaluation: audit requires a nonempty trajectory and boolean final outcome: {audit}')
                if state_outcome is not data['terminated']:
                    raise ValueError(f'Audit/state mismatch: {audit}')
                successes += int(data['terminated'])
        rows.append({'suite': suite, 'label': label, 'trials': 100,
                     'successes': successes, 'failures': 100 - successes,
                     'success_rate_percent': float(successes)})
    if not rows:
        raise ValueError('No recognized LIBERO-Pro suite directories were found.')
    return {'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', action='append', required=True, type=Path,
                        help='Evaluation batch root containing suite directories; repeat for distinct batches.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.results_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
