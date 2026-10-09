"""Resolve model roles for one episode without initializing robot or model clients."""
import argparse
import json
from pathlib import Path


def resolve(profile, suite, task, seed, planner, verifier, enhancer, image):
    if not isinstance(profile, dict):
        raise ValueError('Model profile must be a JSON object.')
    roles = {'planner_model': planner, 'verifier_model': verifier, 'prompt_enhancer_model': enhancer, 'image_model': image}
    if profile.get('suite') == suite:
        defaults = profile.get('default', {})
        overrides = profile.get('trial_overrides', {})
        if not isinstance(defaults, dict) or not isinstance(overrides, dict):
            raise ValueError('Model defaults and trial_overrides must be JSON objects.')
        for layer in [defaults, overrides.get(f't{task}_s{seed}', {})]:
            if not isinstance(layer, dict):
                raise ValueError('Each trial model override must be a JSON object.')
            unknown = set(layer) - set(roles)
            if unknown:
                raise ValueError('Unknown model role fields: '+', '.join(sorted(unknown)))
            roles.update(layer)
    if any(not isinstance(v, str) or not v or any(c.isspace() or ord(c) < 32 for c in v) for v in roles.values()):
        raise ValueError('Model identifiers must be nonempty strings without whitespace or control characters.')
    if roles['image_model'] != image:
        raise ValueError('Per-episode image model must match the batch image service.')
    return [roles['planner_model'], roles['verifier_model'], roles['prompt_enhancer_model']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--task', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--planner-default', required=True)
    parser.add_argument('--verifier-default', required=True)
    parser.add_argument('--enhancer-default', required=True)
    parser.add_argument('--image-model', required=True)
    args = parser.parse_args()
    try:
        roles = resolve(json.loads(args.config.read_text(encoding='utf-8-sig')), args.suite, args.task, args.seed,
                        args.planner_default, args.verifier_default, args.enhancer_default, args.image_model)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print('\n'.join(roles))


if __name__ == '__main__':
    main()
