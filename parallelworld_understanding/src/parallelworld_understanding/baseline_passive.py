from __future__ import annotations

import sys

import baseline_active as base


PIPELINE_VARIANT = "baseline_passive"


def main(argv: list[str] | None = None) -> int:
    passive_argv = list(sys.argv[1:] if argv is None else argv)
    passive_argv.extend(["--provider", "gpt", "--model", "gpt-5.4"])
    return base.main(
        passive_argv,
        pipeline_variant=PIPELINE_VARIANT,
        passive_mode=True,
        nest_variant_outputs=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
