"""Download the exact checkpoints required by the parallelworld Cosmos DROID backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DROID_REPO = "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID"
DROID_REVISION = "ebc29d30a3fab504bcc779a85bca073d14ad39f9"
T5_REPO = "google-t5/t5-11b"
T5_REVISION = "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default=os.environ.get("PW_COSMOS_CHECKPOINTS", str(Path(__file__).resolve().parents[1] / "checkpoints" / "cosmos-predict2-14b-droid")),
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help="Skip the shared T5-11B text encoder when it is already complete.",
    )
    args = parser.parse_args()

    root = Path(args.checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pinned DROID snapshot {DROID_REVISION} into {root}")
    snapshot_download(
        repo_id=DROID_REPO,
        revision=DROID_REVISION,
        local_dir=root / DROID_REPO,
        allow_patterns=["model-480p-16fps.pt", "tokenizer/*", "config.json"],
    )
    if not args.skip_t5:
        print(f"Downloading pinned T5 snapshot {T5_REVISION}")
        snapshot_download(
            repo_id=T5_REPO,
            revision=T5_REVISION,
            local_dir=root / T5_REPO,
            ignore_patterns=["tf_model.h5", "flax_model.msgpack"],
        )
    print("Cosmos-Predict2 DROID checkpoints are ready")


if __name__ == "__main__":
    main()
