"""Wan2.2-5B-LIBERO native TI2V RPC service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from parallelworld_manipulation.utils.rpc import RpcFacade

_REPO_ROOT = Path(os.environ.get("PARALLELWORLD_REPO_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
WAN_ROOT = os.environ.get("PW_WAN_ROOT", str(_REPO_ROOT / "third_party" / "Wan2.2"))
BASE_CHECKPOINT = os.environ.get("PW_WAN_CHECKPOINTS", str(_REPO_ROOT / "checkpoints" / "Wan2.2-TI2V-5B"))
LIBERO_DIT = os.environ.get(
    "PW_WAN_DIT", str(_REPO_ROOT / "checkpoints" / "Wan2.2-5B-Libero" / "checkpoint.safetensors")
)

# Wan's upstream default negative prompt includes terms meaning "static" and
# "motionless".  Those terms fight the requested stable endpoint and can make
# a short robot action continue into an unrelated follow-up.  Keep this list
# quality/physics focused and avoid naming valid manipulation primitives.
WAN_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, overexposed, underexposed, noisy, "
    "compression artifacts, watermark, subtitles, text, camera motion, camera cut, "
    "scene transition, duplicated robot, extra limbs, malformed gripper, object "
    "deformation, interpenetration, unsupported floating objects, teleportation, "
    "multiple sequential actions, cyclic motion"
)

if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)


class WanLiberoWorldModelFacade(RpcFacade):
    def __init__(
        self,
        checkpoint_dir: str,
        *,
        finetuned_dit: str | None = None,
        frame_size: str = "832*480",
        sampling_steps: int = 20,
        frame_num: int = 49,
    ) -> None:
        super().__init__()
        import wan
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

        self._checkpoint_dir = checkpoint_dir
        self._finetuned_dit = finetuned_dit
        self._size = tuple(SIZE_CONFIGS.get(frame_size, (832, 480)))
        self._max_area = MAX_AREA_CONFIGS.get(frame_size, 399360)
        self._steps = int(sampling_steps)
        self._frame_num = max(5, int(frame_num))
        self._pipe = wan.WanTI2V(
            config=WAN_CONFIGS["ti2v-5B"],
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            rank=0,
            t5_cpu=True,
            convert_model_dtype=True,
        )

        if finetuned_dit is None and os.path.isfile(LIBERO_DIT):
            finetuned_dit = LIBERO_DIT
        if finetuned_dit:
            from safetensors.torch import load_file

            state = load_file(finetuned_dit, device="cpu")
            missing, unexpected = self._pipe.model.load_state_dict(state, strict=False)
            print(
                f"loaded LIBERO DiT: {len(state)} tensors, "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        del args
        if method == "model_info":
            return {
                "model": "Wan2.2-TI2V-5B-LIBERO",
                "checkpoint_dir": self._checkpoint_dir,
                "finetuned_dit": self._finetuned_dit,
                "sampling_steps": self._steps,
                "frame_num": self._frame_num,
                "frame_size": list(self._size),
                "views": "front-only I2V rollout",
                "negative_prompt": WAN_NEGATIVE_PROMPT,
            }
        if method == "rollout":
            return self.rollout(**kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def _view(
        self,
        image: np.ndarray,
        instruction: str,
        horizon: int,
        seed: int,
    ) -> np.ndarray:
        del horizon
        video = self._pipe.generate(
            input_prompt=str(instruction),
            n_prompt=WAN_NEGATIVE_PROMPT,
            img=Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB"),
            size=self._size,
            max_area=self._max_area,
            frame_num=self._frame_num,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=self._steps,
            guide_scale=5.0,
            seed=int(seed),
            offload_model=True,
        )
        arr = video.detach().float().cpu().numpy() if torch.is_tensor(video) else np.asarray(video)
        if arr.ndim == 4 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 3, 0))
        if arr.dtype != np.uint8:
            if arr.min() >= -1.01 and arr.max() <= 1.01:
                arr = (arr + 1.0) * 127.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def rollout(
        self,
        *,
        front: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        horizon: int = 49,
        samples: int = 1,
        stochastic: bool = False,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        del stochastic, wrist
        fronts = []
        for sample in range(max(1, int(samples))):
            sample_seed = int(seed) + sample * 1009
            fronts.append(self._view(front, instruction, horizon, sample_seed))
        return {"front": np.asarray(fronts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir", "--model-id", dest="checkpoint_dir", default=BASE_CHECKPOINT
    )
    parser.add_argument("--finetuned-dit", default=LIBERO_DIT)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument(
        "--frame-num",
        type=int,
        default=49,
        help="Wan2.2-5B-Libero is trained for 49-frame I2V rollouts.",
    )
    parser.add_argument("--frame-size", default="832*480")
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--cuda-device", type=int, default=None)
    args = parser.parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    WanLiberoWorldModelFacade(
        checkpoint_dir=args.checkpoint_dir,
        finetuned_dit=args.finetuned_dit,
        frame_size=args.frame_size,
        sampling_steps=args.sampling_steps,
        frame_num=args.frame_num,
    ).serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
