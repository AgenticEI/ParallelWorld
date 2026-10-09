"""Cosmos-Predict2 14B DROID I2V backend for ParallelWorld.

Launch this module with torchrun.  Rank zero owns the parallelworld RPC server while
all ranks participate in Cosmos context-parallel generation.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

_REPO_ROOT = Path(os.environ.get("PARALLELWORLD_REPO_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
DEFAULT_COSMOS_ROOT = os.environ.get(
    "PW_COSMOS_ROOT", str(_REPO_ROOT / "third_party" / "Cosmos-Predict2")
)
DEFAULT_CHECKPOINT_ROOT = os.environ.get(
    "PW_COSMOS_CHECKPOINTS", str(_REPO_ROOT / "checkpoints" / "cosmos-predict2-14b-droid")
)
DEFAULT_SAMPLING_STEPS = 20
MODEL_REPOSITORY = "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID"
MODEL_REVISION = "ebc29d30a3fab504bcc779a85bca073d14ad39f9"
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no "
    "motion, motion blur, over-saturation, shaky footage, low resolution, "
    "distorted robot geometry, extra robot arms, missing gripper fingers, "
    "camera movement, scene cuts, text, watermark, or unrealistic physics."
)
PROMPT_PREFIX = (
    "A multi-view video shows a Franka Panda robot performing a task. "
    "The video is split into four views: the top-left and top-right views show "
    "the robot from fixed external cameras, the bottom-left view shows the "
    "wrist-camera perspective, and the bottom-right view is black and inactive. "
    "All active views depict the same synchronized physical motion. The cameras "
    "remain fixed and the scene appearance remains unchanged. The robot task is: "
)


def _bootstrap_paths(cosmos_root: str, parallelworld_root: str) -> None:
    for path in (cosmos_root, parallelworld_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def required_checkpoint_paths(checkpoint_root: str | Path) -> dict[str, Path]:
    """Return the files needed by the DROID service without importing Cosmos."""
    root = Path(checkpoint_root).expanduser().resolve()
    model_root = root / MODEL_REPOSITORY
    return {
        "dit": model_root / "model-480p-16fps.pt",
        "tokenizer": model_root / "tokenizer/tokenizer.pth",
        "t5_config": root / "google-t5/t5-11b/config.json",
        "t5_weights": root / "google-t5/t5-11b/pytorch_model.bin",
    }


def validate_runtime_paths(
    *, cosmos_root: str | Path, parallelworld_root: str | Path, checkpoint_root: str | Path
) -> dict[str, str]:
    """Fail fast with an actionable list of missing runtime inputs."""
    required = {
        "cosmos checkout": Path(cosmos_root).expanduser().resolve()
        / "cosmos_predict2/__init__.py",
        "parallelworld checkout": Path(parallelworld_root).expanduser().resolve() / "parallelworld_manipulation/__init__.py",
        **required_checkpoint_paths(checkpoint_root),
    }
    missing = {name: path for name, path in required.items() if not path.is_file()}
    if missing:
        details = "\n".join(f"  - {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            "Cosmos-Predict2 DROID runtime is incomplete:\n"
            f"{details}\n"
            f"The model is gated at https://huggingface.co/{MODEL_REPOSITORY}; "
            "accept its terms and authenticate before downloading it."
        )
    return {name: str(path) for name, path in required.items()}


def configure_droid_tokenizer(config: Any, checkpoint_root: str | Path) -> Path:
    """Point the base Video2World config at the identical DROID tokenizer copy."""
    tokenizer_path = required_checkpoint_paths(checkpoint_root)["tokenizer"]
    config.tokenizer.vae_pth = str(tokenizer_path)
    return tokenizer_path


def keep_text_encoder_on_cpu(pipe: Any) -> None:
    """Correct upstream offload behavior so only embeddings move to CUDA."""

    def encode_prompt(
        self: Any,
        prompts: str | list[str],
        max_length: int | None = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        return self.text_encoder.encode_prompts(
            prompts,
            max_length=max_length,
            return_mask=return_mask,
        )

    pipe.encode_prompt = MethodType(encode_prompt, pipe)


def _letterbox_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image without changing its aspect ratio, padding as needed."""
    target_w, target_h = size
    source = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    scale = min(target_w / source.width, target_h / source.height)
    resized_w = max(1, min(target_w, round(source.width * scale)))
    resized_h = max(1, min(target_h, round(source.height * scale)))
    resized = source.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h))
    canvas.paste(resized, ((target_w - resized_w) // 2, (target_h - resized_h) // 2))
    return np.asarray(canvas)


def _restore_letterboxed_view(
    frames: np.ndarray, source_shape: tuple[int, int]
) -> np.ndarray:
    """Remove adapter padding and return frames at the source view resolution."""
    source_h, source_w = source_shape
    frame_h, frame_w = frames.shape[1:3]
    scale = min(frame_w / source_w, frame_h / source_h)
    content_w = max(1, min(frame_w, round(source_w * scale)))
    content_h = max(1, min(frame_h, round(source_h * scale)))
    left = (frame_w - content_w) // 2
    top = (frame_h - content_h) // 2
    cropped = frames[:, top : top + content_h, left : left + content_w]
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (source_w, source_h), Image.Resampling.LANCZOS
                )
            )
            for frame in cropped
        ]
    )


def _make_droid_mosaic(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Map parallelworld's two views onto the four-view layout used by DROID training."""
    cell_w, cell_h = 320, 180
    front_cell = _letterbox_rgb(front, (cell_w, cell_h))
    wrist_cell = _letterbox_rgb(wrist, (cell_w, cell_h))
    mosaic = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)
    mosaic[:cell_h, :cell_w] = front_cell
    # LIBERO has one external view. Repeating it is less misleading than
    # inventing a second camera or leaving a view the model expects blank.
    mosaic[:cell_h, cell_w:] = front_cell
    mosaic[cell_h:, :cell_w] = wrist_cell
    return mosaic


def _split_droid_video(
    video: torch.Tensor,
    horizon: int,
    *,
    front_shape: tuple[int, int] | None = None,
    wrist_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if video.ndim == 5:
        video = video[0]
    frames = video.detach().float().cpu()
    if frames.min() < -0.5:
        frames = (frames + 1.0) / 2.0
    frames = frames.clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().byte().numpy()
    frames = frames[: max(1, int(horizon))]
    height, width = frames.shape[1:3]
    half_h, half_w = height // 2, width // 2
    front = frames[:, :half_h, :half_w]
    wrist = frames[:, half_h:, :half_w]
    if front_shape is not None:
        front = _restore_letterboxed_view(front, front_shape)
    if wrist_shape is not None:
        wrist = _restore_letterboxed_view(wrist, wrist_shape)
    return front, wrist


class CosmosDroidBackend:
    def __init__(
        self,
        *,
        checkpoint_root: str,
        sampling_steps: int,
        guidance: float,
    ) -> None:
        validate_runtime_paths(
            cosmos_root=next(
                (
                    path
                    for path in sys.path
                    if (Path(path) / "cosmos_predict2").is_dir()
                ),
                DEFAULT_COSMOS_ROOT,
            ),
            parallelworld_root=next(
                (path for path in sys.path if (Path(path) / "parallelworld_manipulation").is_dir()),
                Path(__file__).resolve().parents[2],
            ),
            checkpoint_root=checkpoint_root,
        )
        from cosmos_predict2.configs.base.config_video2world import (
            get_cosmos_predict2_video2world_pipeline,
        )
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from imaginaire.constants import get_cosmos_predict2_gr00t_checkpoint
        from imaginaire.utils import distributed, misc
        from megatron.core import parallel_state

        self._parallel_state = parallel_state
        self._sampling_steps = max(1, int(sampling_steps))
        self._guidance = float(guidance)
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self._world_size > 1:
            distributed.init()
            parallel_state.initialize_model_parallel(
                context_parallel_size=self._world_size
            )

        config = get_cosmos_predict2_video2world_pipeline(
            model_size="14B", resolution="480", fps=16
        )
        config.prompt_refiner_config.enabled = False
        config.guardrail_config.enabled = False
        configure_droid_tokenizer(config, checkpoint_root)
        dit_path = get_cosmos_predict2_gr00t_checkpoint(
            gr00t_variant="droid",
            model_size="14B",
            resolution="480",
            fps=16,
            aspect_ratio="16:9",
        )
        required = list(required_checkpoint_paths(checkpoint_root).values())
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Cosmos checkpoint files: " + ", ".join(missing)
            )

        misc.set_random_seed(seed=0, by_rank=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self._pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
            load_prompt_refiner=False,
            offload_text_encoder=True,
            downcast_text_encoder=True,
        )
        keep_text_encoder_on_cpu(self._pipe)

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        return self._world_size

    def _latent_frames(self, requested_pixel_frames: int) -> int:
        latent = max(2, math.ceil((max(1, requested_pixel_frames) - 1) / 4) + 1)
        if self._world_size > 1:
            latent = math.ceil(latent / self._world_size) * self._world_size
        return latent

    def generate(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        *,
        instruction: str,
        horizon: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        self._pipe.config.state_t = self._latent_frames(horizon)
        mosaic = _make_droid_mosaic(front, wrist)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temp_path = handle.name
            Image.fromarray(mosaic).save(temp_path)
            video = self._pipe(
                input_path=temp_path,
                prompt=PROMPT_PREFIX + str(instruction),
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                aspect_ratio="16:9",
                num_conditional_frames=1,
                guidance=self._guidance,
                num_sampling_step=self._sampling_steps,
                seed=int(seed),
            )
            if video is None:
                raise RuntimeError("Cosmos-Predict2 returned no video")
            if self.rank != 0:
                return None
            return _split_droid_video(
                video,
                horizon,
                front_shape=front.shape[:2],
                wrist_shape=wrist.shape[:2],
            )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def rollout(self, **kwargs: Any) -> dict[str, np.ndarray] | None:
        samples = max(1, int(kwargs.get("samples", 1)))
        fronts: list[np.ndarray] = []
        wrists: list[np.ndarray] = []
        for sample in range(samples):
            result = self.generate(
                np.asarray(kwargs["front"], dtype=np.uint8),
                np.asarray(kwargs["wrist"], dtype=np.uint8),
                instruction=str(kwargs["instruction"]),
                horizon=int(kwargs.get("horizon", 22)),
                seed=int(kwargs.get("seed", 0)) + sample * 1009,
            )
            if self.rank == 0:
                assert result is not None
                front_frames, wrist_frames = result
                fronts.append(front_frames)
                wrists.append(wrist_frames)
        if self.rank != 0:
            return None
        return {
            "front": np.asarray(fronts, dtype=np.uint8),
            "wrist": np.asarray(wrists, dtype=np.uint8),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "model": "Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "sampling_steps": self._sampling_steps,
            "guidance": self._guidance,
            "world_size": self.world_size,
            "layout": "DROID four-view mosaic from LIBERO front+wrist",
        }

    def broadcast(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if self._world_size == 1:
            return payload
        objects = [payload]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def worker_loop(self) -> None:
        while True:
            payload = self.broadcast(None)
            if not payload or payload.get("command") == "shutdown":
                return
            if payload.get("command") != "rollout":
                raise ValueError(f"Unknown worker command: {payload.get('command')!r}")
            self.rollout(**payload["kwargs"])

    def close(self) -> None:
        if self._parallel_state.is_initialized():
            self._parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-root", default=DEFAULT_COSMOS_ROOT)
    parser.add_argument(
        "--parallelworld-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--sampling-steps", type=int, default=DEFAULT_SAMPLING_STEPS)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18140)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the checkout and checkpoint layout without loading the model.",
    )
    args = parser.parse_args()

    _bootstrap_paths(args.cosmos_root, args.parallelworld_root)
    checked = validate_runtime_paths(
        cosmos_root=args.cosmos_root,
        parallelworld_root=args.parallelworld_root,
        checkpoint_root=args.checkpoint_root,
    )
    if args.check_only:
        print("Cosmos-Predict2 DROID runtime check passed")
        for name, path in checked.items():
            print(f"{name}: {path}")
        return
    os.environ.setdefault(
        "COSMOS_PREDICT2_ARGS", f"--checkpoints {Path(args.checkpoint_root).resolve()}"
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from parallelworld_manipulation.utils.rpc import RpcFacade

    backend = CosmosDroidBackend(
        checkpoint_root=args.checkpoint_root,
        sampling_steps=args.sampling_steps,
        guidance=args.guidance,
    )
    if backend.rank != 0:
        try:
            backend.worker_loop()
        finally:
            backend.close()
        return

    class Facade(RpcFacade):
        def _dispatch(self, method: str, call_args: tuple, kwargs: dict) -> Any:
            del call_args
            if method == "model_info":
                return {
                    **backend.model_info(),
                    "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
                }
            if method != "rollout":
                raise ValueError(f"Unknown RPC method: {method!r}")
            backend.broadcast({"command": "rollout", "kwargs": kwargs})
            return backend.rollout(**kwargs)

    facade = Facade()

    def request_shutdown(_signum: int, _frame: Any) -> None:
        facade._shutdown_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        try:
            backend.broadcast({"command": "shutdown"})
        finally:
            backend.close()


if __name__ == "__main__":
    main()
