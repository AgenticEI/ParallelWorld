# ruff: noqa: CPY001
"""GPT Image 2 terminal-state world-model service for ParallelWorld.

The existing ParallelWorld client asks a world model for a video tensor.  This
adapter uses the current observation as an Images API edit input and returns
one generated terminal frame per sample with the same RPC tensor contract.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from parallelworld_manipulation.utils.rpc import RpcFacade

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_QUALITY = "low"
DEFAULT_SIZE = "1024x1024"
DEFAULT_REFERENCE_PATHS = (
    Path(__file__).resolve().parents[2]
    / "resources/libero/robot_references/panda_full_reference.png",
    Path(__file__).resolve().parents[2]
    / "resources/libero/robot_references/panda_gripper_reference.png",
)

_TERMINAL_STATE_PROMPT = """\
Edit image 1 to show the immediate visible terminal state after the robot
performs exactly the action described below. Keep the fixed front camera,
scene geometry, robot identity, object identities, lighting, background, and
all unrelated objects consistent with image 1. Apply only the described robot
motion and its physically necessary object response. Produce one
photorealistic future front-camera observation. Do not add text, labels,
diagrams, borders, extra robot parts, or extra objects.

{reference_context}

Coordinate convention: image-left and image-right always mean the viewer's
left and right in Image 1 exactly as displayed. Never interpret left or right
from the robot's own perspective unless an explicit robot or world coordinate
frame is stated. If Image 1 contains a magenta marker labeled A, it is temporary
spatial guidance for the target center or projected terminal end-effector
waypoint identified in the action text. Follow that marker and any supplied
normalized (u, v) coordinate, then remove the marker and label from the
generated image.

Robot action:
{instruction}"""


def validate_gpt_image_2_size(size: str) -> str:
    """Validate the custom-size constraints documented for GPT Image 2."""
    normalized = str(size).strip().lower()
    if normalized == "auto":
        return normalized
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("size must be 'auto' or WIDTHxHEIGHT") from exc

    long_edge, short_edge = max(width, height), min(width, height)
    pixels = width * height
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if width % 16 or height % 16:
        raise ValueError("GPT Image 2 width and height must be multiples of 16")
    if long_edge > 3840:
        raise ValueError("GPT Image 2 edges must be at most 3840 pixels")
    if long_edge > short_edge * 3:
        raise ValueError("GPT Image 2 aspect ratio must not exceed 3:1")
    if not 655_360 <= pixels <= 8_294_400:
        raise ValueError("GPT Image 2 total pixels must be between 655360 and 8294400")
    return f"{width}x{height}"


def _rgb_array(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(
            f"expected {name} to have shape HxWx3 or HxWx4, got {array.shape!r}"
        )
    return np.asarray(Image.fromarray(array).convert("RGB"), dtype=np.uint8)


def _png_upload(image: np.ndarray, filename: str) -> tuple[str, bytes, str]:
    array = _rgb_array(image, name=filename)
    upload = io.BytesIO()
    Image.fromarray(array).save(upload, format="PNG")
    return filename, upload.getvalue(), "image/png"


def _plain_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return dict(usage)
    return {"value": str(usage)}


class GPTImageBackend:
    """Call the Images edit endpoint and decode terminal-state images."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_MODEL,
        quality: str = DEFAULT_QUALITY,
        size: str = DEFAULT_SIZE,
        request_timeout_s: float = 600.0,
        reference_paths: list[str | Path] | tuple[str | Path, ...] = (),
        fatal_error_file: str | Path | None = None,
    ) -> None:
        if quality not in {"low", "medium", "high"}:
            raise ValueError("quality must be one of: low, medium, high")
        if float(request_timeout_s) <= 0:
            raise ValueError("request_timeout_s must be positive")
        self._client = client
        self._fatal_error_file = Path(fatal_error_file) if fatal_error_file else None
        self._model = str(model)
        self._quality = quality
        self._size = validate_gpt_image_2_size(size)
        self._request_timeout_s = float(request_timeout_s)
        self._reference_uploads = []
        for reference_path in reference_paths:
            path = Path(reference_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"robot reference image not found: {path}")
            with Image.open(path) as reference:
                reference_rgb = np.asarray(reference.convert("RGB"), dtype=np.uint8)
            self._reference_uploads.append(_png_upload(reference_rgb, path.name))
        self._last_request_id: str | None = None
        self._last_usage: dict[str, Any] | None = None

    def _record_error(self, reason: str, *, insufficient_balance: bool = False) -> None:
        """Count failed requests across all batch workers; stop at 20 or on quota."""
        if self._fatal_error_file is not None:
            import fcntl

            self._fatal_error_file.parent.mkdir(parents=True, exist_ok=True)
            ledger = self._fatal_error_file.with_name("IMAGE_GENERATION_ERRORS.jsonl")
            # Separate opens plus flock serialize writes across GPU services.
            # Count the append-only journal under the same lock, including errors
            # before a service restart. Successful requests never reset the count.
            with ledger.open("a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                if self._fatal_error_file.exists():
                    return
                stream.seek(0)
                count = sum(1 for line in stream if line.strip()) + 1
                event = {"reason": reason, "model": self._model,
                         "pid": os.getpid(), "error_count": count,
                         "error_limit": 20, "insufficient_balance": insufficient_balance}
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if insufficient_balance or count >= 20:
                    event["stop_reason"] = (
                        "insufficient_balance" if insufficient_balance else "image_error_limit"
                    )
                    try:
                        with self._fatal_error_file.open("x") as marker:
                            json.dump(event, marker)
                    except FileExistsError:
                        pass

    def generate(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        *,
        instruction: str,
        samples: int,
    ) -> np.ndarray:
        """Generate terminal front views using wrist only as an input reference."""
        if not 1 <= int(samples) <= 10:
            raise ValueError("GPT Image Images API samples must be between 1 and 10")
        if not str(instruction).strip():
            raise ValueError("instruction must not be empty")
        front = _rgb_array(front, name="front").copy()
        for attempt in range(1, 6):
            if self._fatal_error_file is not None and self._fatal_error_file.exists():
                raise RuntimeError("Image generation disabled: batch stop already recorded")
            try:
                return self._generate_once(front, wrist, instruction=instruction, samples=samples)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                message = str(exc).lower()
                insufficient_balance = any(term in message for term in (
                    "insufficient_user_quota", "insufficient_quota", "insufficient balance",
                    "insufficient_balance", "insufficient funds", "余额不足", "额度不足", "欠费",
                ))
                reason = (f"image_api_http_{status}" if isinstance(status, int)
                          else f"image_generation_{type(exc).__name__}")
                print(f"image generation attempt={attempt}/5 error={reason} "
                      f"retry={not insufficient_balance and attempt < 5}", flush=True)
                if insufficient_balance or attempt == 5:
                    self._record_error(reason, insufficient_balance=insufficient_balance)
                    raise
        raise AssertionError("unreachable")

    def _generate_once(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        *,
        instruction: str,
        samples: int,
    ) -> np.ndarray:
        """Generate ``samples`` terminal front images with both camera inputs."""
        sample_count = int(samples)
        if not 1 <= sample_count <= 10:
            raise ValueError("GPT Image Images API samples must be between 1 and 10")
        action = str(instruction).strip()
        if not action:
            raise ValueError("instruction must not be empty")

        original = _rgb_array(front, name="front")
        wrist_image = _rgb_array(wrist, name="wrist")
        original_height, original_width = original.shape[:2]

        front_upload = _png_upload(original, "front.png")
        if self._reference_uploads:
            reference_context = (
                "Image 1 is the current front/agentview observation. Image 2 is "
                "the current eye-in-hand wrist observation, used only as reference "
                "for gripper contact and held-object geometry. "
                "Images 3 and 4 are clean identity references of the same Franka "
                "Panda arm and two-finger gripper shown in Images 1 and 2. Use them only "
                "to preserve the robot's identity and distinguish robot pixels from "
                "scene objects. Do not copy their pose, viewpoint, background, or "
                "layout. Never use Images 3 or 4 for spatial directions or target "
                "locations; use Image 1 for layout and Image 2 for close-range contact."
            )
        else:
            reference_context = "Image 1 is front/agentview and Image 2 is the current eye-in-hand wrist camera; use Image 1 for layout and Image 2 for gripper contact and held-object geometry."
        prompt = _TERMINAL_STATE_PROMPT.format(
            instruction=action,
            reference_context=reference_context + " Generate only the terminal fixed front/agentview camera view.",
        )
        wrist_upload = _png_upload(wrist_image, "wrist.png")
        image_inputs = [front_upload, wrist_upload] + list(self._reference_uploads)

        # GPT Image 2 always uses high-fidelity image inputs.  In particular,
        # do not send input_fidelity: the API rejects that parameter for this
        # model.  GPT Image models return base64 image data by default.
        self._last_request_id = None
        self._last_usage = None
        if self._fatal_error_file is not None and self._fatal_error_file.exists():
            raise RuntimeError("Image generation disabled: fatal error already recorded")
        try:
            response = self._client.images.edit(
                model=self._model,
                image=image_inputs,
                prompt=prompt,
                n=sample_count,
                quality=self._quality,
                size=self._size,
                output_format="png",
                background="opaque",
                timeout=self._request_timeout_s,
            )
        except Exception as exc:
            # OpenAI SDK exceptions expose a request ID when the server sent one.
            # Preserve it for support/debugging without retaining stale usage.
            self._last_request_id = getattr(exc, "request_id", None)
            raise
        self._last_usage = _plain_usage(response)
        self._last_request_id = getattr(response, "_request_id", None)

        data = list(getattr(response, "data", None) or [])
        if len(data) != sample_count:
            raise RuntimeError(
                f"GPT Image returned {len(data)} image(s); expected {sample_count}"
            )

        generated = []
        for index, item in enumerate(data):
            encoded = getattr(item, "b64_json", None)
            if not encoded:
                raise RuntimeError(
                    f"GPT Image response {index} has no b64_json payload"
                )
            try:
                decoded = base64.b64decode(encoded, validate=True)
                image = Image.open(io.BytesIO(decoded)).convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    f"could not decode GPT Image response {index} as an image"
                ) from exc
            if image.size != (original_width, original_height):
                image = image.resize(
                    (original_width, original_height), Image.Resampling.LANCZOS
                )
            generated.append(np.asarray(image, dtype=np.uint8))
        return np.stack(generated).astype(np.uint8)

    def info(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "backend": "OpenAI Images API edit",
            "quality": self._quality,
            "size": self._size,
            "views": "front terminal-state images; wrist input reference only",
            "input_views": ["front", "wrist"],
            "generated_views": ["front"],
            "generated_images_per_sample": 1,
            "output_frames_per_sample": 2,
            "seed_supported": False,
            "generation_max_attempts": 5,
            "last_request_id": self._last_request_id,
            "last_usage": self._last_usage,
            "robot_reference_images": [item[0] for item in self._reference_uploads],
        }


class GPTImageWorldModelFacade(RpcFacade):
    """Expose before/after image pairs through the existing rollout contract."""

    def __init__(self, backend: GPTImageBackend) -> None:
        super().__init__()
        self._backend = backend

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "rollout":
            return self.rollout(*args, **kwargs)
        if method == "model_info":
            return self._backend.info()
        raise ValueError(f"unknown RPC method: {method!r}")

    def rollout(
        self,
        *,
        front: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        horizon: int = 1,
        samples: int = 1,
        stochastic: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        # horizon/stochastic/seed remain in the signature for RPC compatibility.
        # The Images API returns a terminal state and does not expose a seed.
        del horizon, stochastic, seed
        front_images = self._backend.generate(
            front, wrist, instruction=instruction, samples=samples
        )
        initial_front = _rgb_array(front, name="front")
        front_batch = np.repeat(initial_front[None, ...], len(front_images), axis=0)
        return {
            "front": np.stack((front_batch, front_images), axis=1),
            "metadata": {
                "representation": "before_and_terminal_state_images",
                "input_views": ["front", "wrist"],
                "generated_views": ["front"],
                "usage": self._backend.info()["last_usage"],
            },
        }


def _build_openai_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required: pip install -U openai"
        ) from exc

    api_key = os.environ.get("GPT_IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GPT_IMAGE_API_KEY or OPENAI_API_KEY must be set")
    options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.request_timeout_s,
        "max_retries": args.max_retries,
    }
    if args.base_url:
        options["base_url"] = args.base_url
    return OpenAI(**options)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve GPT Image 2 terminal-state predictions over parallelworld RPC."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default=DEFAULT_QUALITY,
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help="GPT Image 2 output size, e.g. 1024x1024 or auto.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--fatal-error-file", default=None,
                        help="Write a stop marker on structurally invalid images or permanent API errors.")
    parser.add_argument(
        "--robot-reference",
        action="append",
        default=None,
        help=(
            "Robot identity reference image. Repeat for multiple images; defaults "
            "to the rendered Panda full-arm and gripper references."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="SDK retries; defaults to 0 to avoid duplicate billed generations.",
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("GPT_IMAGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        ),
        help=(
            "Optional complete API root (normally ending in /v1); defaults "
            "to GPT_IMAGE_BASE_URL, OPENAI_BASE_URL, or the official endpoint."
        ),
    )
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error(
            "GPT Image RPC must bind to loopback because paid rollout and shutdown "
            "methods are unauthenticated; use an SSH tunnel for remote access"
        )
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")

    backend = GPTImageBackend(
        _build_openai_client(args),
        model=args.model,
        quality=args.quality,
        size=args.size,
        request_timeout_s=args.request_timeout_s,
        reference_paths=args.robot_reference or DEFAULT_REFERENCE_PATHS,
        fatal_error_file=args.fatal_error_file,
    )
    facade = GPTImageWorldModelFacade(backend)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        facade._shutdown_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
