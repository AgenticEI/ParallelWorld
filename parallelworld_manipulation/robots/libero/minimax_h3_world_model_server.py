"""MiniMax-H3 I2V RPC service backed by ComfyUI's native nodes."""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from parallelworld_manipulation.utils.rpc import RpcFacade

_REPO_ROOT = Path(os.environ.get("PARALLELWORLD_REPO_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
DEFAULT_COMFY_ROOT = os.environ.get(
    "PW_COMFY_ROOT", str(_REPO_ROOT / "third_party" / "ComfyUI-MiniMaxH3")
)
DEFAULT_COMFY_PYTHON = os.environ.get("PW_COMFY_PYTHON", sys.executable)
DEFAULT_EXTRA_MODEL_PATHS_CONFIG = str(
    Path(__file__).with_name("minimax_h3_extra_model_paths.yaml")
)
DEFAULT_DIFFUSION_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def _resolve_extra_model_paths_config(config_path: str) -> str:
    """Resolve the bundled checkpoint template without ComfyUI-specific path assumptions."""
    configured = Path(config_path).expanduser().resolve()
    if configured != Path(DEFAULT_EXTRA_MODEL_PATHS_CONFIG).resolve():
        return str(configured)
    checkpoint_root = Path(os.environ.get(
        "PW_MINIMAX_CHECKPOINTS", str(_REPO_ROOT / "checkpoints" / "minimax-h3")
    )).expanduser().resolve()
    payload = {"parallelworld_manipulation_minimax_h3": {
        "base_path": str(checkpoint_root),
        "diffusion_models": "diffusion_models",
        "text_encoders": "text_encoders",
        "vae": "vae",
        "loras": "loras",
    }}
    # JSON is a YAML subset accepted by ComfyUI's extra-model-paths parser.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="parallelworld_minimax_paths_", suffix=".yaml", delete=False
    ) as handle:
        json.dump(payload, handle)
        return handle.name


class ComfyMiniMaxH3Backend:
    """Submit minimal native MiniMax-H3 I2V graphs to a ComfyUI server."""

    def __init__(
        self,
        *,
        endpoint: str,
        comfy_root: str | Path | None,
        diffusion_model: str,
        text_encoder: str,
        video_vae: str,
        turbo_lora: str | None,
        lora_strength: float,
        width: int,
        height: int,
        frame_num: int,
        sampling_steps: int,
        timeout_s: float,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._comfy_root = Path(comfy_root).resolve() if comfy_root else None
        self._diffusion_model = diffusion_model
        self._text_encoder = text_encoder
        self._video_vae = video_vae
        self._turbo_lora = turbo_lora
        self._lora_strength = float(lora_strength)
        self._width = self._multiple_of_32(width)
        self._height = self._multiple_of_32(height)
        self._frame_num = self._aligned_frame_count(frame_num)
        self._sampling_steps = max(1, int(sampling_steps))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = max(0.1, float(poll_interval_s))
        self._validate_server()

    @staticmethod
    def _multiple_of_32(value: int) -> int:
        return max(32, round(int(value) / 32) * 32)

    @staticmethod
    def _aligned_frame_count(value: int) -> int:
        value = max(5, int(value))
        while value % 17 != 5:
            value += 1
        return value

    def _request_json(
        self,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        timeout_s: float = 30.0,
    ) -> Any:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(
            self._endpoint + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())

    def _validate_server(self) -> None:
        info = self._request_json("/object_info")
        required_nodes = {
            "LoadImage",
            "UNETLoader",
            "LoraLoaderModelOnly",
            "CLIPLoader",
            "VAELoader",
            "MiniMaxH3ImageToVideo",
            "RandomNoise",
            "BasicGuider",
            "KSamplerSelect",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "SaveImage",
        }
        missing = sorted(required_nodes.difference(info))
        if missing:
            raise RuntimeError(f"ComfyUI is missing MiniMax-H3 nodes: {missing}")

        checks = {
            ("UNETLoader", "unet_name"): self._diffusion_model,
            ("CLIPLoader", "clip_name"): self._text_encoder,
            ("VAELoader", "vae_name"): self._video_vae,
        }
        if self._turbo_lora:
            checks[("LoraLoaderModelOnly", "lora_name")] = self._turbo_lora
        for (node, field), filename in checks.items():
            choices = info[node]["input"]["required"][field][0]
            if filename not in choices:
                raise FileNotFoundError(
                    f"{filename!r} is not visible to ComfyUI node {node}.{field}"
                )

    def _upload_image(self, image: np.ndarray) -> dict[str, str]:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").save(
            buffer, format="PNG"
        )
        filename = f"parallelworld_manipulation_h3_{uuid.uuid4().hex}.png"
        boundary = f"----parallelworldMiniMaxH3{uuid.uuid4().hex}"
        fields = [
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode(),
            buffer.getvalue(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="type"\r\n\r\ninput',
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        return self._request_json(
            "/upload/image",
            data=b"".join(fields),
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def _workflow(
        self,
        *,
        image_name: str,
        instruction: str,
        frame_num: int,
        seed: int,
        output_prefix: str,
    ) -> dict[str, Any]:
        model_source: list[Any] = ["2", 0]
        graph: dict[str, Any] = {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": self._diffusion_model,
                    "weight_dtype": "default",
                },
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": self._text_encoder,
                    "type": "minimax",
                    "device": "cpu",
                },
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": self._video_vae},
            },
            "6": {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "prompt": str(instruction),
                    "width": self._width,
                    "height": self._height,
                    "length": frame_num,
                    "first_frame": ["1", 0],
                },
            },
            "7": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": int(seed) % (2**64)},
            },
            "9": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "res_multistep"},
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["5", 0]},
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["12", 0],
                    "filename_prefix": output_prefix,
                },
            },
        }
        if self._turbo_lora:
            graph["3"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["2", 0],
                    "lora_name": self._turbo_lora,
                    "strength_model": self._lora_strength,
                },
            }
            model_source = ["3", 0]
        graph.update(
            {
                "8": {
                    "class_type": "BasicGuider",
                    "inputs": {"model": model_source, "conditioning": ["6", 0]},
                },
                "10": {
                    "class_type": "BasicScheduler",
                    "inputs": {
                        "model": model_source,
                        "scheduler": "simple",
                        "steps": self._sampling_steps,
                        "denoise": 1.0,
                    },
                },
                "11": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["7", 0],
                        "guider": ["8", 0],
                        "sampler": ["9", 0],
                        "sigmas": ["10", 0],
                        "latent_image": ["6", 1],
                    },
                },
            }
        )
        return graph

    def _wait_for_images(self, prompt_id: str) -> list[dict[str, str]]:
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            history = self._request_json(f"/history/{prompt_id}")
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") != "success":
                    raise RuntimeError(
                        "MiniMax-H3 ComfyUI execution failed: "
                        + json.dumps(status.get("messages", []), default=str)[-4000:]
                    )
                images = item.get("outputs", {}).get("13", {}).get("images", [])
                if not images:
                    raise RuntimeError("MiniMax-H3 workflow returned no decoded frames")
                return images
            time.sleep(self._poll_interval_s)
        raise TimeoutError(
            f"MiniMax-H3 generation did not finish within {self._timeout_s:.0f}s"
        )

    def _read_image(self, descriptor: dict[str, str]) -> np.ndarray:
        query = urllib.parse.urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
        )
        with urllib.request.urlopen(
            f"{self._endpoint}/view?{query}", timeout=30.0
        ) as response:
            return np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"))

    def _local_artifact_path(self, descriptor: dict[str, str]) -> Path | None:
        if self._comfy_root is None:
            return None
        artifact_type = descriptor.get("type", "output")
        if artifact_type not in {"input", "output", "temp"}:
            return None
        root = (self._comfy_root / artifact_type).resolve()
        path = (
            root
            / descriptor.get("subfolder", "")
            / descriptor["filename"]
        ).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return None
        return path

    def _remove_local_artifact(self, descriptor: dict[str, str]) -> None:
        path = self._local_artifact_path(descriptor)
        if path is not None:
            path.unlink(missing_ok=True)

    def generate(
        self,
        image: np.ndarray,
        *,
        instruction: str,
        horizon: int,
        seed: int,
    ) -> np.ndarray:
        uploaded = self._upload_image(image)
        frame_num = self._aligned_frame_count(max(self._frame_num, int(horizon)))
        run_id = uuid.uuid4().hex
        graph = self._workflow(
            image_name=uploaded["name"],
            instruction=instruction,
            frame_num=frame_num,
            seed=seed,
            output_prefix=f"parallelworld_manipulation_minimax_h3/{run_id}/frame",
        )
        output_descriptors: list[dict[str, str]] = []
        try:
            reply = self._request_json(
                "/prompt",
                data=json.dumps(
                    {"prompt": graph, "client_id": f"parallelworld_manipulation-{run_id}"}
                ).encode(),
                content_type="application/json",
            )
            node_errors = reply.get("node_errors") or {}
            if node_errors:
                raise RuntimeError(
                    "MiniMax-H3 workflow validation failed: "
                    + json.dumps(node_errors, default=str)
                )
            output_descriptors = self._wait_for_images(reply["prompt_id"])
            return np.stack(
                [self._read_image(item) for item in output_descriptors]
            ).astype(np.uint8)
        finally:
            self._remove_local_artifact(
                {
                    "filename": uploaded["name"],
                    "subfolder": uploaded.get("subfolder", ""),
                    "type": uploaded.get("type", "input"),
                }
            )
            for descriptor in output_descriptors:
                self._remove_local_artifact(descriptor)

    def info(self) -> dict[str, Any]:
        return {
            "model": "MiniMax-H3-FL2VA-int8-convrot",
            "comfy_endpoint": self._endpoint,
            "diffusion_model": self._diffusion_model,
            "text_encoder": self._text_encoder,
            "video_vae": self._video_vae,
            "turbo_lora": self._turbo_lora,
            "sampling_steps": self._sampling_steps,
            "frame_num": self._frame_num,
            "width": self._width,
            "height": self._height,
            "views": "front-only I2V rollout",
        }


class MiniMaxH3WorldModelFacade(RpcFacade):
    """Keep the existing ParallelWorld rollout RPC contract over MiniMax-H3."""

    def __init__(self, backend: ComfyMiniMaxH3Backend) -> None:
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
        horizon: int = 22,
        samples: int = 1,
        stochastic: bool = False,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        del stochastic, wrist
        fronts = []
        for sample in range(max(1, int(samples))):
            sample_seed = int(seed) + sample * 1009
            fronts.append(
                self._backend.generate(
                    front,
                    instruction=str(instruction),
                    horizon=horizon,
                    seed=sample_seed,
                )
            )
        return {"front": np.asarray(fronts, dtype=np.uint8)}


def _wait_for_comfy(endpoint: str, process: subprocess.Popen[Any] | None) -> None:
    deadline = time.monotonic() + 300.0
    url = endpoint.rstrip("/") + "/system_stats"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited with code {process.returncode} before becoming ready"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"ComfyUI did not become ready: {last_error}")


def _spawn_comfy(args: argparse.Namespace) -> subprocess.Popen[Any]:
    comfy_root = Path(args.comfy_root).resolve()
    user_directory = Path(args.comfy_user_directory).resolve()
    user_directory.mkdir(parents=True, exist_ok=True)
    command = [
        args.comfy_python,
        str(comfy_root / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(args.comfy_port),
        "--disable-auto-launch",
        "--extra-model-paths-config",
        _resolve_extra_model_paths_config(args.extra_model_paths_config),
        "--lowvram",
        "--reserve-vram",
        str(args.comfy_reserve_vram),
        "--user-directory",
        str(user_directory),
        "--database-url",
        f"sqlite:///{user_directory / 'comfyui.db'}",
    ]
    environment = os.environ.copy()
    if args.cuda_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    return subprocess.Popen(command, cwd=comfy_root, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-endpoint", default=None)
    parser.add_argument("--comfy-root", default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--comfy-python", default=DEFAULT_COMFY_PYTHON)
    parser.add_argument(
        "--extra-model-paths-config",
        default=DEFAULT_EXTRA_MODEL_PATHS_CONFIG,
    )
    parser.add_argument("--comfy-port", type=int, default=8189)
    parser.add_argument(
        "--comfy-user-directory",
        default=None,
        help="Per-process ComfyUI user directory (defaults to /tmp by port).",
    )
    parser.add_argument("--comfy-reserve-vram", type=float, default=2.0)
    parser.add_argument("--diffusion-model", default=DEFAULT_DIFFUSION_MODEL)
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    parser.add_argument("--turbo-lora", default=DEFAULT_TURBO_LORA)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--frame-num", type=int, default=22)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--generation-timeout-s", type=float, default=900.0)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.comfy_user_directory is None:
        args.comfy_user_directory = f"/tmp/parallelworld_manipulation-minimax-h3-{args.comfy_port}"

    comfy_process = None
    endpoint = args.comfy_endpoint
    if endpoint is None:
        endpoint = f"http://127.0.0.1:{args.comfy_port}"
        comfy_process = _spawn_comfy(args)
    _wait_for_comfy(endpoint, comfy_process)

    backend = ComfyMiniMaxH3Backend(
        endpoint=endpoint,
        comfy_root=args.comfy_root,
        diffusion_model=args.diffusion_model,
        text_encoder=args.text_encoder,
        video_vae=args.video_vae,
        turbo_lora=args.turbo_lora or None,
        lora_strength=args.lora_strength,
        width=args.width,
        height=args.height,
        frame_num=args.frame_num,
        sampling_steps=args.sampling_steps,
        timeout_s=args.generation_timeout_s,
    )
    facade = MiniMaxH3WorldModelFacade(backend)

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
        if comfy_process is not None and comfy_process.poll() is None:
            comfy_process.terminate()
            try:
                comfy_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                comfy_process.kill()
                comfy_process.wait()


if __name__ == "__main__":
    main()
