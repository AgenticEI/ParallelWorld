"""ParallelWorld RPC adapter for a full-precision MiniMax-H3 SGLang server."""
from __future__ import annotations

import argparse
import json
import signal
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image

from parallelworld_manipulation.utils.rpc import RpcFacade


class SGLangMiniMaxH3Backend:
    """Turn an RGB frame into an H3 FL2VA request and decode its MP4 result."""

    def __init__(
        self,
        *,
        endpoint: str,
        short_edge: int = 768,
        duration_seconds: float = 4.0,
        sampling_steps: int = 50,
        request_timeout_s: float = 60.0,
        generation_timeout_s: float = 7200.0,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._short_edge = int(short_edge)
        self._duration_seconds = float(duration_seconds)
        self._sampling_steps = max(1, int(sampling_steps))
        self._request_timeout_s = float(request_timeout_s)
        self._generation_timeout_s = float(generation_timeout_s)
        self._poll_interval_s = max(0.1, float(poll_interval_s))
        if self._short_edge <= 0:
            raise ValueError("short_edge must be positive")
        # MiniMax-H3's native serving contract accepts 4--15 seconds.
        if not 4.0 <= self._duration_seconds <= 15.0:
            raise ValueError("duration_seconds must be in MiniMax-H3's [4, 15] range")
        self.health()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._request_timeout_s
            ) as response:
                return response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"MiniMax-H3 HTTP {exc.code} for {path}: {detail}"
            ) from exc

    def health(self) -> dict[str, Any]:
        raw, _ = self._request("GET", "/health")
        text = raw.decode("utf-8", errors="replace")
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"body": text}
        if isinstance(result, dict) and result.get("status") not in {None, "ok"}:
            raise RuntimeError(f"MiniMax-H3 upstream is unhealthy: {result}")
        return result

    def _submit(self, image_path: Path, *, instruction: str, seed: int) -> str:
        payload = {
            "task": "fl2va",
            "prompt": str(instruction),
            "conditions": [
                {
                    "type": "image",
                    "uri": image_path.resolve().as_uri(),
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
            "target": {
                "short_edge": self._short_edge,
                "aspect_ratio": "auto",
                "duration_seconds": self._duration_seconds,
            },
            "num_inference_steps": self._sampling_steps,
            "seed": int(seed),
        }
        raw, _ = self._request("POST", "/v1/videos", payload)
        response = json.loads(raw)
        job_id = response.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"MiniMax-H3 submit response lacks id: {response}")
        return job_id

    def _wait_for_video(self, job_id: str, output_path: Path) -> dict[str, Any]:
        started = time.monotonic()
        status_history: list[dict[str, Any]] = []
        while True:
            raw, _ = self._request("GET", f"/v1/videos/{job_id}")
            status = json.loads(raw)
            state = status.get("status")
            status_history.append(
                {"elapsed_s": round(time.monotonic() - started, 3), "status": state}
            )
            if state == "completed":
                break
            if state in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"MiniMax-H3 job {job_id} ended as {state}: {status}"
                )
            if time.monotonic() - started > self._generation_timeout_s:
                raise TimeoutError(
                    f"MiniMax-H3 job {job_id} exceeded "
                    f"{self._generation_timeout_s:.0f}s"
                )
            time.sleep(self._poll_interval_s)

        video, headers = self._request("GET", f"/v1/videos/{job_id}/content")
        if len(video) < 1024:
            raise RuntimeError(
                f"MiniMax-H3 job {job_id} returned only {len(video)} bytes"
            )
        output_path.write_bytes(video)
        return {
            "job_id": job_id,
            "elapsed_s": round(time.monotonic() - started, 3),
            "bytes": len(video),
            "content_type": headers.get("Content-Type"),
            "status_history": status_history,
        }

    @staticmethod
    def _decode_and_sample(video_path: Path, horizon: int) -> np.ndarray:
        frames = np.asarray(iio.imread(video_path, plugin="FFMPEG"), dtype=np.uint8)
        if frames.ndim != 4 or frames.shape[-1] < 3 or frames.shape[0] == 0:
            raise RuntimeError(f"unexpected MiniMax-H3 video shape: {frames.shape}")
        frames = frames[..., :3]
        indices = np.rint(
            np.linspace(0, frames.shape[0] - 1, max(1, int(horizon)))
        ).astype(np.int64)
        return np.ascontiguousarray(frames[indices], dtype=np.uint8)

    def generate(
        self,
        image: np.ndarray,
        *,
        instruction: str,
        horizon: int,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="parallelworld_manipulation_h3_sglang_") as tmp:
            work_dir = Path(tmp)
            image_path = work_dir / "front.png"
            video_path = work_dir / "future.mp4"
            Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").save(
                image_path
            )
            job_id = self._submit(
                image_path, instruction=str(instruction), seed=int(seed)
            )
            metadata = self._wait_for_video(job_id, video_path)
            frames = self._decode_and_sample(video_path, horizon)
            metadata.update(
                {
                    "returned_frames": int(frames.shape[0]),
                    "height": int(frames.shape[1]),
                    "width": int(frames.shape[2]),
                }
            )
            return frames, metadata

    def info(self) -> dict[str, Any]:
        return {
            "model": "MiniMax-H3-FL2VA-BF16",
            "backend": "SGLang",
            "sglang_endpoint": self._endpoint,
            "short_edge": self._short_edge,
            "duration_seconds": self._duration_seconds,
            "native_fps": 24,
            "sampling_steps": self._sampling_steps,
            "views": "front-only I2V rollout",
            "upstream_health": self.health(),
        }


class MiniMaxH3SGLangWorldModelFacade(RpcFacade):
    """Expose full-precision H3 through ParallelWorld's existing RPC contract."""

    def __init__(self, backend: SGLangMiniMaxH3Backend) -> None:
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
        horizon: int = 49,
        samples: int = 1,
        stochastic: bool = False,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        del stochastic, wrist
        fronts = []
        for sample in range(max(1, int(samples))):
            frames, _metadata = self._backend.generate(
                front,
                instruction=str(instruction),
                horizon=max(1, int(horizon)),
                seed=int(seed) + sample * 1009,
            )
            fronts.append(frames)
        return {"front": np.asarray(fronts, dtype=np.uint8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-endpoint", required=True)
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--generation-timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    backend = SGLangMiniMaxH3Backend(
        endpoint=args.sglang_endpoint,
        short_edge=args.short_edge,
        duration_seconds=args.duration_seconds,
        sampling_steps=args.sampling_steps,
        request_timeout_s=args.request_timeout_s,
        generation_timeout_s=args.generation_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    facade = MiniMaxH3SGLangWorldModelFacade(backend)

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
