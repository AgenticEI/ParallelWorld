"""RPC client for text-conditioned video world models."""

from __future__ import annotations

from typing import Any

import numpy as np

from parallelworld_manipulation.utils.rpc import RpcClient


class VideoWorldModelClient:
    """Backend-neutral client for the ParallelWorld rollout RPC contract."""

    def __init__(self, client: RpcClient, *, timeout_s: float = 300.0) -> None:
        self._client = client
        self._timeout_s = float(timeout_s)

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self._client.call("healthz", timeout_s=timeout_s)

    def model_info(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Return backend identity and runtime configuration."""
        return self._client.call("model_info", timeout_s=timeout_s)

    def rollout(
        self,
        *,
        front: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        horizon: int,
        samples: int,
        stochastic: bool,
        seed: int,
    ) -> dict[str, np.ndarray]:
        result = self._client.call(
            "rollout",
            kwargs={
                "front": np.asarray(front, dtype=np.uint8),
                "wrist": np.asarray(wrist, dtype=np.uint8),
                "instruction": str(instruction),
                "horizon": int(horizon),
                "samples": int(samples),
                "stochastic": bool(stochastic),
                "seed": int(seed),
            },
            timeout_s=self._timeout_s,
        )
        rollout = {"front": np.asarray(result["front"], dtype=np.uint8)}
        # Independent front-only servers intentionally omit wrist.  Joint
        # multi-view backends may still return it for their internal lineage.
        if result.get("wrist") is not None:
            rollout["wrist"] = np.asarray(result["wrist"], dtype=np.uint8)
        return rollout


# Backward compatibility for existing MiniMax-H3 launchers and imports.
MiniMaxH3WorldModelClient = VideoWorldModelClient
