from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robots.libero import minimax_h3_world_model_server as h3_server
from robots.libero.minimax_h3_world_model_server import (
    ComfyMiniMaxH3Backend,
    MiniMaxH3WorldModelFacade,
)


def backend_without_server(tmp_path: Path) -> ComfyMiniMaxH3Backend:
    backend = object.__new__(ComfyMiniMaxH3Backend)
    backend._comfy_root = tmp_path
    backend._diffusion_model = "h3-int8.safetensors"
    backend._text_encoder = "qwen-nvfp4.safetensors"
    backend._video_vae = "h3-video-vae.safetensors"
    backend._turbo_lora = "h3-turbo.safetensors"
    backend._lora_strength = 1.0
    backend._width = 512
    backend._height = 512
    backend._sampling_steps = 8
    return backend


def test_h3_frame_count_uses_native_grid():
    align = ComfyMiniMaxH3Backend._aligned_frame_count
    assert align(1) == 5
    assert align(5) == 5
    assert align(6) == 22
    assert align(22) == 22
    assert align(23) == 39


def test_minimal_workflow_uses_int8_h3_and_turbo_lora(tmp_path):
    backend = backend_without_server(tmp_path)
    graph = backend._workflow(
        image_name="input.png",
        instruction="move the robot arm",
        frame_num=22,
        seed=7,
        output_prefix="parallelworld_manipulation/frame",
    )

    assert graph["2"]["inputs"]["unet_name"] == "h3-int8.safetensors"
    assert graph["3"]["inputs"]["lora_name"] == "h3-turbo.safetensors"
    assert graph["4"]["inputs"]["type"] == "minimax"
    assert graph["6"]["inputs"]["first_frame"] == ["1", 0]
    assert graph["6"]["inputs"]["length"] == 22
    assert graph["8"]["inputs"]["model"] == ["3", 0]
    assert graph["10"]["inputs"]["steps"] == 8


def test_rollout_preserves_front_only_parallelworld_rpc_shape_and_seed_offsets():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def generate(self, image, **kwargs):
            self.calls.append((image.copy(), kwargs))
            return np.repeat(image[None], 5, axis=0)

    backend = FakeBackend()
    facade = MiniMaxH3WorldModelFacade(backend)
    front = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)

    result = facade.rollout(
        front=front,
        wrist=wrist,
        instruction="release",
        horizon=5,
        samples=2,
        stochastic=True,
        seed=10,
    )

    assert result["front"].shape == (2, 5, 8, 8, 3)
    assert result["front"].dtype == np.uint8
    assert set(result) == {"front"}
    assert [call[1]["seed"] for call in backend.calls] == [10, 1019]
    assert len(backend.calls) == 2
    for image, arguments in backend.calls:
        np.testing.assert_array_equal(image, front)
        assert arguments["instruction"] == "release"
        assert arguments["horizon"] == 5
    np.testing.assert_array_equal(result["front"][0, 0], front)
    np.testing.assert_array_equal(result["front"][1, 0], front)


def test_spawned_comfy_uses_an_isolated_database(tmp_path, monkeypatch):
    comfy_root = tmp_path / "comfy"
    user_directory = tmp_path / "user"
    comfy_root.mkdir()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(h3_server.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        comfy_root=str(comfy_root),
        comfy_user_directory=str(user_directory),
        comfy_python="python",
        extra_model_paths_config=str(tmp_path / "extra_model_paths.yaml"),
        comfy_port=18212,
        comfy_reserve_vram=2.0,
        cuda_device=2,
    )

    h3_server._spawn_comfy(args)

    assert user_directory.is_dir()
    config_index = captured["command"].index("--extra-model-paths-config")
    assert captured["command"][config_index + 1] == str(
        (tmp_path / "extra_model_paths.yaml").resolve()
    )
    assert captured["command"][-2:] == [
        "--database-url",
        f"sqlite:///{user_directory / 'comfyui.db'}",
    ]
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
