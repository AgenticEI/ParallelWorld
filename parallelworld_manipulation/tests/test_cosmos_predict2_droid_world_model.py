from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robots.libero.cosmos_predict2_droid_world_model_server import (
    DEFAULT_SAMPLING_STEPS,
    CosmosDroidBackend,
    _make_droid_mosaic,
    _split_droid_video,
    configure_droid_tokenizer,
    keep_text_encoder_on_cpu,
    required_checkpoint_paths,
    validate_runtime_paths,
)


def test_droid_mosaic_layout_uses_synchronised_front_and_wrist_views():
    front = np.full((8, 8, 3), 17, dtype=np.uint8)
    wrist = np.full((6, 6, 3), 29, dtype=np.uint8)

    mosaic = _make_droid_mosaic(front, wrist)

    assert mosaic.shape == (360, 640, 3)
    assert np.all(mosaic[:180, 70:250] == 17)
    assert np.all(mosaic[:180, :70] == 0)
    assert np.all(mosaic[:180, 390:570] == 17)
    assert np.all(mosaic[180:, 70:250] == 29)
    assert np.all(mosaic[180:, :70] == 0)
    assert np.all(mosaic[180:, 320:] == 0)


def test_generated_mosaic_is_split_back_into_parallelworld_views():
    video = torch.zeros((1, 3, 5, 360, 640), dtype=torch.float32)
    video[:, :, :, :180, :320] = 0.25
    video[:, :, :, 180:, :320] = 0.75

    front, wrist = _split_droid_video(video, horizon=3)

    assert front.shape == (3, 180, 320, 3)
    assert wrist.shape == (3, 180, 320, 3)
    assert np.all(front == 64)
    assert np.all(wrist == 191)


def test_generated_views_remove_letterbox_and_restore_source_resolution():
    video = torch.zeros((1, 3, 5, 360, 640), dtype=torch.float32)
    video[:, :, :, :180, 70:250] = 0.25
    video[:, :, :, 180:, 70:250] = 0.75

    front, wrist = _split_droid_video(
        video,
        horizon=3,
        front_shape=(256, 256),
        wrist_shape=(256, 256),
    )

    assert front.shape == (3, 256, 256, 3)
    assert wrist.shape == (3, 256, 256, 3)
    assert np.all(front == 64)
    assert np.all(wrist == 191)


def test_default_sampling_uses_twenty_diffusion_steps():
    assert DEFAULT_SAMPLING_STEPS == 20


def test_latent_frame_count_aligns_to_context_parallelworld_size():
    backend = object.__new__(CosmosDroidBackend)
    backend._world_size = 2

    assert backend._latent_frames(1) == 2
    assert backend._latent_frames(22) == 8
    assert backend._latent_frames(93) == 24


def test_droid_tokenizer_overrides_base_video2world_path(tmp_path: Path):
    config = SimpleNamespace(tokenizer=SimpleNamespace(vae_pth="base/tokenizer.pth"))

    selected = configure_droid_tokenizer(config, tmp_path)

    assert selected == required_checkpoint_paths(tmp_path)["tokenizer"]
    assert config.tokenizer.vae_pth == str(selected)


def test_text_encoder_offload_keeps_model_inputs_on_cpu():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode_prompts(self, prompts, *, max_length, return_mask):
            self.calls.append((prompts, max_length, return_mask))
            return torch.ones((1, 2, 3))

    pipe = SimpleNamespace(text_encoder=Encoder())
    keep_text_encoder_on_cpu(pipe)

    encoded = pipe.encode_prompt("move", max_length=12, return_mask=False)

    assert encoded.device.type == "cpu"
    assert pipe.text_encoder.calls == [("move", 12, False)]


def test_runtime_validation_reports_all_missing_inputs(tmp_path: Path):
    cosmos_root = tmp_path / "cosmos"
    parallelworld_root = tmp_path / "parallelworld_manipulation"
    checkpoints = tmp_path / "checkpoints"

    with pytest.raises(FileNotFoundError) as error:
        validate_runtime_paths(
            cosmos_root=cosmos_root,
            parallelworld_root=parallelworld_root,
            checkpoint_root=checkpoints,
        )

    message = str(error.value)
    assert "Cosmos-Predict2 DROID runtime is incomplete" in message
    assert "model-480p-16fps.pt" in message
    assert "tokenizer.pth" in message
    assert "pytorch_model.bin" in message


def test_runtime_validation_accepts_complete_layout(tmp_path: Path):
    cosmos_root = tmp_path / "cosmos"
    parallelworld_root = tmp_path / "parallelworld_manipulation"
    (cosmos_root / "cosmos_predict2").mkdir(parents=True)
    (cosmos_root / "cosmos_predict2/__init__.py").touch()
    (parallelworld_root / "parallelworld_manipulation").mkdir(parents=True)
    (parallelworld_root / "parallelworld_manipulation/__init__.py").touch()
    for path in required_checkpoint_paths(tmp_path / "checkpoints").values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    checked = validate_runtime_paths(
        cosmos_root=cosmos_root,
        parallelworld_root=parallelworld_root,
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert set(checked) == {
        "cosmos checkout",
        "parallelworld checkout",
        "dit",
        "tokenizer",
        "t5_config",
        "t5_weights",
    }
