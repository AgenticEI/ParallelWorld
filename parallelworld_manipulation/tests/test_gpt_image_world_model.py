# ruff: noqa: CPY001

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def test_image_error_limit_shared_between_backends(tmp_path):
    marker = tmp_path / "IMAGE_GENERATION_FATAL.json"
    class FailingImages:
        def edit(self, **kwargs):
            exc = RuntimeError("provider custom error")
            exc.status_code = 488
            raise exc
    backends = [GPTImageBackend(SimpleNamespace(images=FailingImages()), fatal_error_file=marker) for _ in range(8)]
    front = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(19):
        with pytest.raises(RuntimeError, match="provider custom error"):
            backends[i % 8].generate(front, front, instruction="hold still", samples=1)
        assert not marker.exists()
    # Success between errors must not reset the batch counter.
    good = GPTImageBackend(SimpleNamespace(images=FakeImages([encoded_png((0, 0, 0), (16, 16))])), fatal_error_file=marker)
    good.generate(front, front, instruction="hold still", samples=1)
    with pytest.raises(RuntimeError, match="provider custom error"):
        backends[0].generate(front, front, instruction="hold still", samples=1)
    assert json.loads(marker.read_text())["error_count"] == 20
    assert json.loads(marker.read_text())["stop_reason"] == "image_error_limit"
    with pytest.raises(RuntimeError, match="disabled"):
        good.generate(front, front, instruction="hold still", samples=1)


def test_image_quota_stops_immediately(tmp_path):
    marker = tmp_path / "IMAGE_GENERATION_FATAL.json"
    class QuotaImages:
        calls = 0
        def edit(self, **kwargs):
            self.calls += 1
            exc = RuntimeError("insufficient_user_quota: 用户额度不足")
            exc.status_code = 403
            raise exc
    backend = GPTImageBackend(SimpleNamespace(images=QuotaImages()), fatal_error_file=marker)
    front = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        backend.generate(front, front, instruction="hold still", samples=1)
    event = json.loads(marker.read_text())
    assert event["error_count"] == 1
    assert event["stop_reason"] == "insufficient_balance"
    assert backend._client.images.calls == 1


def test_retry_identical_request_then_success_does_not_count_error(tmp_path):
    marker = tmp_path / "IMAGE_GENERATION_FATAL.json"
    class FlakyImages(FakeImages):
        def edit(self, **kwargs):
            if len(self.calls) < 4:
                self.calls.append(kwargs)
                raise RuntimeError("temporary failure")
            return super().edit(**kwargs)
    images = FlakyImages([encoded_png((1, 2, 3), (16, 16))])
    backend = GPTImageBackend(SimpleNamespace(images=images), fatal_error_file=marker)
    front = np.zeros((16, 16, 3), dtype=np.uint8)
    result = backend.generate(front, front, instruction="hold the bowl", samples=1)
    assert result.shape == (1, 16, 16, 3)
    assert len(images.calls) == 5
    assert all(call == images.calls[0] for call in images.calls)
    assert not (tmp_path / "IMAGE_GENERATION_ERRORS.jsonl").exists()


def test_five_failed_requests_count_as_one_error(tmp_path):
    class FailingImages:
        calls = 0
        def edit(self, **kwargs):
            self.calls += 1
            raise TimeoutError("request timeout")
    images = FailingImages()
    backend = GPTImageBackend(SimpleNamespace(images=images), fatal_error_file=tmp_path / "IMAGE_GENERATION_FATAL.json")
    front = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(TimeoutError):
        backend.generate(front, front, instruction="hold still", samples=1)
    assert images.calls == 5
    lines = (tmp_path / "IMAGE_GENERATION_ERRORS.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["error_count"] == 1


def test_invalid_image_counts_once_without_stopping(tmp_path):
    marker = tmp_path / "IMAGE_GENERATION_FATAL.json"
    backend = GPTImageBackend(SimpleNamespace(images=FakeImages(["bad"])), fatal_error_file=marker)
    front = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="could not decode"):
        backend.generate(front, front, instruction="hold still", samples=1)
    assert not marker.exists()
    assert len((tmp_path / "IMAGE_GENERATION_ERRORS.jsonl").read_text().splitlines()) == 1


def test_concurrent_error_counter(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    marker = tmp_path / "IMAGE_GENERATION_FATAL.json"
    backends = [GPTImageBackend(SimpleNamespace(images=None), fatal_error_file=marker) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: backends[i % 8]._record_error("image_api_timeout"), range(20)))
    assert json.loads(marker.read_text())["error_count"] == 20
    events = [json.loads(s) for s in (tmp_path / "IMAGE_GENERATION_ERRORS.jsonl").read_text().splitlines()]
    assert [e["error_count"] for e in events] == list(range(1, 21))

from robots.libero.gpt_image_world_model_server import (
    GPTImageBackend,
    GPTImageWorldModelFacade,
    validate_gpt_image_2_size,
)


def encoded_png(color: tuple[int, int, int], size: tuple[int, int]) -> str:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


class FakeImages:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "input_tokens": 123,
                "output_tokens": 456,
            }
        )
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=item) for item in self.payloads],
            usage=usage,
            _request_id="req_test_123",
        )


def test_gpt_image_2_size_validation():
    assert validate_gpt_image_2_size("1024x1024") == "1024x1024"
    assert validate_gpt_image_2_size("1536X1024") == "1536x1024"
    assert validate_gpt_image_2_size("auto") == "auto"
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_gpt_image_2_size("1025x1024")
    with pytest.raises(ValueError, match="total pixels"):
        validate_gpt_image_2_size("256x256")
    with pytest.raises(ValueError, match="3:1"):
        validate_gpt_image_2_size("3072x768")


def test_rollout_calls_edit_and_preserves_parallelworld_shape():
    images = FakeImages([encoded_png((10, 20, 30), (1024, 1024))])
    backend = GPTImageBackend(
        SimpleNamespace(images=images),
        model="gpt-image-2",
        quality="low",
        size="1024x1024",
        request_timeout_s=42,
    )
    facade = GPTImageWorldModelFacade(backend)
    front = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.ones((32, 32, 3), dtype=np.uint8)

    result = facade.rollout(
        front=front,
        wrist=wrist,
        instruction="the robot closes its gripper on the red block",
        horizon=22,
        samples=1,
        stochastic=False,
        seed=91,
    )

    assert result["front"].shape == (1, 2, 48, 64, 3)
    assert result["front"].dtype == np.uint8
    assert "wrist" not in result
    assert len(images.calls) == 1
    np.testing.assert_array_equal(result["front"][0, 0], front)
    np.testing.assert_array_equal(result["front"][0, 1, 0, 0], [10, 20, 30])
    assert result["metadata"]["representation"] == "before_and_terminal_state_images"
    assert result["metadata"]["usage"]["output_tokens"] == 456
    assert backend.info()["last_request_id"] == "req_test_123"

    call = images.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["n"] == 1
    assert call["quality"] == "low"
    assert call["size"] == "1024x1024"
    assert call["output_format"] == "png"
    assert call["timeout"] == 42
    assert call["image"][0][0] == "front.png"
    assert call["image"][0][2] == "image/png"
    assert call["image"][1][0] == "wrist.png"
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(call["image"][1][1]))), wrist
    )
    assert "input_fidelity" not in call
    assert "response_format" not in call
    assert "seed" not in call
    assert "fixed front camera" in call["prompt"]
    assert "closes its gripper" in call["prompt"]
    assert "image-left and image-right" in call["prompt"]
    assert "viewer's" in call["prompt"]
    assert "magenta marker labeled A" in call["prompt"]
    assert "target center or projected terminal" in call["prompt"]
    assert "remove the marker" in call["prompt"]


def test_multiple_samples_use_one_front_request_with_wrist_reference():
    images = FakeImages(
        [
            encoded_png((1, 2, 3), (1024, 1024)),
            encoded_png((4, 5, 6), (1024, 1024)),
        ]
    )
    backend = GPTImageBackend(SimpleNamespace(images=images))
    facade = GPTImageWorldModelFacade(backend)

    result = facade.rollout(
        front=np.zeros((20, 30, 3), dtype=np.uint8),
        wrist=np.zeros((12, 18, 3), dtype=np.uint8),
        instruction="move above the bowl",
        samples=2,
    )

    assert result["front"].shape == (2, 2, 20, 30, 3)
    assert len(images.calls) == 1
    assert images.calls[0]["n"] == 2
    assert "wrist" not in result
    assert images.calls[0]["image"][0][0] == "front.png"
    assert images.calls[0]["image"][1][0] == "wrist.png"
    assert "wrist-camera" not in images.calls[0]["prompt"]


def test_model_info_marks_one_generated_image_and_no_seed_support():
    backend = GPTImageBackend(SimpleNamespace(images=FakeImages([])))
    facade = GPTImageWorldModelFacade(backend)

    info = facade._dispatch("model_info", (), {})

    assert info["model"] == "gpt-image-2"
    assert info["generated_images_per_sample"] == 1
    assert info["input_views"] == ["front", "wrist"]
    assert info["generated_views"] == ["front"]
    assert info["output_frames_per_sample"] == 2
    assert info["seed_supported"] is False


@pytest.mark.parametrize(
    ("payloads", "message"),
    [
        ([], "returned 0 image"),
        (["not-valid-base64"], "could not decode"),
    ],
)
def test_bad_api_image_payloads_fail_clearly(payloads, message):
    backend = GPTImageBackend(SimpleNamespace(images=FakeImages(payloads)))

    with pytest.raises(RuntimeError, match=message):
        backend.generate(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
            instruction="close the gripper",
            samples=1,
        )


def test_nonpositive_sample_count_is_rejected_before_api_call():
    images = FakeImages([])
    backend = GPTImageBackend(SimpleNamespace(images=images))

    with pytest.raises(ValueError, match="between 1 and 10"):
        backend.generate(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
            instruction="close the gripper",
            samples=0,
        )

    assert images.calls == []


def test_failed_request_clears_stale_usage_and_records_request_id():
    class FailingImages:
        def edit(self, **_kwargs):
            error = RuntimeError("upstream failed")
            error.request_id = "req_failed_123"
            raise error

    backend = GPTImageBackend(SimpleNamespace(images=FailingImages()))
    backend._last_request_id = "req_previous"
    backend._last_usage = {"output_tokens": 999}

    with pytest.raises(RuntimeError, match="upstream failed"):
        backend.generate(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
            instruction="close the gripper",
            samples=1,
        )

    assert backend.info()["last_request_id"] == "req_failed_123"
    assert backend.info()["last_usage"] is None


def test_search_falls_back_to_first_candidate_when_every_api_rollout_fails():
    from parallelworld_manipulation.parallelworld.explorer import ParallelWorldExplorer

    class FailingClient:
        def rollout(self, **_kwargs):
            raise RuntimeError("image API unavailable")

    observation = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    explorer = ParallelWorldExplorer(
        primitives=SimpleNamespace(_last_obs=observation),
        client=FailingClient(),
        tts=False,
        tts_samples=3,
        rollout_steps=1,
    )

    result = explorer.search(
        [
            {"id": "a", "primitive": "release", "arguments": {}},
            {"id": "b", "primitive": "release", "arguments": {}},
        ]
    )

    assert result["_selected_candidate"]["id"] == "a"
    assert result["_selection_status"] == "normal"
    assert all("world_model_error" in branch for branch in result["branches"])


def test_search_preserves_first_candidate_fallback_after_partial_failure():
    from parallelworld_manipulation.parallelworld.explorer import ParallelWorldExplorer

    class PartiallyFailingClient:
        def __init__(self):
            self.call_count = 0

        def rollout(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("first image edit failed")
            return {"front": np.asarray(kwargs["front"])[None, None, ...]}

    observation = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    explorer = ParallelWorldExplorer(
        primitives=SimpleNamespace(_last_obs=observation),
        client=PartiallyFailingClient(),
        tts=False,
        tts_samples=3,
        rollout_steps=1,
    )
    result = explorer.search(
        [
            {"id": "failed", "primitive": "release", "arguments": {}},
            {"id": "usable", "primitive": "release", "arguments": {}},
        ]
    )

    assert result["_selected_candidate"]["id"] == "failed"


def test_search_executes_first_candidate_when_verifier_uses_provider_fallback():
    from parallelworld_manipulation.parallelworld.explorer import ParallelWorldExplorer

    class Client:
        def rollout(self, **kwargs):
            return {"front": np.asarray(kwargs["front"])[None, None, ...]}

    class Verifier:
        def decide(self, **_kwargs):
            return {
                "keep": ["a"],
                "dominant_id": "a",
                "continue_search": False,
                "reason": "verifier_fallback: provider timeout",
            }

    observation = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    explorer = ParallelWorldExplorer(
        primitives=SimpleNamespace(_last_obs=observation),
        client=Client(),
        tts=False,
        tts_samples=3,
        rollout_steps=1,
        verifier=Verifier(),
    )

    result = explorer.search([{"id": "a", "primitive": "release", "arguments": {}}])

    assert result["_selected_candidate"]["id"] == "a"
    assert result["_selection_status"] == "normal"


def test_search_falls_back_to_first_candidate_when_verifier_selects_no_route():
    from parallelworld_manipulation.parallelworld.explorer import ParallelWorldExplorer

    class Client:
        def rollout(self, **kwargs):
            return {"front": np.asarray(kwargs["front"])[None, None, ...]}

    class Verifier:
        def decide(self, **_kwargs):
            return {
                "keep": [],
                "dominant_id": None,
                "continue_search": False,
                "reason": "no candidate is safe",
            }

    observation = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    explorer = ParallelWorldExplorer(
        primitives=SimpleNamespace(_last_obs=observation),
        client=Client(),
        tts=False,
        tts_samples=3,
        rollout_steps=1,
        verifier=Verifier(),
    )

    result = explorer.search([{"id": "a", "primitive": "release", "arguments": {}}])

    assert result["_selected_candidate"]["id"] == "a"
    assert result["_selection_status"] == "normal"
