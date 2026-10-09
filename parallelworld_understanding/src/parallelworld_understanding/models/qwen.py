from __future__ import annotations

import json
import os
import re
import site
import sys
from pathlib import Path
from typing import Any


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text or ""):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_partial_json_fields(raw_text: str) -> dict[str, Any] | None:
    raw = raw_text or ""
    recovered: dict[str, Any] = {}
    for key in ("action", "answer", "reasoning"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)', raw, flags=re.DOTALL)
        if match:
            recovered[key] = match.group(1).strip()
    confidence = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if confidence:
        try:
            recovered["confidence"] = float(confidence.group(1))
        except ValueError:
            pass
    return recovered or None


def parse_json_response(raw_text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (raw_text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = extract_json_object(raw)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict):
        return parsed
    partial = extract_partial_json_fields(raw)
    if partial is not None:
        output = dict(fallback or {})
        output.update(partial)
        return output
    output = dict(fallback or {})
    output.setdefault("raw_response", raw)
    output.setdefault("error", "JSON parse failed")
    return output


def _default_model_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    local_3b = repo_root / "models" / "Qwen2.5-VL-3B-Instruct"
    local_7b = repo_root / "models" / "Qwen2.5-VL-7B-Instruct"
    if local_3b.exists():
        return local_3b
    if local_7b.exists():
        return local_7b
    return local_3b


def _prefer_qwen_dependency_paths() -> None:
    """Keep Isaac's bundled botocore from shadowing normal Python deps."""
    preferred = [*site.getsitepackages()]
    explicit_deps = os.environ.get("QWEN_DEPS_DIR")
    if explicit_deps:
        preferred.insert(0, explicit_deps)
    for path in reversed([p for p in preferred if p and Path(p).exists()]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    for name in list(sys.modules):
        if name == "botocore" or name.startswith(("botocore.", "boto3", "boto3.", "s3transfer", "s3transfer.")):
            module = sys.modules.get(name)
            module_file = str(getattr(module, "__file__", ""))
            if "pip_prebundle" in module_file or "isaacsim/extscache" in module_file:
                sys.modules.pop(name, None)


class QwenModel:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        del api_key
        requested_model = model or os.environ.get("QWEN_MODEL_PATH") or str(_default_model_dir())
        self.model_path = self._resolve_model_path(requested_model)
        self.device = os.environ.get("QWEN_DEVICE")
        self.device_map = os.environ.get("QWEN_DEVICE_MAP")
        self.attn_implementation = os.environ.get("QWEN_ATTN_IMPLEMENTATION")
        self._model = None
        self._processor = None
        self._torch = None
        self._auto_processor_cls = None
        self._model_cls = None
        self._load_dependencies()

    def _resolve_model_path(self, model: str) -> str:
        repo_root = Path(__file__).resolve().parents[3]
        aliases = {
            "qwen": _default_model_dir(),
            "qwen2.5-vl-3b": repo_root / "models" / "Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl-7b": repo_root / "models" / "Qwen2.5-VL-7B-Instruct",
            "3b": repo_root / "models" / "Qwen2.5-VL-3B-Instruct",
            "7b": repo_root / "models" / "Qwen2.5-VL-7B-Instruct",
        }
        candidate = aliases.get(str(model).strip().lower())
        if candidate is not None:
            return str(candidate)
        return str(Path(model).expanduser()) if any(sep in str(model) for sep in ("/", "\\")) else str(model)

    def _load_dependencies(self) -> None:
        if self._torch is not None and self._auto_processor_cls is not None and self._model_cls is not None:
            return
        _prefer_qwen_dependency_paths()
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen provider requires local dependencies. Please install at least `torch`, "
                "`transformers`, and `Pillow` in the Python environment used by ESI-Bench."
            ) from exc
        self._torch = torch
        self._auto_processor_cls = AutoProcessor
        self._model_cls = Qwen2_5_VLForConditionalGeneration

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        self._load_dependencies()
        assert self._torch is not None
        assert self._auto_processor_cls is not None
        assert self._model_cls is not None
        torch = self._torch

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
        }
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self._model = self._model_cls.from_pretrained(self.model_path, **model_kwargs)
        if not self.device_map:
            self._model.to(self.device)
        self._processor = self._auto_processor_cls.from_pretrained(self.model_path)
        self._log_runtime_placement()

    def _log_runtime_placement(self) -> None:
        assert self._model is not None
        assert self._torch is not None

        parameter_bytes: dict[str, int] = {}
        parameter_dtypes: set[str] = set()
        for parameter in self._model.parameters():
            device = str(parameter.device)
            parameter_bytes[device] = parameter_bytes.get(device, 0) + parameter.numel() * parameter.element_size()
            parameter_dtypes.add(str(parameter.dtype).removeprefix("torch."))

        placements = ", ".join(
            f"{device}={size / 1024**3:.2f} GiB"
            for device, size in sorted(parameter_bytes.items())
        )
        print(
            "[qwen] "
            f"model_path={self.model_path} requested_device={self.device} "
            f"device_map={self.device_map or 'none'} "
            f"parameter_devices=[{placements}] dtypes={sorted(parameter_dtypes)}",
            flush=True,
        )

        for device in sorted(parameter_bytes):
            torch_device = self._torch.device(device)
            if torch_device.type != "cuda":
                continue
            print(
                "[qwen] "
                f"{device} cuda_allocated={self._torch.cuda.memory_allocated(torch_device) / 1024**3:.2f} GiB "
                f"cuda_reserved={self._torch.cuda.memory_reserved(torch_device) / 1024**3:.2f} GiB",
                flush=True,
            )

    def _convert_contents(self, contents: list[Any]) -> list[dict[str, Any]]:
        message_contents: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, Path):
                path = item.expanduser().resolve()
                message_contents.append({"type": "image", "image": str(path)})
            else:
                message_contents.append({"type": "text", "text": str(item)})
        return message_contents

    def _move_inputs_to_device(self, inputs):
        if self.device_map:
            return inputs
        if self.device:
            return inputs.to(self.device)
        return inputs

    def generate_json(
        self,
        contents: list[Any],
        system_instruction: str,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        fallback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, str | None]:
        del response_schema
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        user_content = self._convert_contents(contents)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": str(system_instruction)}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "do_sample": temperature is not None and temperature > 0,
        }
        if generation_kwargs["do_sample"] and temperature is not None:
            generation_kwargs["temperature"] = temperature
        if generation_kwargs["do_sample"] and top_p is not None:
            generation_kwargs["top_p"] = top_p

        with self._torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)

        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return parse_json_response(output_text, fallback=fallback), output_text, "stop"
