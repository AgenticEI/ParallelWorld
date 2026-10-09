from __future__ import annotations

import os
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

class GLMModel:
    provider_name = "glm"

    def __init__(self, api_key: str | None = None, model: str = "glm-4.5v"):
        key = api_key if api_key is not None else os.environ.get("ZHIPU_API_KEY", "")
        if not key:
            key = os.environ.get("GLM_API_KEY", "")
        if not key:
            raise ValueError(
                "Missing GLM API key. Set ZHIPU_API_KEY or GLM_API_KEY, or pass --api-key."
            )
        self.api_key = key
        self.endpoint = self._resolve_endpoint()
        self.model = model

    @staticmethod
    def _resolve_endpoint() -> str:
        endpoint = (
            os.environ.get("GLM_ENDPOINT")
            or os.environ.get("ZHIPU_ENDPOINT")
            or ""
        ).strip()
        base_url = (
            os.environ.get("GLM_BASE_URL")
            or os.environ.get("ZHIPU_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4"
        ).strip()
        if endpoint:
            return endpoint.rstrip("/")
        return f"{base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _image_url_part(image_path: Path) -> dict[str, Any]:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type or 'image/png'};base64,{encoded}"},
        }

    def _convert_contents(self, contents: list[Any]) -> list[dict[str, Any]]:
        converted = []
        for item in contents:
            if isinstance(item, Path):
                converted.append(self._image_url_part(item))
            else:
                converted.append({"type": "text", "text": str(item)})
        return converted

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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": self._convert_contents(contents)},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GLM API HTTP {exc.code}: {detail}") from exc

        choice = body["choices"][0]
        raw_text = (choice.get("message", {}).get("content") or "").strip()
        parsed = self._parse_json_response(raw_text, fallback)
        return parsed, raw_text, str(choice.get("finish_reason"))

    @staticmethod
    def _parse_json_response(
        raw_text: str, fallback: dict[str, Any] | None
    ) -> dict[str, Any]:
        raw = raw_text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].removeprefix("json").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
            decoder = json.JSONDecoder()
            for index, char in enumerate(raw):
                if char not in "{[":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(raw[index:])
                except json.JSONDecodeError:
                    continue
                parsed = candidate
                break
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if isinstance(parsed, dict):
            return parsed
        output = dict(fallback or {})
        output.update({"raw_response": raw_text, "error": "JSON parse failed"})
        return output
