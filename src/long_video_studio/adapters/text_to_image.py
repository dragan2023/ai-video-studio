from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from long_video_studio.config import Settings


@dataclass(frozen=True)
class TextToImageRequest:
    prompt: str
    output_path: Path
    width: int
    height: int
    negative_prompt: str = ""
    seed: int | None = None


class TextToImageProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    async def generate(self, request: TextToImageRequest) -> Path: ...


class DisabledTextToImageProvider:
    @property
    def configured(self) -> bool:
        return False

    async def generate(self, request: TextToImageRequest) -> Path:
        del request
        raise RuntimeError("text-to-image provider is not configured")


class VllmOmniTextToImageProvider:
    """OpenAI-compatible `/v1/images/generations` client for vLLM-Omni."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str | None,
        api_key: str | None,
        timeout_seconds: float,
        steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        # Keep connection establishment bounded while allowing the model's
        # synchronous generation/read to run for the full configured budget.
        # Explicit fields make this contract visible (and testable) instead
        # of relying on httpx's overloaded scalar timeout constructor.
        connect_timeout = min(30.0, max(0.1, timeout_seconds))
        self.timeout = httpx.Timeout(
            connect=connect_timeout,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=connect_timeout,
        )
        self.steps = steps
        self.true_cfg_scale = true_cfg_scale
        self.guidance_scale = guidance_scale
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def generate(self, request: TextToImageRequest) -> Path:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, object] = {
            "prompt": request.prompt,
            "n": 1,
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
            "output_format": "png",
            "num_inference_steps": self.steps,
            "true_cfg_scale": self.true_cfg_scale,
            "guidance_scale": self.guidance_scale,
        }
        if self.model:
            payload["model"] = self.model
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            payload["seed"] = request.seed
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(
                    self._endpoint(),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                image = await self._extract_image(response, client)
        except httpx.RequestError as error:
            # ReadTimeout's string representation is often empty.  Preserve
            # the concrete exception type so the render job/UI can tell a
            # stalled upstream apart from a refused connection.
            detail = str(error).strip() or repr(error)
            raise RuntimeError(
                f"text-to-image request failed at {self._endpoint()}: {type(error).__name__}: {detail}"
            ) from error
        temporary = request.output_path.with_suffix(f"{request.output_path.suffix}.tmp")
        temporary.write_bytes(image)
        temporary.replace(request.output_path)
        return request.output_path

    async def _extract_image(self, response: httpx.Response, client: httpx.AsyncClient) -> bytes:
        if response.headers.get("content-type", "").startswith("image/"):
            return response.content
        payload = response.json()
        candidates: list[object] = []
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                candidates.extend(data)
            images = payload.get("images")
            if isinstance(images, list):
                candidates.extend(images)
            output = payload.get("output")
            if output is not None:
                candidates.append(output)
        for candidate in candidates:
            raw = await self._decode_candidate(candidate, client)
            if raw is not None:
                return raw
        raise RuntimeError(
            "text-to-image response did not contain b64_json, image bytes, or a downloadable URL: "
            + json.dumps(payload, ensure_ascii=False)[:500]
        )

    async def _decode_candidate(self, candidate: object, client: httpx.AsyncClient) -> bytes | None:
        if isinstance(candidate, dict):
            for key in ("b64_json", "base64", "image_base64"):
                value = candidate.get(key)
                if isinstance(value, str):
                    return self._decode_base64(value)
            value = candidate.get("url")
            if isinstance(value, str):
                return await self._download_or_decode(value, client)
        if isinstance(candidate, str):
            return await self._download_or_decode(candidate, client)
        return None

    async def _download_or_decode(self, value: str, client: httpx.AsyncClient) -> bytes:
        if value.startswith("data:"):
            return self._decode_base64(value.split(",", 1)[1])
        if value.startswith("http://") or value.startswith("https://"):
            response = await client.get(value)
            response.raise_for_status()
            return response.content
        return self._decode_base64(value)

    @staticmethod
    def _decode_base64(value: str) -> bytes:
        return base64.b64decode(value)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1/images/generations"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/images/generations"
        return f"{self.base_url}/v1/images/generations"


def text_to_image_provider_from_settings(settings: Settings) -> TextToImageProvider:
    if settings.text_to_image_provider in {"disabled", "none", ""}:
        return DisabledTextToImageProvider()
    if settings.text_to_image_provider not in {"vllm-omni", "openai-compatible"}:
        raise ValueError(f"unsupported text-to-image provider: {settings.text_to_image_provider}")
    if not settings.text_to_image_base_url:
        raise ValueError("text-to-image provider requires STUDIO_T2I_BASE_URL")
    return VllmOmniTextToImageProvider(
        base_url=settings.text_to_image_base_url,
        model=settings.text_to_image_model,
        api_key=settings.text_to_image_api_key,
        timeout_seconds=settings.text_to_image_timeout_seconds,
        steps=settings.text_to_image_steps,
        true_cfg_scale=settings.text_to_image_true_cfg_scale,
        guidance_scale=settings.text_to_image_guidance_scale,
    )
