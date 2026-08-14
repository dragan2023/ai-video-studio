"""Provider-neutral image-edit contracts and an OpenAI-compatible adapter.

The studio treats image editing as an optional conditioning stage.  A provider
receives an ordered reference manifest (scene, characters, props, ...), while
the transport remains replaceable: local vLLM-Omni and hosted vendor APIs can
implement the same contract without changing the planner or runner.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class ImageEditCapabilities:
    provider: str
    protocol: str
    supports_multiple_images: bool
    max_references: int


@dataclass(frozen=True)
class ImageEditReference:
    path: Path
    label: str
    role: str
    tags: tuple[str, ...] = ()
    caption: str | None = None


@dataclass(frozen=True)
class ImageEditRequest:
    prompt: str
    references: tuple[ImageEditReference, ...]
    output_path: Path
    model: str | None = None
    width: int | None = None
    height: int | None = None
    negative_prompt: str | None = None
    extra_body: dict[str, object] | None = None


class ImageEditProvider(Protocol):
    @property
    def capabilities(self) -> ImageEditCapabilities: ...

    async def edit(self, request: ImageEditRequest) -> Path: ...


def known_multi_image_support(model: str) -> bool | None:
    """Return a fail-closed capability for known Qwen Image Edit checkpoints.

    Unknown vendor model names keep their operator-configured reference limit;
    only checkpoint families whose public pipeline contract is known are
    inferred here.
    """

    name = model.rstrip("/").rsplit("/", 1)[-1].lower()
    if name == "qwen-image-edit":
        return False
    if name in {"qwen-image-edit-2509", "qwen-image-edit-2511"}:
        return True
    return None


def build_reference_manifest(references: tuple[ImageEditReference, ...]) -> str:
    """Turn asset metadata into deterministic prompt context.

    Metadata is deliberately textual and ordered; image APIs generally do not
    know the studio's role/tag vocabulary.  The numbered labels let a vendor
    model map each image to the requested scene element.
    """

    lines = [
        "REFERENCE MANIFEST (images are supplied in the listed order; bind each "
        "image by its explicit name, role, and visual description):"
    ]
    for index, reference in enumerate(references, start=1):
        tags = ", ".join(reference.tags) or "none"
        caption = reference.caption.strip() if reference.caption and reference.caption.strip() else ""
        role_hint = _role_hint(reference.role)
        description = caption or _fallback_reference_description(reference.role, reference.label, tags)
        lines.append(
            f"[{index}] name={reference.label}; role={reference.role} ({role_hint}); "
            f"tags={tags}; visual_description={description}"
        )
    return "\n".join(lines)


def _role_hint(role: str) -> str:
    return {
        "location": "scene/background 场景/背景",
        "character": "character identity 角色身份",
        "prop": "prop/object 道具/物体",
        "style": "lighting/style 光线/风格",
        "start_frame": "creator-selected composition 创作者指定构图",
        "continuity": "previous-shot boundary 上一镜头边界",
        "audio": "audio metadata (not a visual source)",
        "reference": "supporting visual reference 辅助视觉参考",
    }.get(role, "supporting visual reference 辅助视觉参考")


def _fallback_reference_description(role: str, label: str, tags: str) -> str:
    """Give the image model useful semantics even when caption is empty.

    We deliberately do not hallucinate pixels that were never described by the
    creator.  The model is told what to preserve and is still expected to
    inspect the actual image supplied after the text.
    """

    tag_text = f" Creator tags: {tags}." if tags != "none" else ""
    if role == "character":
        return f"{label} 的角色身份参考；从图像本身读取并保持脸部、发型、体型和服装特征。{tag_text}"
    if role == "location":
        return f"{label} 的场景/地点参考；从图像本身读取并保持建筑、空间布局、透视和环境光。{tag_text}"
    if role == "prop":
        return f"{label} 的道具参考；从图像本身读取并保持形状、材质、颜色和尺度。{tag_text}"
    if role == "style":
        return f"{label} 的风格参考；从图像本身读取色调、光线和质感。{tag_text}"
    if role == "continuity":
        return f"{label} 的上一镜头边界参考；保持主体姿态、空间和运动方向，只从该时刻之后继续。{tag_text}"
    if role == "start_frame":
        return f"{label} 的创作者指定首帧；尽可能保留其构图、主体位置和空间关系。{tag_text}"
    return f"{label} 的辅助视觉参考；只吸收图像中与镜头方向一致的可见特征。{tag_text}"


def compose_image_edit_prompt(references: tuple[ImageEditReference, ...], prompt: str) -> str:
    """Build the exact text sent to an OpenAI-compatible image endpoint."""

    return f"{build_reference_manifest(references)}\n\nDIRECTOR INSTRUCTION:\n{prompt.strip()}"


class OpenAICompatibleImageEditProvider:
    """Qwen/vLLM-Omni or vendor endpoint using multimodal chat content."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 600,
        max_references: int = 4,
        protocol: str = "chat-completions",
        tokenizer_path: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds, connect=30)
        self.max_references = max_references
        if protocol not in {"chat-completions", "images-edits"}:
            raise ValueError(f"unsupported image edit protocol: {protocol}")
        self.protocol = protocol
        self.tokenizer_path = tokenizer_path
        self._tokenizer: object | None = None
        self.transport = transport

    @property
    def capabilities(self) -> ImageEditCapabilities:
        return ImageEditCapabilities(
            provider="openai-compatible",
            protocol=("images-edits-multipart" if self.protocol == "images-edits" else "chat-completions-multimodal"),
            supports_multiple_images=self.max_references > 1,
            max_references=self.max_references,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def build_payload(self, request: ImageEditRequest) -> dict[str, object]:
        if not request.references:
            raise ValueError("image edit requires at least one reference image")
        if len(request.references) > self.max_references:
            raise ValueError(f"provider accepts at most {self.max_references} reference images")
        prompt = compose_image_edit_prompt(request.references, request.prompt)
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": self._data_url(reference.path)},
            }
            for reference in request.references
        )
        body: dict[str, object] = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": content}],
        }
        generation: dict[str, object] = {}
        if request.width is not None:
            generation["width"] = request.width
        if request.height is not None:
            generation["height"] = request.height
        if request.negative_prompt:
            generation["negative_prompt"] = request.negative_prompt
        if request.extra_body:
            generation.update(request.extra_body)
        if generation:
            body["extra_body"] = generation
        return body

    async def edit(self, request: ImageEditRequest) -> Path:
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            self._validate_prompt_budget(request)
            if self.protocol == "images-edits":
                response = await self._post_images_edit(client, request)
            else:
                payload = self.build_payload(request)
                endpoint = self.base_url
                if not endpoint.endswith("/chat/completions"):
                    endpoint = f"{endpoint}/v1/chat/completions"
                response = await client.post(endpoint, headers=self._headers(), json=payload)
            response.raise_for_status()
            output = await self._extract_image(response, client)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = request.output_path.with_suffix(request.output_path.suffix + ".tmp")
        temporary.write_bytes(output)
        temporary.replace(request.output_path)
        return request.output_path

    def _validate_prompt_budget(self, request: ImageEditRequest) -> None:
        """Apply the cheap local cap; Qwen enforces the exact token gate.

        vLLM's generic ``/tokenize`` route does not use a pure diffusion
        pipeline's internal Qwen-Image template. The configured local
        ``tokenizer.json`` is the exact text tokenizer used by the 20260811
        Qwen-Image-Edit pipeline; its direct prompt count equals the pipeline's
        template-baseline-subtracted count. Never silently truncate intent.
        """

        if len(request.prompt) > 1000:
            raise ValueError("image edit prompt exceeds the Studio 1000-character preflight")
        if self.tokenizer_path is None:
            return
        tokenizer_json = (
            self.tokenizer_path if self.tokenizer_path.is_file() else self.tokenizer_path / "tokenizer.json"
        )
        if self._tokenizer is None:
            if not tokenizer_json.is_file():
                # The tokenizer path is an optional local optimization. A
                # remote vLLM-Omni image endpoint may have the checkpoint
                # mounted inside its container while the Studio host does not.
                return
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise RuntimeError(
                    "STUDIO_IMAGE_EDIT_TOKENIZER_PATH requires the optional tokenizers package"
                ) from error
            self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        token_count = len(self._tokenizer.encode(request.prompt, add_special_tokens=False).ids)
        if token_count > 1000:
            raise ValueError(f"image edit prompt is {token_count} Qwen text tokens; the Studio limit is 1000")

    async def _post_images_edit(
        self,
        client: httpx.AsyncClient,
        request: ImageEditRequest,
    ) -> httpx.Response:
        if not request.references:
            raise ValueError("image edit requires at least one reference image")
        if len(request.references) > self.max_references:
            raise ValueError(f"image edit provider accepts at most {self.max_references} reference images")

        data: dict[str, str] = {
            "model": self.model,
            # The first-frame template already contains named reference
            # bindings. Repeating the full manifest here can overflow Qwen's
            # 1024-token prompt limit when several described images are used.
            "prompt": request.prompt.strip(),
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
            "output_format": "png",
        }
        generation: dict[str, object] = {}
        if request.negative_prompt:
            generation["negative_prompt"] = request.negative_prompt
        if request.extra_body:
            generation.update(request.extra_body)
        for key, value in generation.items():
            data[key] = json.dumps(value) if isinstance(value, dict | list | bool) else str(value)

        files = [
            (
                "image",
                (
                    reference.path.name,
                    reference.path.read_bytes(),
                    mimetypes.guess_type(reference.path.name)[0] or "application/octet-stream",
                ),
            )
            for reference in request.references
        ]
        endpoint = self.base_url
        if not endpoint.endswith("/images/edits"):
            endpoint = f"{endpoint}/v1/images/edits"
        headers = {key: value for key, value in self._headers().items() if key != "Content-Type"}
        return await client.post(endpoint, headers=headers, data=data, files=files)

    async def _extract_image(self, response: httpx.Response, client: httpx.AsyncClient) -> bytes:
        payload = response.json()
        candidates: list[object] = []
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            candidates.extend(data)
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            candidates.append(content)
            if isinstance(message, dict):
                candidates.append(message.get("image_url"))
        for candidate in candidates:
            value = self._find_image_value(candidate)
            if not value:
                continue
            if value.startswith("data:"):
                _, encoded = value.split(",", 1)
                return base64.b64decode(encoded)
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"}:
                continue
            fetched = await client.get(value)
            fetched.raise_for_status()
            return fetched.content
        raise ValueError("image edit response did not contain an image")

    @classmethod
    def _find_image_value(cls, value: object) -> str | None:
        if isinstance(value, str):
            match = re.search(r"(?:data:image/[^;]+;base64,[A-Za-z0-9+/=]+|https?://[^\s)]+)", value)
            return match.group(0) if match else None
        if isinstance(value, dict):
            for key in ("b64_json", "url", "image_url"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    if key == "b64_json":
                        return f"data:image/png;base64,{candidate}"
                    return candidate
                nested = cls._find_image_value(candidate)
                if nested:
                    return nested
            return None
        if isinstance(value, list):
            for item in value:
                nested = cls._find_image_value(item)
                if nested:
                    return nested
        return None


def provider_from_settings(settings: object) -> ImageEditProvider | None:
    """Build an optional provider without forcing any vendor dependency."""

    provider = getattr(settings, "image_edit_provider", "disabled").lower()
    if provider in {"", "disabled", "none"}:
        return None
    if provider in {"openai-compatible", "vllm-omni", "qwen-image-edit"}:
        url = getattr(settings, "image_edit_base_url", None)
        model = getattr(settings, "image_edit_model", None)
        if not url or not model:
            raise ValueError("image edit provider requires IMAGE_EDIT_BASE_URL and IMAGE_EDIT_MODEL")
        max_references = getattr(settings, "image_edit_max_references", 4)
        if max_references > 1 and known_multi_image_support(model) is False:
            raise ValueError(
                "Qwen-Image-Edit accepts one input image; set IMAGE_EDIT_MAX_REFERENCES=1 or use Qwen-Image-Edit-2509"
            )
        return OpenAICompatibleImageEditProvider(
            url,
            model,
            api_key=getattr(settings, "image_edit_api_key", None),
            timeout_seconds=getattr(settings, "image_edit_timeout_seconds", 600),
            max_references=max_references,
            protocol=("chat-completions" if provider == "openai-compatible" else "images-edits"),
            tokenizer_path=getattr(settings, "image_edit_tokenizer_path", None),
        )
    raise ValueError(f"unsupported image edit provider: {provider}")
