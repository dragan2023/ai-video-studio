from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from long_video_studio.adapters.text_to_image import (
    TextToImageRequest,
    VllmOmniTextToImageProvider,
)
from long_video_studio.runner import RenderManager


def test_vllm_omni_text_to_image_posts_generation_payload(tmp_path):
    image = b"generated-png"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/generations"
        payload = json.loads(request.content)
        assert payload["model"] == "Qwen/Qwen-Image-2512"
        assert payload["size"] == "1280x720"
        assert payload["num_inference_steps"] == 50
        assert payload["true_cfg_scale"] == 4.0
        assert payload["guidance_scale"] == 1.0
        assert payload["negative_prompt"] == "text, watermark"
        assert payload["seed"] == 7
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(image).decode()}]},
        )

    provider = VllmOmniTextToImageProvider(
        base_url="http://t2i.test/v1",
        model="Qwen/Qwen-Image-2512",
        api_key=None,
        timeout_seconds=30,
        steps=50,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        transport=httpx.MockTransport(handler),
    )
    output = tmp_path / "anchor.png"
    result = asyncio.run(
        provider.generate(
            TextToImageRequest(
                prompt="A cinematic opening frame.",
                negative_prompt="text, watermark",
                output_path=output,
                width=1280,
                height=720,
                seed=7,
            )
        )
    )

    assert result == output
    assert output.read_bytes() == image
    assert not output.with_suffix(".png.tmp").exists()


def test_text_to_image_model_is_optional_and_complete_route_is_preserved(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/images/generations"
        assert "model" not in payload
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"image").decode()}]},
        )

    provider = VllmOmniTextToImageProvider(
        base_url="http://t2i.test/v1/images/generations",
        model=None,
        api_key=None,
        timeout_seconds=30,
        steps=2,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(
        provider.generate(
            TextToImageRequest(
                prompt="frame",
                output_path=tmp_path / "frame.png",
                width=64,
                height=64,
            )
        )
    )


def test_text_to_image_rejects_response_without_image(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"data": []})

    provider = VllmOmniTextToImageProvider(
        base_url="http://t2i.test",
        model=None,
        api_key=None,
        timeout_seconds=30,
        steps=2,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="did not contain"):
        asyncio.run(
            provider.generate(
                TextToImageRequest(
                    prompt="frame",
                    output_path=tmp_path / "frame.png",
                    width=64,
                    height=64,
                )
            )
        )


def test_text_to_image_uses_long_read_budget_and_bounded_connect_timeout():
    provider = VllmOmniTextToImageProvider(
        base_url="http://t2i.test",
        model=None,
        api_key=None,
        timeout_seconds=7200,
        steps=2,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
    )

    assert provider.timeout_seconds == 7200
    assert provider.timeout.connect == 30
    assert provider.timeout.read == 7200
    assert provider.timeout.write == 7200


def test_text_to_image_surfaces_timeout_exception_type(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    provider = VllmOmniTextToImageProvider(
        base_url="http://t2i.test",
        model=None,
        api_key=None,
        timeout_seconds=7200,
        steps=2,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match=r"ReadTimeout"):
        asyncio.run(
            provider.generate(
                TextToImageRequest(
                    prompt="frame",
                    output_path=tmp_path / "frame.png",
                    width=64,
                    height=64,
                )
            )
        )


def test_render_error_formatter_keeps_empty_timeout_type():
    error = httpx.ReadTimeout("", request=httpx.Request("POST", "http://t2i.test"))

    assert RenderManager._format_error(error).startswith("ReadTimeout:")
