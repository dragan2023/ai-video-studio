"""ComfyUI H3 adapter contract tests; no ComfyUI server required."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from long_video_studio.adapters.comfyui_h3 import ComfyUIH3Client
from long_video_studio.config import Settings
from long_video_studio.domain import ShotSpec
from long_video_studio.runner import RenderManager

WORKFLOW = Path(r"E:/Comfyui-WF-2026.8.8/ComfyUI/user/default/workflows/【Work-Fisher】Minimax-H3 整合流程.json")


@pytest.fixture
def media_files(tmp_path: Path):
    image = tmp_path / "start.png"
    image.write_bytes(b"\x89PNG\r\n")
    return image


def shot(duration: float = 5.0) -> ShotSpec:
    return ShotSpec(
        index=0,
        title="test shot",
        purpose="test",
        duration_seconds=duration,
        prompt="THICK PROMPT DIRECT",
        seed=123,
        inference_steps=10,
    )


def test_frame_alignment():
    assert ComfyUIH3Client.frames_for_duration(5) == 124
    assert ComfyUIH3Client.frames_for_duration(10) == 243
    assert ComfyUIH3Client.frames_for_duration(15) == 362


def test_render_manager_selects_comfyui_backend(monkeypatch):
    monkeypatch.setenv("STUDIO_H3_BACKEND", "comfyui")
    monkeypatch.setenv("STUDIO_COMFYUI_URL", "http://127.0.0.1:8188")
    monkeypatch.setenv("STUDIO_COMFYUI_WORKFLOW", str(WORKFLOW))
    settings = Settings.from_env()
    manager = RenderManager.__new__(RenderManager)
    manager.settings = settings
    assert manager._h3_configured("fl2va")
    assert manager._h3_configured("ref2va")
    assert isinstance(manager._h3_client(None), ComfyUIH3Client)


def test_workflow_template_submission(tmp_path: Path, media_files: Path):
    if not WORKFLOW.is_file():
        pytest.skip(f"workflow not found: {WORKFLOW}")
    state: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/image":
            return httpx.Response(200, json={"name": "start.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            state["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"prompt_id": "p-test"})
        if request.url.path == "/history/p-test":
            return httpx.Response(200, json={"p-test": {"outputs": {"96": {"gifs": [{"filename": "result.mp4", "subfolder": "", "type": "output"}]}}}})
        if request.url.path == "/view":
            return httpx.Response(200, content=b"fake-mp4", headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    client = ComfyUIH3Client(
        "http://comfy.test",
        WORKFLOW,
        transport=httpx.MockTransport(handler),
        poll_seconds=0,
    )
    output = tmp_path / "shot-001.mp4"
    import asyncio
    asyncio.run(client.generate_fl2va(shot(), media_files, output, width=864, height=480))

    graph = state["payload"]["prompt"]
    assert graph["87"]["inputs"]["prompt"] == "THICK PROMPT DIRECT"
    assert graph["87"]["inputs"]["width"] == 864
    assert graph["87"]["inputs"]["height"] == 480
    assert graph["87"]["inputs"]["length"] == 124
    assert graph["89"]["inputs"]["steps"] == 10
    assert graph["92"]["inputs"]["noise_seed"] == 123
    assert graph["87"]["inputs"]["first_frame"] == ["9001", 0]
    assert graph["9001"]["class_type"] == "LoadImage"
    assert graph["9001"]["inputs"]["image"] == "start.png"
    assert output.read_bytes() == b"fake-mp4"


def test_ref2va_template_submission(tmp_path: Path, media_files: Path):
    if not WORKFLOW.is_file():
        pytest.skip(f"workflow not found: {WORKFLOW}")
    state: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/image":
            return httpx.Response(200, json={"name": "ref.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            state["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"prompt_id": "p-ref"})
        if request.url.path == "/history/p-ref":
            return httpx.Response(200, json={"p-ref": {"outputs": {"149": {"gifs": [{"filename": "ref-result.mp4", "type": "output"}]}}}})
        if request.url.path == "/view":
            return httpx.Response(200, content=b"ref-mp4", headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    client = ComfyUIH3Client("http://comfy.test", WORKFLOW, transport=httpx.MockTransport(handler), poll_seconds=0)
    output = tmp_path / "ref-shot.mp4"
    import asyncio
    asyncio.run(client.generate_ref2va(shot(15), media_files, media_files, output, width=864, height=480))

    graph = state["payload"]["prompt"]
    assert graph["153"]["inputs"]["prompt"] == "THICK PROMPT DIRECT"
    assert graph["153"]["inputs"]["width"] == 864
    assert graph["153"]["inputs"]["height"] == 480
    assert graph["153"]["inputs"]["length"] == 362
    assert graph["141"]["inputs"]["steps"] == 10
    assert graph["154"]["inputs"]["image"] == "ref.png"
    assert graph["155"]["inputs"]["image"] == "ref.png"
    assert output.read_bytes() == b"ref-mp4"
