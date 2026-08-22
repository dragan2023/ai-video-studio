"""ComfyUI Work-Fisher MiniMax H3 adapter."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from long_video_studio.adapters.comfyui_api import load_ui_workflow, patch_inputs, ui_workflow_to_api
from long_video_studio.domain import ProjectBrief, ShotSpec, WorldBible


@dataclass(frozen=True)
class _Branch:
    h3: str
    prompt: str
    output: str
    scheduler: str
    noise: str
    default_load_image: str | None = None


_BRANCHES = {
    "fl2va_short": _Branch("87", "6", "96", "89", "92"),
    "fl2va_long": _Branch("113", "118", "122", "112", "105", "123"),
    "ref2va": _Branch("153", "151", "149", "141", "139"),
}


class ComfyUIH3Client:
    """Submit the pinned Work-Fisher UI workflow through ComfyUI's API."""

    def __init__(
        self,
        endpoint: str,
        workflow_path: str | Path,
        *,
        timeout_seconds: float = 7200,
        width: int = 864,
        height: int = 480,
        steps: int = 10,
        poll_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        client_id: str | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.workflow_path = Path(workflow_path)
        self.timeout_seconds = timeout_seconds
        self.width = width
        self.height = height
        self.steps = steps
        self.poll_seconds = poll_seconds
        self.transport = transport
        self.client_id = client_id or f"nautilus-{uuid.uuid4().hex}"

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
                response = await client.get(f"{self.endpoint}/system_stats")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def generate_fl2va(
        self,
        shot: ShotSpec,
        start_frame: Path,
        output_path: Path,
        *,
        width: int | None = None,
        height: int | None = None,
        async_job: bool = True,
        brief: ProjectBrief | None = None,
        world_bible: WorldBible | None = None,
        previous_shot: ShotSpec | None = None,
        speaker_ids: dict[str, str] | None = None,
    ) -> Path:
        del async_job, brief, world_bible, previous_shot, speaker_ids
        branch = _BRANCHES["fl2va_short" if shot.duration_seconds <= 5 else "fl2va_long"]
        prompt, uploads = await self._new_prompt(branch, output_path)
        first = await self._upload(start_frame)
        filename = _uploaded_name(first)
        load_node = branch.default_load_image
        if load_node is None:
            load_node = self._add_load_image(prompt, filename, "9001")
        else:
            patch_inputs(prompt, load_node, image=filename)
        patch_inputs(
            prompt,
            branch.h3,
            first_frame=[str(load_node), 0],
            prompt=shot.prompt,
            width=int(width or self.width),
            height=int(height or self.height),
            length=self.frames_for_duration(shot.duration_seconds),
        )
        self._patch_sampler(prompt, branch, shot)
        await self._submit_and_download(prompt, branch.output, output_path)
        return output_path

    async def generate_ref2va(
        self,
        shot: ShotSpec,
        reference_image: Path,
        reference_media: Path,
        output_path: Path,
        *,
        width: int | None = None,
        height: int | None = None,
        async_job: bool = True,
        brief: ProjectBrief | None = None,
        world_bible: WorldBible | None = None,
        previous_shot: ShotSpec | None = None,
        speaker_ids: dict[str, str] | None = None,
    ) -> Path:
        del async_job, brief, world_bible, previous_shot, speaker_ids
        branch = _BRANCHES["ref2va"]
        prompt, _ = await self._new_prompt(branch, output_path)
        image_payload = await self._upload(reference_image)
        image_name = _uploaded_name(image_payload)
        patch_inputs(prompt, "154", image=image_name)
        media_payload = await self._upload(reference_media)
        media_name = _uploaded_name(media_payload)
        suffix = reference_media.suffix.lower()
        if suffix in {".mp4", ".mov", ".webm", ".avi"}:
            video_node = self._add_node(prompt, "VHS_LoadVideo", {"video": media_name}, "9002")
            patch_inputs(prompt, branch.h3, **{"ref_videos.ref_video_0": [video_node, 0]})
        elif suffix in {".wav", ".mp3", ".m4a", ".flac", ".aac"}:
            audio_node = self._add_node(prompt, "LoadAudio", {"audio": media_name}, "9003")
            patch_inputs(prompt, branch.h3, **{"ref_audios.ref_audio_0": [audio_node, 0]})
        else:
            patch_inputs(prompt, "155", image=media_name)
        patch_inputs(
            prompt,
            branch.h3,
            prompt=shot.prompt,
            width=int(width or self.width),
            height=int(height or self.height),
            length=self.frames_for_duration(shot.duration_seconds),
            ref_image_size="match",
        )
        self._patch_sampler(prompt, branch, shot)
        await self._submit_and_download(prompt, branch.output, output_path)
        return output_path

    @staticmethod
    def frames_for_duration(duration_seconds: float) -> int:
        raw = max(5, round(float(duration_seconds) * 24))
        return raw + (5 - raw % 17) % 17

    async def _new_prompt(self, branch: _Branch, output_path: Path) -> tuple[dict[str, dict[str, Any]], list[Any]]:
        prompt = ui_workflow_to_api(load_ui_workflow(self.workflow_path))
        prefix = output_path.stem.replace(chr(92), "/")
        patch_inputs(prompt, branch.output, filename_prefix=f"NautilusH3/{prefix}", save_output=True)
        return prompt, []

    @staticmethod
    def _patch_sampler(prompt: dict[str, dict[str, Any]], branch: _Branch, shot: ShotSpec) -> None:
        patch_inputs(prompt, branch.scheduler, steps=int(shot.inference_steps or 10))
        patch_inputs(prompt, branch.noise, noise_seed=int(shot.seed or 0))

    async def _upload(self, path: Path) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=120, transport=self.transport) as client:
                with path.open("rb") as handle:
                    response = await client.post(
                        f"{self.endpoint}/upload/image",
                        data={"overwrite": "true", "type": "input"},
                        files={"image": (path.name, handle, _media_type(path))},
                    )
                response.raise_for_status()
                return response.json()
        except (OSError, httpx.HTTPError) as error:
            raise RuntimeError(f"ComfyUI upload failed for {path}: {error}") from error

    async def _submit_and_download(
        self,
        prompt: dict[str, dict[str, Any]],
        output_node: str,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), transport=self.transport) as client:
                response = await client.post(
                    f"{self.endpoint}/prompt",
                    json={"prompt": prompt, "client_id": self.client_id},
                )
                response.raise_for_status()
                prompt_id = response.json().get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"ComfyUI /prompt returned no prompt_id: {response.text[:500]}")
                deadline = time.monotonic() + self.timeout_seconds
                while True:
                    history = await client.get(f"{self.endpoint}/history/{prompt_id}")
                    history.raise_for_status()
                    payload = history.json().get(str(prompt_id)) or history.json().get(prompt_id)
                    if payload:
                        status = payload.get("status") or {}
                        if status.get("status_str") == "error" or status.get("completed") is False:
                            raise RuntimeError(f"ComfyUI prompt {prompt_id} failed: {status}")
                        media = _find_output(payload.get("outputs", {}).get(str(output_node), {}))
                        if media:
                            view = await client.get(
                                f"{self.endpoint}/view",
                                params={
                                    "filename": media["filename"],
                                    "subfolder": media.get("subfolder", ""),
                                    "type": media.get("type", "output"),
                                },
                            )
                            view.raise_for_status()
                            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                            temporary.write_bytes(view.content)
                            temporary.replace(output_path)
                            return
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {self.timeout_seconds:g}s")
                    await asyncio.sleep(self.poll_seconds)
        except httpx.HTTPError as error:
            raise RuntimeError(f"ComfyUI request failed at {self.endpoint}: {error}") from error

    @staticmethod
    def _add_load_image(prompt: dict[str, dict[str, Any]], filename: str, node_id: str) -> str:
        return ComfyUIH3Client._add_node(prompt, "LoadImage", {"image": filename}, node_id)

    @staticmethod
    def _add_node(prompt: dict[str, dict[str, Any]], class_type: str, inputs: dict[str, Any], node_id: str) -> str:
        prompt[node_id] = {"class_type": class_type, "inputs": dict(inputs)}
        return node_id


def _uploaded_name(payload: dict[str, Any]) -> str:
    name = payload.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload returned no file name: {payload}")
    subfolder = str(payload.get("subfolder") or "").strip("/")
    return f"{subfolder}/{name}" if subfolder else str(name)


def _find_output(output: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("gifs", "videos", "images"):
        values = output.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and value.get("filename"):
                    return value
    return None


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }.get(path.suffix.lower(), "application/octet-stream")
