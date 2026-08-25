from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from pathlib import Path

from test_assets import png_bytes

from long_video_studio.assets import AssetService
from long_video_studio.domain import (
    AssetRole,
    FilmProject,
    ProjectBrief,
    ShotSpec,
    ShotTask,
    WorldBible,
)
from long_video_studio.repository import StudioRepository
from long_video_studio.runner import RenderManager


class FakeH3Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_fl2va(self, _shot, _prepared_start, output_path, **_kwargs):
        self.calls.append("fl2va")
        Path(output_path).write_bytes(b"fake-mp4")
        return Path(output_path)

    async def generate_ref2va(self, _shot, _prepared_start, output_path, **_kwargs):
        self.calls.append("ref2va")
        Path(output_path).write_bytes(b"fake-mp4")
        return Path(output_path)


class FakeMedia:
    def __init__(self) -> None:
        self.concatenated: list[Path] = []

    def fit_image_to_canvas(self, src, dst, _w, _h):
        shutil.copyfile(src, dst)
        return Path(dst)

    def extract_last_stable_frame(self, _src, dst):
        Path(dst).write_bytes(b"fake-png")
        return Path(dst)

    def concatenate(self, clips, out, **_kwargs):
        self.concatenated.append(out)
        Path(out).write_bytes(b"fake-final")
        return Path(out)


async def _no_anchor(*_args, **_kwargs):
    return None


def _make_manager(settings, repository):
    manager = RenderManager(settings, repository)
    manager._h3_configured = lambda _task: True  # type: ignore[method-assign]
    fake_h3 = FakeH3Client()
    manager._h3_client = lambda _endpoint=None: fake_h3  # type: ignore[method-assign]
    manager._maybe_make_anchor = _no_anchor  # type: ignore[method-assign]
    manager.media = FakeMedia()
    return manager, fake_h3


def _build_project(settings, shot_count: int = 3):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    frame = assets.ingest_stream(
        png_bytes(), "frame.png", "image/png", roles=[AssetRole.START_FRAME]
    )
    shots = [
        ShotSpec(
            index=i,
            title=f"shot-{i}",
            purpose="test",
            prompt="a test shot",
            duration_seconds=4,
            task=ShotTask.FL2VA,
            start_frame_asset_id=frame.id,
            reference_asset_ids=[frame.id],
        )
        for i in range(shot_count)
    ]
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="test"),
            world_bible=WorldBible(logline="test", visual_style="v"),
            shots=shots,
        )
    )
    return repository, project


def test_single_shot_render_only_renders_target(settings):
    async def scenario() -> None:
        repository, project = _build_project(settings)
        manager, fake_h3 = _make_manager(settings, repository)

        target = sorted(project.shots, key=lambda s: s.index)[1]
        job = manager.submit(project.id, shot_ids=[target.id])
        assert job.shot_ids == [target.id]

        await manager._tasks[job.id]

        assert fake_h3.calls == ["fl2va"]
        persisted = repository.get_project(project.id)
        by_id = {s.id: s for s in persisted.shots}
        assert by_id[target.id].status.value == "complete"
        assert by_id[target.id].selected_take_path
        others = [s for s in persisted.shots if s.id != target.id]
        assert all(s.status.value == "planned" for s in others)
        assert not manager.media.concatenated

        done = repository.get_job(job.id)
        assert done.status == "complete"
        assert "shot-002.mp4" in str(done.output_path)
        assert done.subtitle_path is None

    asyncio.run(scenario())


def test_single_shot_force_does_not_clear_other_takes_or_final(settings):
    async def scenario() -> None:
        repository, project = _build_project(settings)
        manager, fake_h3 = _make_manager(settings, repository)
        ordered = sorted(project.shots, key=lambda s: s.index)
        target = ordered[1]
        other = ordered[0]
        output_dir = manager.settings.output_dir / project.id
        output_dir.mkdir(parents=True, exist_ok=True)
        other_take = output_dir / "shot-001.mp4"
        other_boundary = output_dir / "shot-001-boundary.png"
        old_target_take = output_dir / "shot-002.mp4"
        final = output_dir / "final.mp4"
        for path in (other_take, other_boundary, old_target_take, final):
            path.write_bytes(b"old")
        other.status = other.status.COMPLETE
        other.selected_take_path = str(other_take)
        other.boundary_frame_path = str(other_boundary)
        target.status = target.status.COMPLETE
        target.selected_take_path = str(old_target_take)
        repository.save_project(project)

        job = manager.submit(project.id, force=True, shot_ids=[target.id])
        await manager._tasks[job.id]

        assert fake_h3.calls == ["fl2va"]
        persisted = repository.get_project(project.id)
        by_id = {s.id: s for s in persisted.shots}
        assert by_id[other.id].status.value == "complete"
        assert by_id[other.id].selected_take_path == str(other_take)
        assert other_take.is_file() and other_boundary.is_file()
        assert final.is_file()  # 单镜 force 不删旧 final；后续全量 render 负责重组装
        assert by_id[target.id].status.value == "complete"
        assert by_id[target.id].selected_take_path == str(old_target_take)

    asyncio.run(scenario())


def test_full_render_still_assembles(settings):
    async def scenario() -> None:
        repository, project = _build_project(settings)
        manager, fake_h3 = _make_manager(settings, repository)

        job = manager.submit(project.id)
        assert job.shot_ids is None
        await manager._tasks[job.id]

        assert len(fake_h3.calls) == 3
        assert len(manager.media.concatenated) == 1
        done = repository.get_job(job.id)
        assert done.status == "complete"
        assert "final.mp4" in str(done.output_path)

    asyncio.run(scenario())
