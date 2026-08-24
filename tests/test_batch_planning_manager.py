from __future__ import annotations

import asyncio
from dataclasses import replace

from long_video_studio.config import PlannerProfile
from long_video_studio.domain import (
    BatchPlanningRun,
    BatchPlanningStatus,
    FilmProject,
    ProjectBrief,
    ShotSpec,
    WorldBible,
)
from long_video_studio.planner import PlannerService
from long_video_studio.planning import PlanningManager
from long_video_studio.repository import StudioRepository


def test_imported_h3_worker_persists_seventy_shots_in_six_shot_batches(settings, monkeypatch) -> None:
    configured = replace(
        settings,
        planner_profiles=(
            PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "secret", "qwen-plus", "chat_completions"),
        ),
    )
    project = FilmProject(
        brief=ProjectBrief(prompt="Imported long script", duration_seconds=350),
        world_bible=WorldBible(logline="Imported", visual_style="cinematic"),
        shots=[
            ShotSpec(
                index=index,
                title=f"Shot {index}",
                purpose="test",
                source_section="source",
                duration_seconds=5,
                prompt="A stable scene",
            )
            for index in range(70)
        ],
    )
    repository = StudioRepository(configured.database_path)
    repository.save_project(project)
    batches: list[set[str]] = []

    async def fake_enrich(_self, incoming, shot_ids=None, on_progress=None):
        batches.append(shot_ids or set())
        return incoming

    monkeypatch.setattr(PlannerService, "enrich_imported_shots", fake_enrich)
    manager = PlanningManager(configured, repository, PlannerService(configured, repository))

    async def run() -> None:
        while True:
            await manager.start_imported_h3(project.id, profile_id="qwen")
            task = manager._tasks.get(project.id)
            if task is None:
                break
            await task

    asyncio.run(run())
    saved = repository.get_project(project.id)
    assert saved and saved.batch_planning_run
    assert saved.batch_planning_run.status == BatchPlanningStatus.COMPLETE
    assert saved.batch_planning_run.completed_count == 70
    assert [len(batch) for batch in batches] == [6] * 11 + [4]


def test_imported_h3_worker_waits_for_review_after_first_batch(settings, monkeypatch) -> None:
    configured = replace(
        settings,
        planner_profiles=(
            PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "secret", "qwen-plus", "chat_completions"),
        ),
    )
    project = FilmProject(
        brief=ProjectBrief(prompt="Imported long script", duration_seconds=40),
        world_bible=WorldBible(logline="Imported", visual_style="cinematic"),
        shots=[
            ShotSpec(
                index=index,
                title=f"Shot {index}",
                purpose="test",
                source_section="source",
                duration_seconds=5,
                prompt="A stable scene",
            )
            for index in range(8)
        ],
    )
    repository = StudioRepository(configured.database_path)
    repository.save_project(project)
    batches: list[set[str]] = []

    async def fake_enrich(_self, incoming, shot_ids=None, on_progress=None):
        batches.append(shot_ids or set())
        return incoming

    monkeypatch.setattr(PlannerService, "enrich_imported_shots", fake_enrich)
    manager = PlanningManager(configured, repository, PlannerService(configured, repository))

    async def run() -> None:
        await manager.start_imported_h3(project.id, profile_id="qwen")
        await manager._tasks[project.id]

    asyncio.run(run())
    saved = repository.get_project(project.id)
    assert saved and saved.batch_planning_run
    assert saved.batch_planning_run.status == BatchPlanningStatus.PAUSED
    assert saved.batch_planning_run.completed_count == 6
    assert [len(batch) for batch in batches] == [6]


def test_imported_h3_running_state_becomes_resumable_after_restart(settings):
    project = FilmProject(
        brief=ProjectBrief(prompt="Interrupted script"),
        world_bible=WorldBible(logline="Interrupted", visual_style="cinematic"),
        shots=[],
        batch_planning_run=BatchPlanningRun(status=BatchPlanningStatus.RUNNING, completed_shot_ids=["shot_1"]),
    )
    repository = StudioRepository(settings.database_path)
    repository.save_project(project)

    PlanningManager(settings, repository, PlannerService(settings, repository))

    resumed = repository.get_project(project.id)
    assert resumed and resumed.batch_planning_run
    assert resumed.batch_planning_run.status == BatchPlanningStatus.PAUSED
    assert resumed.batch_planning_run.completed_shot_ids == ["shot_1"]


def test_retry_after_complete_resets_and_uses_selected_profile(settings, monkeypatch) -> None:
    configured = replace(
        settings,
        planner_profiles=(
            PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "secret", "qwen-plus", "chat_completions"),
        ),
    )
    prior = BatchPlanningRun(
        status=BatchPlanningStatus.COMPLETE,
        profile_id="default",
        model="deepseek-flash",
        batch_size=6,
        completed_shot_ids=["shot_1"],
    )
    project = FilmProject(
        brief=ProjectBrief(prompt="Re-run script", duration_seconds=40),
        world_bible=WorldBible(logline="Re-run", visual_style="cinematic"),
        shots=[
            ShotSpec(
                index=0, title="Shot 1", purpose="test", source_section="source",
                duration_seconds=5, prompt="A stable scene",
            )
        ],
        batch_planning_run=prior,
    )
    repository = StudioRepository(configured.database_path)
    repository.save_project(project)

    async def fake_enrich(_self, incoming, shot_ids=None, on_progress=None):
        if on_progress is not None:
            for shot_id in (shot_ids or set()):
                await on_progress(shot_id)
        return incoming

    monkeypatch.setattr(PlannerService, "enrich_imported_shots", fake_enrich)
    manager = PlanningManager(configured, repository, PlannerService(configured, repository))

    async def run() -> None:
        await manager.start_imported_h3(project.id, profile_id="qwen", retry=True)
        await manager._tasks[project.id]

    asyncio.run(run())
    saved = repository.get_project(project.id)
    assert saved and saved.batch_planning_run
    assert saved.batch_planning_run.status == BatchPlanningStatus.COMPLETE
    assert saved.batch_planning_run.profile_id == "qwen"
    assert saved.batch_planning_run.model == "qwen-plus"
    assert saved.batch_planning_run.completed_count == 1
