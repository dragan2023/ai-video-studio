from __future__ import annotations

from dataclasses import replace

import pytest

from long_video_studio.domain import (
    FilmProject,
    ProjectBrief,
    RenderObservation,
    ShotSpec,
    ShotTask,
    WorldBible,
)
from long_video_studio.estimator import RenderEstimator
from long_video_studio.repository import StudioRepository


def test_render_estimator_uses_robust_profile_history_and_scales_work(settings):
    configured = replace(
        settings,
        h3_fl2va_url="http://fl2va.test",
        h3_ref2va_url="http://ref2va.test",
    )
    repository = StudioRepository(configured.database_path)
    estimator = RenderEstimator(configured, repository)
    for index, elapsed in enumerate([1700.0, 1710.0, 1714.0, 1720.0, 6000.0]):
        repository.save_render_observation(
            RenderObservation(
                source_key=f"history-{index}",
                project_id=f"project-{index}",
                shot_id=f"shot-{index}",
                render_profile=configured.render_profile,
                task=ShotTask.REF2VA,
                continuation_mode="quality",
                aspect_ratio="16:9",
                duration_seconds=10,
                inference_steps=50,
                elapsed_seconds=elapsed,
            )
        )
    first = ShotSpec(
        index=0,
        title="Opening",
        purpose="Open",
        duration_seconds=5,
        task=ShotTask.FL2VA,
        prompt="Opening.",
    )
    continuation = ShotSpec(
        index=1,
        title="Continuation",
        purpose="Continue",
        duration_seconds=5,
        inference_steps=25,
        task=ShotTask.REF2VA,
        prompt="Continue.",
        continuity_from_shot_id=first.id,
        continuation_mode="quality",
    )
    project = FilmProject(
        brief=ProjectBrief(prompt="A continuous scene.", continuation_mode="quality"),
        world_bible=WorldBible(logline="Continuous", visual_style="Natural"),
        shots=[first, continuation],
    )

    estimate = estimator.estimate_shot(project, continuation)

    assert estimate.source == "history"
    assert estimate.sample_count == 5
    assert estimate.confidence == "high"
    assert estimate.estimated_seconds == pytest.approx(428.0)


def test_render_estimator_falls_back_to_current_deployment_baseline(settings):
    repository = StudioRepository(settings.database_path)
    estimator = RenderEstimator(settings, repository)
    shot = ShotSpec(
        index=0,
        title="Opening",
        purpose="Open",
        duration_seconds=10,
        inference_steps=50,
        task=ShotTask.FL2VA,
        prompt="Opening.",
    )
    project = FilmProject(
        brief=ProjectBrief(prompt="A short scene."),
        world_bible=WorldBible(logline="Short", visual_style="Natural"),
        shots=[shot],
    )

    estimate = estimator.estimate_project(project)

    assert estimate.source == "configured"
    assert estimate.shots[0].estimated_seconds == pytest.approx(396.2)
    assert estimate.total_seconds == pytest.approx(408.2)
