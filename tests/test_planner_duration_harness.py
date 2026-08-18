from __future__ import annotations

import random

import pytest

from long_video_studio.domain import ProjectBrief
from long_video_studio.planner import PlannerOutput, PlannerService
from long_video_studio.repository import StudioRepository


def _assert_schedule(durations: list[float], requested_duration: int) -> None:
    assert sum(round(duration * 1000) for duration in durations) == requested_duration * 1000
    assert all(4 <= duration <= 15 for duration in durations)
    assert all(round(duration, 3) == duration for duration in durations)


def test_duration_harness_replays_negative_rounding_residual(settings):
    """Replay the 60s/8-shot planner failure that previously produced -0.01s."""

    source_durations = [7, 8, 8, 7, 9, 9, 8, 12]
    planner = PlannerService(settings, StudioRepository(settings.database_path))

    durations = planner._fit_shot_durations(source_durations, 60)

    _assert_schedule(durations, 60)
    assert durations == planner._fit_shot_durations(source_durations, 60)


@pytest.mark.parametrize(
    ("source_durations", "requested_duration"),
    [
        ([8, 8, 8], 12),
        ([8, 8, 8], 42),
        ([1, 100], 18),
        ([4, 14, 9, 6], 33),
        ([0.1] * 225, 900),
    ],
)
def test_duration_harness_covers_bounds_and_skew(
    source_durations: list[float],
    requested_duration: int,
):
    _assert_schedule(
        PlannerService._fit_shot_durations(source_durations, requested_duration),
        requested_duration,
    )


def test_duration_harness_rejects_infeasible_or_invalid_inputs():
    with pytest.raises(ValueError, match="too many shots"):
        PlannerService._fit_shot_durations([8] * 16, 60)
    with pytest.raises(ValueError, match="too few shots"):
        PlannerService._fit_shot_durations([15] * 3, 60)
    with pytest.raises(ValueError, match="positive and finite"):
        PlannerService._fit_shot_durations([8, 0, 8], 24)
    with pytest.raises(ValueError, match="no shots"):
        PlannerService._fit_shot_durations([], 24)


def test_duration_harness_randomized_feasible_matrix():
    rng = random.Random(20260817)
    for _ in range(1_000):
        shot_count = rng.randint(1, 64)
        maximum_duration = min(shot_count * 15, 900)
        requested_duration = rng.randint(shot_count * 4, maximum_duration)
        source_durations = [10 ** rng.uniform(-1, 2) for _ in range(shot_count)]

        durations = PlannerService._fit_shot_durations(source_durations, requested_duration)

        _assert_schedule(durations, requested_duration)


def test_duration_harness_replay_survives_full_output_normalization(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(
        title="Duration replay",
        prompt="Eight distinct shots cross a landscape.",
        duration_seconds=60,
    )
    base = planner._plan_heuristically(brief, [])
    source_durations = [7, 8, 8, 7, 9, 9, 8, 12]
    shots = []
    for index, duration in enumerate(source_durations, start=1):
        shot = base.shots[0].model_copy(
            deep=True,
            update={
                "id": f"shot_{index:03d}",
                "index": index - 1,
                "title": f"Replay shot {index}",
                "prompt": f"{base.shots[0].prompt} Continue with distinct beat {index}.",
                "duration_seconds": duration,
                "visual_beats": [
                    base.shots[0].visual_beats[0].model_copy(update={"start_seconds": 0, "end_seconds": duration})
                ],
                "dialogue": [],
            },
        )
        shots.append(shot)

    normalized = planner._normalize_agent_output(
        PlannerOutput(world_bible=base.world_bible, shots=shots),
        brief,
        [],
    )

    _assert_schedule([shot.duration_seconds for shot in normalized.shots], 60)
    assert all(shot.visual_beats[-1].end_seconds == shot.duration_seconds for shot in normalized.shots)
