from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from long_video_studio.domain import (
    FilmProject,
    PreproductionPlan,
    PreproductionShotPlan,
    PreproductionStatus,
    ProjectBrief,
    ShotSpec,
    StartFrameSource,
    WorldBible,
)


def project() -> FilmProject:
    return FilmProject(
        brief=ProjectBrief(prompt="A short proof film."),
        world_bible=WorldBible(logline="Proof", visual_style="cinematic"),
        shots=[ShotSpec(index=0, title="Opening", purpose="Set tone", duration_seconds=5, prompt="A black screen")],
    )


def test_long_project_duration_is_allowed_for_thick_script_imports():
    assert ProjectBrief(prompt="Long film", duration_seconds=1640).duration_seconds == 1640


def test_legacy_project_payload_without_preproduction_plan_still_loads():
    payload = project().model_dump(mode="json")
    payload.pop("preproduction_plan")

    restored = FilmProject.model_validate(payload)

    assert restored.preproduction_plan is None


def test_creator_asset_preproduction_plan_round_trips_on_project():
    value = project()
    value.preproduction_plan = PreproductionPlan(
        status=PreproductionStatus.AWAITING_APPROVAL,
        shot_plans=[
            PreproductionShotPlan(
                shot_id=value.shots[0].id,
                shot_index=0,
                start_frame_source=StartFrameSource.CREATOR_ASSET,
                selected_asset_id="asset_opening",
                confidence=1,
            )
        ],
    )

    restored = FilmProject.model_validate(value.model_dump(mode="json"))

    assert restored.preproduction_plan is not None
    assert restored.preproduction_plan.shots[0].selected_asset_id == "asset_opening"


def test_first_shot_cannot_use_previous_boundary():
    with pytest.raises(ValidationError, match="first shot"):
        PreproductionPlan(
            shot_plans=[
                PreproductionShotPlan(
                    shot_id="shot_1",
                    shot_index=0,
                    start_frame_source=StartFrameSource.PREVIOUS_BOUNDARY,
                    source_shot_id="shot_before",
                )
            ]
        )


def test_ready_plan_requires_approval_and_no_blockers():
    approved_at = datetime.now(timezone.utc)
    ready = PreproductionPlan(
        status=PreproductionStatus.READY,
        approved_at=approved_at,
        shot_plans=[PreproductionShotPlan(shot_id="shot_1", shot_index=0)],
    )
    assert ready.status == PreproductionStatus.READY

    with pytest.raises(ValidationError, match="requires approval"):
        PreproductionPlan(status=PreproductionStatus.READY)
    with pytest.raises(ValidationError, match="blockers"):
        PreproductionPlan(status=PreproductionStatus.READY, approved_at=approved_at, blockers=["asset missing"])
