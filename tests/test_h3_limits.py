from __future__ import annotations

import pytest

from long_video_studio.domain import ProjectBrief, ShotSpec
from long_video_studio.h3_limits import (
    H3_MAX_ALIGNED_OUTPUT_SECONDS,
    H3_MAX_OUTPUT_FRAMES,
    H3_MAX_SHOT_SECONDS,
    h3_aligned_duration_seconds,
    h3_aligned_frame_count,
)


def test_h3_maximum_request_is_accepted_but_encoded_duration_is_15_083_seconds():
    assert H3_MAX_SHOT_SECONDS == 15.0
    assert H3_MAX_OUTPUT_FRAMES == 362
    assert h3_aligned_frame_count(round(H3_MAX_SHOT_SECONDS * 24)) == 362
    assert h3_aligned_duration_seconds(H3_MAX_SHOT_SECONDS) == pytest.approx(15.0833333333)
    assert pytest.approx(15.0833333333) == H3_MAX_ALIGNED_OUTPUT_SECONDS


def test_h3_frame_alignment_handles_values_around_the_maximum():
    assert h3_aligned_frame_count(round(14.99 * 24)) == 362
    assert h3_aligned_frame_count(0) == 1
    assert h3_aligned_frame_count(-1) == 1


def test_shot_spec_accepts_h3_maximum_but_not_above_it():
    brief = ProjectBrief(prompt="test", duration_seconds=15)
    assert brief.duration_seconds == 15
    shot = ShotSpec(
        index=0,
        title="Opening",
        purpose="Establish the scene",
        prompt="A continuous opening shot.",
        duration_seconds=15,
    )
    assert shot.duration_seconds == 15
    with pytest.raises(ValueError):
        ShotSpec(
            index=0,
            title="Opening",
            purpose="Establish the scene",
            prompt="A continuous opening shot.",
            duration_seconds=15.01,
        )
