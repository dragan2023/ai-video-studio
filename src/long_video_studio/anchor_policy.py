from __future__ import annotations

from .domain import ShotSpec, ShotTask, TransitionKind

IMAGE_EDIT_ANCHOR_MODES = frozenset({"first-shot", "scene-cuts", "every-shot"})


def anchor_selected(shot: ShotSpec, position: int, mode: str) -> bool:
    """Return whether an image provider should create a planner-authored anchor."""

    if shot.task is not ShotTask.FL2VA or shot.start_frame_asset_id:
        return False
    if shot.transition_kind in {
        TransitionKind.ANCHOR,
        TransitionKind.HARD_CUT,
        TransitionKind.MATCH_CUT,
        TransitionKind.OCCLUSION_CUT,
    }:
        return True
    if mode == "first-shot":
        return position == 0
    if mode == "scene-cuts":
        return not shot.continuity_from_shot_id
    return mode == "every-shot"
