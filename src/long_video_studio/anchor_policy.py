from __future__ import annotations

from .domain import ShotSpec, ShotTask

IMAGE_EDIT_ANCHOR_MODES = frozenset({"first-shot", "scene-cuts", "every-shot"})


def anchor_selected(shot: ShotSpec, position: int, mode: str) -> bool:
    """Return whether Image Edit should create a planner-authored anchor."""

    if shot.task is not ShotTask.FL2VA or shot.start_frame_asset_id:
        return False
    if mode == "first-shot":
        return position == 0
    if mode == "scene-cuts":
        return not shot.continuity_from_shot_id
    return mode == "every-shot"
