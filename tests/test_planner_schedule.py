from __future__ import annotations

from long_video_studio.domain import ProjectBrief
from long_video_studio.planner import PlannerService


def _brief(duration: int, prompt: str) -> ProjectBrief:
    return ProjectBrief(prompt=prompt, duration_seconds=duration, style_preset="cinematic")


def test_explicit_four_scene_script_uses_four_fifteen_second_clips():
    prompt = """
**1-1 GPU 黑市摊位 夜 内**
问价。
**1-2 GPU 检查 夜 内**
拆穿。
**1-3 仓库门口 夜 外**
围场。
**1-4 离场 夜 外**
打斗后离开。
"""
    schedule = PlannerService._director_shot_schedule(_brief(60, prompt))
    assert schedule["explicit_source_sections"] == [
        "1-1 GPU 黑市摊位 夜 内",
        "1-2 GPU 检查 夜 内",
        "1-3 仓库门口 夜 外",
        "1-4 离场 夜 外",
    ]
    assert schedule["minimum_shot_count"] == 4
    assert schedule["target_shot_count"] == 4


def test_time_ranges_are_not_treated_as_source_scene_headings():
    schedule = PlannerService._director_shot_schedule(_brief(60, "0-10秒（开场）\n11-25秒（冲突）\n26-60秒（结尾）"))
    assert schedule["explicit_source_sections"] == []
    assert schedule["target_shot_count"] == 4


def test_dynamic_director_schema_keeps_bounded_dialogue_overflow_budget():
    brief = _brief(60, "**1-1 开场**\n**1-2 冲突**\n**1-3 反转**\n**1-4 离场**")
    schema = PlannerService._director_json_schema(brief)
    assert schema["properties"]["shot_blueprints"]["minItems"] == 4
    assert schema["properties"]["shot_blueprints"]["maxItems"] == 6
    assert schema["$defs"]["ShotBlueprint"]["properties"]["duration_seconds"]["maximum"] == 15


def test_duration_boundaries_use_fewer_clips_than_the_old_fourteen_second_cap():
    assert PlannerService._director_shot_schedule(_brief(57, "plain story")).get("target_shot_count") == 4
    assert PlannerService._director_shot_schedule(_brief(58, "plain story")).get("target_shot_count") == 4
    assert PlannerService._director_shot_schedule(_brief(900, "plain story")).get("target_shot_count") == 60
