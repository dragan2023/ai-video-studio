"""项目组装器测试（用场1前5镜）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from long_video_studio.project_builder import build_film_project
from long_video_studio.script_importer import parse_shot_script_file

SCRIPT_PATH = Path(
    r"C:/Users/Administrator/Desktop/桌面备份/我的文学/正则宇宙/极乐城/《极乐城》分镜视频脚本·场1厚版.md"
)


@pytest.fixture(scope="module")
def project():
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"脚本不存在: {SCRIPT_PATH}")
    result = parse_shot_script_file(SCRIPT_PATH)
    # 用编号做占位 asset id，验证映射与首帧逻辑
    code_to_id = {f"R-{i:02d}": f"asset:R-{i:02d}" for i in range(1, 8)}
    code_to_id.update({f"S-{i:02d}": f"asset:S-{i:02d}" for i in range(1, 4)})
    code_to_id.update({f"P-{i:02d}": f"asset:P-{i:02d}" for i in range(1, 9)})
    return build_film_project(result.shots, code_to_id, title="极乐城 场1")


def test_project_shot_count(project):
    assert len(project.shots) == 70


def test_first_five_shot_mapping(project):
    shots = {s.title: s for s in project.shots[:5]}
    assert project.shots[0].start_frame_asset_id is None  # 1-01 黑屏
    assert project.shots[1].start_frame_asset_id is None  # 1-02 字幕
    assert project.shots[1].continuity_from_shot_id == project.shots[0].id
    assert project.shots[1].continuation_mode.value == "quality"
    assert project.shots[2].reference_asset_ids == ["asset:S-01"]  # 1-03
    assert project.shots[3].reference_asset_ids == ["asset:S-01"]  # 1-04
    assert project.shots[4].reference_asset_ids == ["asset:S-03"]  # 1-05


def test_prompt_kept_verbatim(project):
    # 厚版直通：prompt 保留原文，含 @图片 标签
    assert "@图片1" in project.shots[2].prompt
    assert project.shots[2].prompt.startswith("生成一段15秒")


def test_timeline(project):
    assert len(project.timeline) == 70
    assert project.timeline[0].duration_seconds == 5.0
    assert project.timeline[2].duration_seconds == 15.0
