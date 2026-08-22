"""厚版脚本解析器回归测试（场1厚版 70 镜）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from long_video_studio.script_importer import parse_shot_script_file

SCRIPT_PATH = Path(
    r"C:/Users/Administrator/Desktop/桌面备份/我的文学/正则宇宙/极乐城/《极乐城》分镜视频脚本·场1厚版.md"
)


@pytest.fixture(scope="module")
def result():
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"脚本不存在: {SCRIPT_PATH}")
    return parse_shot_script_file(SCRIPT_PATH)


def test_parse_all_70_shots(result):
    assert len(result.shots) == 70
    assert result.error_count == 0


def test_no_ref_shots(result):
    assert result.no_ref_shots == ["1-01", "1-02", "1-59"]


def test_dialogue_shots(result):
    # 有口型台词原文的镜头（精确口径）
    expected = ["1-17", "1-31", "1-32", "1-36", "1-37", "1-40", "1-43", "1-56", "1-67"]
    assert sorted(result.dialogue_shots) == sorted(expected)


def test_first_five_shots(result):
    shots = {s.shot_no: s for s in result.shots[:5]}
    assert shots["1-01"].duration_seconds == 5.0
    assert shots["1-01"].refs == {}
    assert shots["1-02"].duration_seconds == 10.0
    assert shots["1-02"].refs == {}
    assert shots["1-03"].refs == {1: ["S-01"]}
    assert shots["1-03"].duration_seconds == 15.0
    assert shots["1-04"].refs == {1: ["S-01"]}
    assert shots["1-05"].refs == {1: ["S-03"]}


def test_dialogue_text_extracted(result):
    shots = {s.shot_no: s for s in result.shots}
    idol_texts = [d.text for d in shots["1-17"].dialogue]
    assert any("加入极乐" in text for text in idol_texts)

    adang_texts = [d.text for d in shots["1-31"].dialogue]
    assert any("又摸进来了" in text for text in adang_texts)

    xiaohe_texts = [d.text for d in shots["1-67"].dialogue]
    assert any("围巾系好" in text for text in xiaohe_texts)


def test_beats_parsed(result):
    first = result.shots[0]
    assert len(first.beats) == 3
    assert first.beats[0].start_seconds == 0.0
    assert first.beats[0].end_seconds == 1.60
