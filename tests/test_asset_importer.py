"""资产导入器测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from long_video_studio.asset_importer import detect_missing_codes, scan_asset_codes

ASSET_ROOT = Path(r"C:/Users/Administrator/Desktop/桌面备份/我的文学/正则宇宙/极乐城/图像资产")


@pytest.fixture(scope="module")
def entries():
    if not ASSET_ROOT.is_dir():
        pytest.skip(f"资产目录不存在: {ASSET_ROOT}")
    return scan_asset_codes(ASSET_ROOT)


def test_characters_scanned(entries):
    for code in ["R-01", "R-02", "R-03", "R-04", "R-05", "R-06", "R-07"]:
        assert code in entries, f"缺少角色 {code}"
        assert entries[code].kind == "character"


def test_locations_scanned(entries):
    for code in ["S-01", "S-02", "S-03"]:
        assert code in entries, f"缺少场景 {code}"
        assert entries[code].kind == "location"


def test_character_primary_is_front_view(entries):
    primary = entries["R-01"].primary
    assert primary.name.startswith("①")
    assert len(entries["R-01"].extras) == 2


def test_missing_props_detected(entries):
    missing = detect_missing_codes({"P-01", "P-02", "P-07", "S-01"}, entries)
    assert missing == ["P-01", "P-02", "P-07"]
