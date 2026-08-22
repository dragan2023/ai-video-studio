"""美术资产清单导入器（通用型）。

扫描用户资产目录，建立"编号 → 文件路径"映射，并导入 Nautilus 素材库。
编号约定：R-XX（角色）、S-XX（场景）、P-XX（道具），与厚版脚本 @图片N 行一致。
角色目录内按 ①②③ 命名：① 正面定妆为主参考（identity 锚点），②③ 为补充。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from long_video_studio.assets import AssetService
from long_video_studio.domain import AssetRole

# 编号：R-01 / S-01 / P-07
_CODE_RE = re.compile(r"(?P<kind>[RSP])-(?P<num>\d{2,3})")

_KIND_TO_ROLE = {
    "character": AssetRole.CHARACTER,
    "location": AssetRole.LOCATION,
    "prop": AssetRole.PROP,
}


@dataclass
class AssetCodeEntry:
    code: str
    kind: str  # character / location / prop
    primary: Path
    extras: list[Path] = field(default_factory=list)


@dataclass
class AssetScanResult:
    entries: dict[str, AssetCodeEntry] = field(default_factory=dict)
    missing_codes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        kinds = {"character": 0, "location": 0, "prop": 0}
        for entry in self.entries.values():
            kinds[entry.kind] = kinds.get(entry.kind, 0) + 1
        return (
            f"资产扫描：角色 {kinds.get('character', 0)}、场景 {kinds.get('location', 0)}、"
            f"道具 {kinds.get('prop', 0)}；缺失编号 {len(self.missing_codes)} 个"
        )


def scan_asset_codes(root: str | Path) -> dict[str, AssetCodeEntry]:
    """扫描资产目录，返回编号 → 主图（+补充图）映射。"""
    root = Path(root)
    entries: dict[str, AssetCodeEntry] = {}
    if not root.is_dir():
        return entries

    # 场景图 / 道具图：目录下直接是 S-XX.png / P-XX.png
    for folder, kind in (("场景图", "location"), ("道具图", "prop")):
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        for image in sorted(folder_path.glob("*.png")):
            match = _CODE_RE.search(image.stem)
            if not match:
                continue
            code = f"{match.group('kind')}-{match.group('num')}"
            entries[code] = AssetCodeEntry(code=code, kind=kind, primary=image)

    # 人设图：R-XX 角色目录内 ①②③ 命名
    character_root = root / "人设图"
    if character_root.is_dir():
        for char_dir in sorted(character_root.iterdir()):
            if not char_dir.is_dir():
                continue
            match = _CODE_RE.search(char_dir.name)
            if not match:
                continue
            code = f"{match.group('kind')}-{match.group('num')}"
            images = sorted(char_dir.glob("*.png"))
            if not images:
                continue
            # ① 正面定妆为主；缺 ① 时退回第一张
            primary = next((img for img in images if img.name.startswith("①")), images[0])
            extras = [img for img in images if img != primary]
            entries[code] = AssetCodeEntry(code=code, kind="character", primary=primary, extras=extras)

    return entries


def import_code_assets(asset_service: AssetService, entries: dict[str, AssetCodeEntry]) -> dict[str, str]:
    """把主参考图导入素材库，返回 编号 → asset_id。"""
    imported: dict[str, str] = {}
    for code in sorted(entries):
        entry = entries[code]
        role = _KIND_TO_ROLE[entry.kind]
        records = asset_service.import_path(
            str(entry.primary),
            recursive=False,
            tags=[code.lower(), entry.kind],
            roles=[role],
        )
        if records:
            imported[code] = records[0].id
    return imported


def detect_missing_codes(referenced_codes: set[str], scanned: dict[str, AssetCodeEntry]) -> list[str]:
    """脚本引用但资产目录缺失的编号（按 R/S/P 排序）。"""
    missing = sorted(referenced_codes - set(scanned))
    return missing
