"""可审生成计划导出器（厚版直通）。

把组装好的 FilmProject 导出为 Markdown 计划文档，供人工审核后一键渲染。
"""

from __future__ import annotations

from pathlib import Path

from long_video_studio.domain import FilmProject, ShotTask


def render_plan_markdown(
    project: FilmProject,
    *,
    code_hints: dict[str, str] | None = None,
    missing_codes: list[str] | None = None,
    max_shots: int | None = None,
) -> str:
    """渲染可审计划。code_hints 把 asset_id 反查回 R/S/P 编号。"""
    code_hints = code_hints or {}
    shots = project.shots if max_shots is None else project.shots[:max_shots]

    lines: list[str] = []
    lines.append(f"# 生成计划 · {project.brief.title}")
    lines.append("")
    lines.append(f"- 总时长：{project.brief.duration_seconds}s｜镜头数：{len(project.shots)}｜展示前 {len(shots)} 镜")
    lines.append(f"- 画幅：{project.brief.aspect_ratio}｜连续策略：{project.brief.continuation_mode.value}")
    lines.append(f"- 锚点策略：{project.brief.ultra_fast_anchor_strategy.value}")
    if missing_codes:
        lines.append(f"- **资产缺口**：{', '.join(missing_codes)}（缺图镜走黑场/T2I 锚点或人工补图）")
    lines.append("")

    for shot in shots:
        ref_codes = [code_hints.get(asset_id, asset_id[:12]) for asset_id in shot.reference_asset_ids]
        start_code = code_hints.get(shot.start_frame_asset_id) if shot.start_frame_asset_id else None
        lines.append(f"## {shot.index + 1:02d} · {shot.title}（{shot.duration_seconds:g}s）")
        lines.append("")
        lines.append(f"- 镜号：{shot.source_section or 'shot'}｜任务：{shot.task.value.upper()}")
        if start_code:
            lines.append(f"- 首帧锚点：{start_code}")
        if ref_codes:
            lines.append(f"- 参考图：{', '.join(ref_codes)}")
        else:
            lines.append("- 参考图：无（黑场/字幕镜，渲染层用黑场基底）")
        if shot.dialogue:
            lines.append(f"- 台词：{len(shot.dialogue)} 句（口型同步）")
            for line in shot.dialogue:
                lines.append(f"  - {line.speaker}：{line.text}")
        if shot.visual_beats:
            lines.append(f"- 时间分段：{len(shot.visual_beats)} 段")
            lines.append(f"  - {shot.visual_beats[0].start_seconds:g}s → {shot.visual_beats[-1].end_seconds:g}s")
        lines.append(f"- 提示词：{_shorten(shot.prompt, 120)}")
        lines.append(f"- 声音设计：{_shorten(shot.audio_prompt, 80) or '（无）'}")
        lines.append("")

    lines.append("---")
    lines.append("> 本计划为厚版直通：shot.prompt 保留脚本原文（含 @图片N 标签），ComfyUI 适配器直通提交，不做二次改写。")
    return "\n".join(lines)


def write_plan(project: FilmProject, output_path: str | Path, **kwargs) -> Path:
    """把可审计划写到文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_plan_markdown(project, **kwargs), encoding="utf-8")
    return path


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
