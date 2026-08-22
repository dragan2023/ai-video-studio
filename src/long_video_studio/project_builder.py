"""把厚版脚本解析结果组装为 FilmProject（Nautilus 领域模型）。"""

from __future__ import annotations

from long_video_studio.adapters.black_frame import BLACK_FRAME_MARKER
from long_video_studio.domain import (
    ContinuationMode,
    DialogueLine,
    FilmProject,
    ProjectBrief,
    ShotSpec,
    ShotStatus,
    ShotTask,
    StoryboardBeat,
    UltraFastAnchorStrategy,
    WorldBible,
)
from long_video_studio.script_importer import RawShot

DEFAULT_CAMERA = "medium shot, stable cinematic camera"
DEFAULT_TITLE = "AI视频全自动生产"


def build_film_project(
    raw_shots: list[RawShot],
    code_to_asset_id: dict[str, str],
    *,
    title: str = DEFAULT_TITLE,
    visual_style: str = "",
) -> FilmProject:
    """组装 FilmProject。

    厚版直通约定：
    - shot.prompt = 厚版正文原样（含 @图片N 标签、时间分段、声音设计），
      后续 ComfyUI 适配器直通使用，不经过 render_h3_prompt 二次包装。
    - 有图镜 task=FL2VA，第一张图作 start_frame（首帧锚点），其余进 reference。
    - 无图镜（黑场字幕等）仍生成 ShotSpec，但无参考图，由渲染层用黑场基底处理。
    """
    shots: list[ShotSpec] = []
    for raw in raw_shots:
        asset_ids = [code_to_asset_id[code] for code in raw.ordered_ref_codes if code in code_to_asset_id]
        start_frame_asset_id = asset_ids[0] if asset_ids else None
        dialogue = [
            DialogueLine(
                speaker=item.speaker or "speaker",
                text=item.text,
                language="Chinese",
                delivery="natural",
                mode="on_screen",
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
            )
            for item in raw.dialogue
        ]
        beats = [
            StoryboardBeat(
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                visual_action=item.action,
            )
            for item in raw.beats
        ]
        is_black_card = not asset_ids and any(marker in raw.title for marker in ("黑屏", "字幕"))
        shot = ShotSpec(
            index=raw.index,
            title=raw.title,
            purpose=raw.story_background or f"{raw.shot_type} · {raw.title}",
            source_section=raw.source_section,
            duration_seconds=raw.duration_seconds,
            task=ShotTask.FL2VA,
            prompt=raw.prompt,
            audio_prompt=raw.audio_prompt,
            dialogue=dialogue,
            visual_beats=beats,
            camera=raw.camera or DEFAULT_CAMERA,
            reference_asset_ids=asset_ids,
            start_frame_asset_id=start_frame_asset_id,
            subtitle_text=BLACK_FRAME_MARKER if is_black_card else None,
            seed=42 + raw.index,
            inference_steps=10,
            status=ShotStatus.PLANNED,
        )
        shots.append(shot)

    # 无独立参考图的镜头默认接上前一镜；有明确首帧的镜头保留独立锚点。
    # 这样黑场字幕/缺图镜不会阻塞通用流程，也不会意外重置连续动作。
    previous: ShotSpec | None = None
    for shot in shots:
        if previous is not None and not shot.start_frame_asset_id:
            shot.continuity_from_shot_id = previous.id
            shot.continuation_mode = ContinuationMode.QUALITY
        previous = shot

    total_seconds = int(sum(shot.duration_seconds for shot in shots))
    brief = ProjectBrief(
        title=title,
        prompt=title,
        duration_seconds=max(total_seconds, 15),
        aspect_ratio="16:9",
        style="赛博朋克霓虹 + 超写实摄影（默认）",
        style_preset="cinematic",
        continuation_mode=ContinuationMode.ULTRA_FAST,
        ultra_fast_anchor_strategy=UltraFastAnchorStrategy.INDEPENDENT,
    )
    world_bible = WorldBible(logline=title, visual_style=visual_style)
    return FilmProject(brief=brief, world_bible=world_bible, shots=shots)
