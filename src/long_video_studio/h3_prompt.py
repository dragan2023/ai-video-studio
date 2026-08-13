from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .domain import DialogueLine, ProjectBrief, ShotSpec, ShotTask, WorldBible
from .h3_context import _state_text, compile_ref2va_context
from .style_registry import get_style_contract

# Format contract derived from MiniMax-AI/MiniMax-H3's official
# skills/h3-prompt-writing at fa6891ff7cdaaa03fa4497e89ac64ff169219acf.


@dataclass(frozen=True)
class H3Reference:
    kind: str
    label: str
    description: str
    role: str = "reference"
    relationship: str = ""
    source: str = ""


def render_h3_prompt(
    shot: ShotSpec,
    references: tuple[H3Reference, ...] = (),
    *,
    task: ShotTask | None = None,
    brief: ProjectBrief | None = None,
    world_bible: WorldBible | None = None,
    previous_shot: ShotSpec | None = None,
    speaker_ids: Mapping[str, str] | None = None,
) -> str:
    """Render a structured storyboard shot using the official H3 section order."""

    if (task or shot.task) is ShotTask.REF2VA:
        return compile_ref2va_context(
            shot,
            references,
            brief=brief,
            world_bible=world_bible,
            previous_shot=previous_shot,
            speaker_ids=speaker_ids,
        ).render()
    if brief is not None or world_bible is not None:
        return _render_fl2va_context(
            shot,
            brief=brief,
            world_bible=world_bible,
            speaker_ids=speaker_ids,
        )
    return _render_fl2va(shot)


def _render_fl2va_context(
    shot: ShotSpec,
    *,
    brief: ProjectBrief | None,
    world_bible: WorldBible | None,
    speaker_ids: Mapping[str, str] | None = None,
) -> str:
    style = ""
    if brief is not None:
        contract = get_style_contract(brief.style_preset, brief.style_instructions)
        style = f"The target video uses {contract.compact()} at aspect ratio {brief.aspect_ratio}."
        if getattr(brief, "style_instructions", ""):
            style += f" The creator's visual direction further requires {brief.style_instructions}."
    if world_bible is not None and world_bible.visual_style:
        style += f" The world consistently maintains {world_bible.visual_style}."
    integrated = _integrated_description(shot, style=style, speaker_ids=speaker_ids)
    return "\n\n".join(
        (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
            f"integrated_multimodal_description: {integrated}",
            "overall_soundscape: "
            + (
                shot.audio_prompt
                or "Natural synchronized ambience and physical action sounds continue without a hard seam."
            ),
            f"non_diegetic_music: {shot.music_prompt or 'N/A'}",
        )
    )


def _render_fl2va(shot: ShotSpec) -> str:
    sections = [
        (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        ),
        "integrated_multimodal_description: " + _integrated_description(shot),
        "overall_soundscape: "
        + (
            shot.audio_prompt
            or "Natural synchronized ambience and action sounds only. No dialogue, narration, or voice-over."
        ),
        f"non_diegetic_music: {shot.music_prompt or 'N/A'}",
    ]
    return "\n\n".join(sections)


def _integrated_description(
    shot: ShotSpec,
    *,
    style: str = "",
    speaker_ids: Mapping[str, str] | None = None,
) -> str:
    pieces = [
        f"[Shot 1] {style} " if style else "[Shot 1] ",
        (
            "<Picture 1> establishes the opening-frame subject identities, appearance, composition, "
            "and scene geography. "
        ),
        f"The opening frame shows {shot.opening_state or _state_text(shot.continuity_in)}. ",
        f"{shot.prompt.strip()} The camera follows this direction: {shot.camera.strip()}. ",
    ]
    if shot.hook:
        pieces.append(f"The primary attention beat is {shot.hook}. ")
    if shot.reference_anchors:
        pieces.append("The active reference anchors preserve " + "; ".join(shot.reference_anchors) + ". ")
    for beat in shot.visual_beats:
        pieces.append(
            f"From {beat.start_seconds:.3f}s to {beat.end_seconds:.3f}s, {beat.visual_action}. "
            f"This changes the visible state so that {beat.state_change}. "
            f"During the action, {beat.camera}. The synchronized physical sound is {beat.sound}. "
        )
        pieces.extend(
            _dialogue_for_interval(
                shot,
                beat.start_seconds,
                beat.end_seconds,
                speaker_ids=speaker_ids,
            )
        )
    if not shot.visual_beats:
        pieces.append(_dialogue_description(shot) + " ")
    elif not shot.dialogue:
        pieces.append("No spoken dialogue, narration, or voice-over occurs. ")
    pieces.append(f"By the end, {shot.ending_state or _state_text(shot.continuity_out)}. ")
    if shot.continuity_handoff:
        pieces.append(f"The final moment preserves this continuity for the next clip: {shot.continuity_handoff}. ")
    pieces.append(
        "Do not vocalize visual direction or render burned-in subtitles, captions, UI overlays, logos, "
        "watermarks, or other on-screen text. Preserve temporal consistency, subject identity, object "
        "permanence, and physically plausible motion."
    )
    if shot.negative_prompt:
        pieces.append(f" Avoid these visual outcomes: {shot.negative_prompt.strip()}")
    return "".join(pieces)


def _dialogue_for_interval(
    shot: ShotSpec,
    start: float,
    end: float,
    *,
    speaker_ids: Mapping[str, str] | None = None,
) -> list[str]:
    speakers = _speaker_ids(shot.dialogue, supplied=speaker_ids)
    values: list[str] = []
    for line in shot.dialogue:
        line_start = line.start_seconds if line.start_seconds is not None else 0.0
        if start <= line_start < end or (line_start == shot.duration_seconds == end):
            values.append(_render_dialogue(line, speakers[line.speaker][1]) + " ")
    return values


def _dialogue_description(
    shot: ShotSpec,
    *,
    speakers: dict[str, tuple[int, str]] | None = None,
) -> str:
    if not shot.dialogue:
        return (
            "No spoken dialogue, no narration, and no voice-over. Do not vocalize visual direction, action, "
            "camera instructions, sound notes, or metadata."
        )
    speaker_ids = speakers or _speaker_ids(shot.dialogue)
    return "\n".join(_render_dialogue(line, speaker_ids[line.speaker][1]) for line in shot.dialogue)


def _render_dialogue(line: DialogueLine, speaker_id: str) -> str:
    timing = ""
    if line.start_seconds is not None and line.end_seconds is not None:
        timing = f" From {line.start_seconds:g}s to {line.end_seconds:g}s."
    elif line.start_seconds is not None:
        timing = f" Starting at {line.start_seconds:g}s."
    mode_instruction = {
        "on_screen": "The speaker remains visible and lip movement synchronizes naturally.",
        "off_screen": "The speaker is off screen; visible characters keep their lips closed.",
        "voice_over": "This is voice-over; visible characters keep their lips closed.",
    }[line.mode]
    return (
        f"({speaker_id}) {line.speaker} speaks in a {line.delivery} manner.{timing} {mode_instruction}\n"
        f"<d>[{line.language}] {line.text}</d>"
    )


def _speaker_ids(
    lines: list[DialogueLine],
    *,
    supplied: Mapping[str, str] | None = None,
) -> dict[str, tuple[int, str]]:
    values: dict[str, tuple[int, str]] = {}
    for line in lines:
        if line.speaker not in values:
            supplied_id = supplied.get(line.speaker) if supplied else None
            if supplied_id:
                try:
                    index = int(supplied_id.removeprefix("S"))
                except ValueError:
                    index = len(values) + 1
                values[line.speaker] = (index, supplied_id)
            else:
                index = len(values) + 1
                values[line.speaker] = (index, f"S{index}")
    return values
