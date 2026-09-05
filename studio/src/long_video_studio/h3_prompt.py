from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from .dialogue_harness import build_speaker_roster, canonicalize_dialogue
from .domain import DialogueLine, ProjectBrief, ShotSpec, ShotTask, WorldBible
from .h3_context import _state_text, audit_context_ir, compile_ref2va_context, sanitize_audio_prompt
from .style_registry import get_style_contract

# Format contract derived from MiniMax-AI/MiniMax-H3's official
# skills/h3-prompt-writing at fa6891ff7cdaaa03fa4497e89ac64ff169219acf.

LOGGER = logging.getLogger(__name__)


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
        context = compile_ref2va_context(
            shot,
            references,
            brief=brief,
            world_bible=world_bible,
            previous_shot=previous_shot,
            speaker_ids=speaker_ids,
        )
        for warning in audit_context_ir(context):
            LOGGER.warning("H3 Ref2VA prompt audit for shot %s: %s", shot.index + 1, warning)
        return context.render()
    if brief is not None or world_bible is not None:
        return _render_fl2va_context(
            shot,
            brief=brief,
            world_bible=world_bible,
            previous_shot=previous_shot,
            speaker_ids=speaker_ids,
        )
    return _render_fl2va(shot, previous_shot=previous_shot)


def _render_fl2va_context(
    shot: ShotSpec,
    *,
    brief: ProjectBrief | None,
    world_bible: WorldBible | None,
    previous_shot: ShotSpec | None,
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
    integrated = _integrated_description(
        shot,
        style=style,
        previous_shot=previous_shot,
        speaker_ids=speaker_ids,
        world_bible=world_bible,
    )
    return "\n\n".join(
        (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
            f"integrated_multimodal_description: {integrated}",
            "overall_soundscape: "
            + (
                sanitize_audio_prompt(shot.audio_prompt, has_dialogue=bool(shot.dialogue))
                or "Natural synchronized ambience and physical action sounds continue without a hard seam."
            ),
            f"non_diegetic_music: {shot.music_prompt or 'N/A'}",
        )
    )


def _render_fl2va(shot: ShotSpec, *, previous_shot: ShotSpec | None = None) -> str:
    sections = [
        (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        ),
        "integrated_multimodal_description: " + _integrated_description(shot, previous_shot=previous_shot),
        "overall_soundscape: "
        + (
            sanitize_audio_prompt(shot.audio_prompt, has_dialogue=bool(shot.dialogue))
            or "Natural synchronized ambience and action sounds only. No dialogue, narration, or voice-over."
        ),
        f"non_diegetic_music: {shot.music_prompt or 'N/A'}",
    ]
    return "\n\n".join(sections)


def _integrated_description(
    shot: ShotSpec,
    *,
    style: str = "",
    previous_shot: ShotSpec | None = None,
    speaker_ids: Mapping[str, str] | None = None,
    world_bible: WorldBible | None = None,
) -> str:
    if world_bible and world_bible.subjects and shot.dialogue:
        shot = shot.model_copy(update={"dialogue": canonicalize_dialogue(shot.dialogue, world_bible)})
    pieces = [
        f"[Shot 1] {style} " if style else "[Shot 1] ",
        (
            "<Picture 1> establishes the opening-frame subject identities, appearance, composition, "
            "and scene geography. "
        ),
        f"The opening frame shows {shot.opening_state or _state_text(shot.continuity_in)}. ",
        f"{shot.prompt.strip()} The camera follows this direction: {shot.camera.strip()}. ",
        "The boundary strategy is "
        f"{shot.transition_kind.value if hasattr(shot.transition_kind, 'value') else shot.transition_kind}. ",
    ]
    if world_bible and world_bible.subjects:
        pieces.append(_fl2va_speaker_manifest(world_bible, shot.dialogue))
    if shot.continuity_in.fixed_landmarks:
        pieces.append("Fixed landmarks: " + "; ".join(shot.continuity_in.fixed_landmarks) + ". ")
    if shot.continuity_in.character_positions:
        pieces.append("Character positions and facing: " + "; ".join(shot.continuity_in.character_positions) + ". ")
    if shot.continuity_in.exited_characters:
        pieces.append(
            "Keep these subjects off screen unless explicitly reintroduced: "
            + "; ".join(shot.continuity_in.exited_characters)
            + ". "
        )
    if previous_shot:
        previous_end = previous_shot.ending_state or _state_text(previous_shot.continuity_out)
        pieces.append(
            "The previous clip ends in this exact state: "
            f"{previous_end}. Begin after it, hold the inherited pose briefly, and do not replay the prior action. "
        )
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
        if beat.performance:
            pieces.append(f"The visible performance is {beat.performance}. ")
        if beat.spatial_anchor:
            pieces.append(f"The screen-space anchor is {beat.spatial_anchor}. ")
        if beat.handoff:
            pieces.append(f"The beat hands off as follows: {beat.handoff}. ")
        pieces.extend(
            _dialogue_for_interval(
                shot,
                beat.start_seconds,
                beat.end_seconds,
                speaker_ids=speaker_ids,
                world_bible=world_bible,
            )
        )
    if not shot.visual_beats:
        pieces.append(_dialogue_description(shot, world_bible=world_bible) + " ")
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


def _fl2va_speaker_manifest(world_bible: WorldBible, lines: list[DialogueLine]) -> str:
    """Render the same canonical visual/voice binding used by Ref2VA."""

    roster = build_speaker_roster(world_bible, lines)
    if not roster:
        return ""
    entries = []
    for index, binding in enumerate(roster, start=1):
        aliases = ", ".join(binding.aliases) or "none"
        if binding.speaker_id:
            entries.append(
                f"<Subject {index}> is canonical label {binding.label}, with aliases {aliases}, "
                f"and exclusively owns voice ({binding.speaker_id})."
            )
        else:
            entries.append(
                f"<Subject {index}> is silent canonical label {binding.label}, with aliases {aliases}; "
                "assign no speaker ID in this clip."
            )
    return (
        "Canonical speaker/visual bindings: "
        + " ".join(entries)
        + " An alias never creates a new voice, and no visible subject may speak another subject's line. "
    )


def _subject_reference(line: DialogueLine, world_bible: WorldBible | None) -> str:
    if not world_bible or not world_bible.subjects or not line.subject_id:
        return ""
    for index, card in enumerate(world_bible.subjects, start=1):
        if card.subject_id == line.subject_id:
            return f"Subject {index}"
    return ""


def _dialogue_for_interval(
    shot: ShotSpec,
    start: float,
    end: float,
    *,
    speaker_ids: Mapping[str, str] | None = None,
    world_bible: WorldBible | None = None,
) -> list[str]:
    speakers = _speaker_ids(shot.dialogue, supplied=speaker_ids)
    values: list[str] = []
    for line in shot.dialogue:
        line_start = line.start_seconds if line.start_seconds is not None else 0.0
        if start <= line_start < end or (line_start == shot.duration_seconds == end):
            values.append(
                _render_dialogue(
                    line,
                    speakers[line.speaker][1],
                    subject_reference=_subject_reference(line, world_bible),
                )
                + " "
            )
    return values


def _dialogue_description(
    shot: ShotSpec,
    *,
    speakers: dict[str, tuple[int, str]] | None = None,
    world_bible: WorldBible | None = None,
) -> str:
    if not shot.dialogue:
        return (
            "No spoken dialogue, no narration, and no voice-over. Do not vocalize visual direction, action, "
            "camera instructions, sound notes, or metadata."
        )
    speaker_ids = speakers or _speaker_ids(shot.dialogue)
    return "\n".join(
        _render_dialogue(
            line,
            speaker_ids[line.speaker][1],
            subject_reference=_subject_reference(line, world_bible),
        )
        for line in shot.dialogue
    )


def _render_dialogue(
    line: DialogueLine,
    speaker_id: str,
    *,
    subject_reference: str = "",
) -> str:
    timing = ""
    if line.start_seconds is not None and line.end_seconds is not None:
        timing = f" From {line.start_seconds:g}s to {line.end_seconds:g}s."
    elif line.start_seconds is not None:
        timing = f" Starting at {line.start_seconds:g}s."
    mode_instruction = {
        "on_screen": (
            "This bound visual subject is the only speaker; its lip movement synchronizes naturally while every "
            "other visible subject keeps their lips closed and does not inherit this voice."
        ),
        "off_screen": "The speaker is off screen; visible characters keep their lips closed.",
        "voice_over": "This is voice-over; visible characters keep their lips closed.",
    }[line.mode]
    subject_prefix = f"<{subject_reference}> " if subject_reference else ""
    return (
        f"{subject_prefix}({speaker_id}) {line.speaker} speaks in a {line.delivery} manner.{timing} "
        f"{mode_instruction}\n"
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
            supplied_id = line.speaker_id or (supplied.get(line.speaker) if supplied else None)
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
