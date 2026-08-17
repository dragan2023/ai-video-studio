"""Context-IR-lite compilation for MiniMax-H3 full-reference prompts.

The planner's JSON is intentionally provider-neutral.  This module is the
small boundary that turns one editable :class:`ShotSpec` into the six-section
English-ish rewrite expected by MiniMax-H3 Ref2VA.  It does not call a model or
inspect files, and therefore remains useful for hosted H3-compatible providers
as well as the local vLLM-Omni adapter.

The implementation accepts the existing ``H3Reference`` objects, mappings, or
small duck-typed objects.  Reference labels are canonicalised once and then
reused in every section.  Callers rendering a whole film can pass the same
``speaker_ids`` mapping to every shot to keep speaker identities stable across
clip boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .domain import DialogueLine, ProjectBrief, ShotSpec, WorldBible
from .style_registry import get_style_contract

SECTION_ORDER = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


@dataclass(frozen=True)
class ContextReference:
    """A normalised reference entry used by the Context IR compiler."""

    kind: str
    label: str
    description: str
    source: str = ""
    role: str = "reference"
    relationship: str = ""


@dataclass(frozen=True)
class ContextIR:
    """The six semantic sections of a MiniMax-H3 Ref2VA rewrite."""

    subject_definitions: str
    summary: str
    retention_analysis: str
    detailed_description: str
    overall_soundscape: str
    non_diegetic_music: str

    def render(self) -> str:
        """Render sections in the exact order required by the H3 skill."""

        values = {
            "subject_definitions": self.subject_definitions,
            "summary": self.summary,
            "retention_analysis": self.retention_analysis,
            "detailed_description": self.detailed_description,
            "overall_soundscape": self.overall_soundscape,
            "non_diegetic_music": self.non_diegetic_music,
        }
        return "\n\n".join(f"{name}:\n{values[name].strip()}" for name in SECTION_ORDER)

    # A prompt-like spelling is convenient for adapters and avoids callers
    # depending on the internal dataclass name.
    to_prompt = render


def audit_context_ir(context: ContextIR, *, minimum_words: int = 300, maximum_words: int = 750) -> tuple[str, ...]:
    """Return actionable quality warnings without mutating a creator prompt.

    H3's guide gives 350-500 words as a useful generation range, not a hard
    tokenizer limit.  The audit deliberately leaves a little headroom for
    dialogue-heavy or reference-heavy shots and reports rather than chopping
    content at arbitrary field boundaries.
    """

    warnings: list[str] = []
    words = len(context.detailed_description.split())
    if words < minimum_words:
        warnings.append(f"detailed_description is short ({words} words; target is about 350-500)")
    if words > maximum_words:
        warnings.append(f"detailed_description is verbose ({words} words; remove repeated boilerplate)")
    rendered = context.render()
    positions = [rendered.find(f"{name}:") for name in SECTION_ORDER]
    if positions != sorted(positions):
        warnings.append("H3 section order is invalid")
    if "<d>" in context.subject_definitions or "<d>" in context.detailed_description:
        warnings.append("dialogue tags leaked into a non-dialogue H3 section")
    return tuple(warnings)


def compile_ref2va_context(
    shot: ShotSpec,
    references: Sequence[object] = (),
    *,
    brief: ProjectBrief | None = None,
    world_bible: WorldBible | None = None,
    previous_shot: ShotSpec | None = None,
    speaker_ids: Mapping[str, str] | None = None,
) -> ContextIR:
    """Compile one shot into a Context-IR-lite Ref2VA prompt.

    ``previous_shot`` is optional because the runtime usually supplies the
    previous clip as ``<Video 1>``.  When available, its ending continuity is
    included as an explicit source state.  ``speaker_ids`` should be a
    project-wide mapping when compiling multiple shots; absent a mapping, IDs
    are assigned deterministically from this shot's dialogue order.
    """

    refs = _normalise_references(references)
    if not refs:
        raise ValueError("Ref2VA context compilation requires at least one reference")

    speaker_map = _normalise_speaker_ids(shot.dialogue, speaker_ids, world_bible=world_bible)
    subject_entries, subject_lookup = _subject_definitions(
        shot,
        refs,
        world_bible=world_bible,
        speaker_ids=speaker_map,
    )
    style_lock = _style_lock(brief, world_bible)
    summary = _summary(shot, refs, previous_shot=previous_shot)
    retention = _retention_analysis(
        shot,
        refs,
        subject_entries,
        previous_shot=previous_shot,
    )
    detailed = _detailed_description(
        shot,
        refs,
        subject_entries,
        subject_lookup,
        style_lock,
        speaker_map,
        previous_shot=previous_shot,
    )
    return ContextIR(
        subject_definitions="\n".join(subject_entries),
        summary=summary,
        retention_analysis="\n".join(retention),
        detailed_description=detailed,
        overall_soundscape=_soundscape(shot, refs),
        non_diegetic_music=shot.music_prompt.strip() or "N/A",
    )


# Name aliases make the boundary discoverable without forcing a particular
# integration spelling on the existing H3 adapter.
render_ref2va_context = compile_ref2va_context
compile_context_ir = compile_ref2va_context


def stable_speaker_ids(shots: Sequence[ShotSpec]) -> dict[str, str]:
    """Assign project-stable speaker IDs in target playback order."""

    result: dict[str, str] = {}
    for shot in sorted(shots, key=lambda value: value.index):
        for line in shot.dialogue:
            if line.speaker not in result:
                result[line.speaker] = f"S{len(result) + 1}"
    return result


def _normalise_references(values: Sequence[object]) -> tuple[ContextReference, ...]:
    counters: dict[str, int] = {}
    used: set[str] = set()
    result: list[ContextReference] = []
    for value in values:
        raw_kind = _field(value, "kind", "reference")
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            raw_kind = value[0] if len(value) > 0 else "reference"
        kind = _canonical_kind(str(raw_kind))
        raw_label = _field(value, "label", "")
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            raw_label = value[1] if len(value) > 1 else ""
        raw_label = str(raw_label or "").strip().strip("<>")
        number = _label_number(raw_label, kind)
        if number is None or f"{kind}:{number}" in used:
            number = counters.get(kind, 0) + 1
            while f"{kind}:{number}" in used:
                number += 1
        counters[kind] = max(counters.get(kind, 0), number)
        used.add(f"{kind}:{number}")
        label = f"{_label_title(kind)} {number}"
        description = _field(value, "description", "")
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            description = value[2] if len(value) > 2 else ""
        description = _sentence(str(description or ""))
        if not description:
            description = f"the {kind} reference supplied for this shot"
        source = _field(value, "source", _field(value, "name", _field(value, "path", "")))
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            source = value[3] if len(value) > 3 else ""
        source = str(source or "").strip()
        role = _field(value, "role", "reference")
        relationship = _field(value, "relationship", "")
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            role = value[4] if len(value) > 4 else "reference"
            relationship = value[5] if len(value) > 5 else ""
        result.append(
            ContextReference(
                kind,
                label,
                description,
                source,
                str(role or "reference").strip().casefold().replace("-", "_"),
                str(relationship or "").strip().casefold(),
            )
        )
    return tuple(result)


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_kind(value: str) -> str:
    value = value.casefold().strip()
    if value in {"image", "picture", "still", "photo"}:
        return "picture"
    if value in {"video", "movie", "clip"}:
        return "video"
    if value in {"audio", "sound", "voice"}:
        return "audio"
    if value in {"subject", "character", "location", "prop", "style"}:
        return "subject"
    return "picture"


def _label_title(kind: str) -> str:
    return {"picture": "Picture", "video": "Video", "audio": "Audio", "subject": "Subject"}[kind]


def _label_number(label: str, kind: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(_label_title(kind))}\s+(\d+)", label, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _sentence(value: str) -> str:
    value = " ".join(value.split()).strip()
    if value and value[-1] not in ".!?。！？":
        value += "."
    return value


def _limit_words(value: str, limit: int) -> str:
    words = " ".join(value.split()).strip()
    parts = words.split()
    if len(parts) <= limit:
        return words.rstrip(". ")
    return " ".join(parts[:limit]).rstrip(".,;: ")


def _prompt_text(value: str, *, limit: int = 12000) -> str:
    """Normalise model-authored prose without discarding useful directives.

    The previous compiler used small word caps (35 words for the main prompt
    and 10 words per beat field).  That made a detailed storyboard look rich in
    the database while silently sending a skeletal prompt to H3.  Keep a very
    generous character guard only for pathological provider output; normal
    shot prose is passed through intact.
    """

    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    marker = " [detail safely capped; beginning and ending retained]"
    if limit <= len(marker):
        return text[:limit]
    available = limit - len(marker)
    # Continuation/no-replay directives are often appended to a planner field.
    # Retain both sides when the pathological-response guard is needed so that
    # the ending directive is not silently lost.
    head = (available + 1) // 2
    tail = available - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _normalise_speaker_ids(
    lines: Sequence[DialogueLine],
    supplied: Mapping[str, str] | None,
    *,
    world_bible: WorldBible | None = None,
) -> dict[str, str]:
    result: dict[str, str] = dict(supplied or {})
    cards = world_bible.subjects if world_bible else ()
    for line in lines:
        for card in cards:
            if card.speaker_id and line.speaker.casefold() in {
                card.label.casefold(),
                card.subject_id.casefold(),
                *(alias.casefold() for alias in card.aliases),
            }:
                result.setdefault(line.speaker, card.speaker_id)
    for line in lines:
        if line.speaker not in result:
            result[line.speaker] = f"S{len(result) + 1}"
    return result


def _subject_definitions(
    shot: ShotSpec,
    refs: Sequence[ContextReference],
    *,
    world_bible: WorldBible | None,
    speaker_ids: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    entries: list[str] = []
    lookup: dict[str, str] = {}
    seen: set[str] = set()

    # World-bible cards are canonical. Per-shot continuity only fills an
    # otherwise missing category, preventing the same person from being
    # redefined as a short name, wardrobe string, and prop on every clip.
    if world_bible and world_bible.subjects:
        card_values = []
        for card in world_bible.subjects:
            details = [
                f"subject_id={card.subject_id}",
                f"label={card.label}",
                f"aliases={', '.join(card.aliases) or 'none'}",
                f"visual_identity={card.visual_identity or 'preserve the canonical identity'}",
                f"wardrobe={card.wardrobe or 'preserve the established wardrobe'}",
            ]
            if card.reference_asset_ids:
                details.append(f"reference_asset_ids={', '.join(card.reference_asset_ids)}")
            if card.speaker_id:
                details.append(f"speaker_id={card.speaker_id}")
            card_values.append("; ".join(details))
        character_values = tuple(card_values[:3]) or _active_canonical_values(
            tuple(world_bible.character_notes),
            tuple(shot.continuity_in.characters),
            limit=3,
        )
    else:
        character_values = _active_canonical_values(
            tuple(world_bible.character_notes) if world_bible else (),
            tuple(shot.continuity_in.characters),
            limit=3,
        )
    location_values = _active_canonical_values(
        tuple(world_bible.location_notes) if world_bible else (),
        (shot.continuity_in.location,) if shot.continuity_in.location else (),
        limit=2,
    )
    prop_values = _active_canonical_values(
        tuple(world_bible.prop_notes) if world_bible else (),
        tuple(shot.continuity_in.props),
        limit=4,
    )
    groups: tuple[tuple[str, Sequence[str]], ...] = (
        ("character", character_values),
        ("location", location_values),
        ("prop", prop_values),
    )

    subject_number = 1
    visual_sources = ", ".join(f"<{reference.label}>" for reference in refs if reference.kind in {"picture", "video"})
    for category, values in groups:
        for value in values:
            text = " ".join(str(value).split()).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            speaker_suffix = ""
            if category == "character":
                for speaker, speaker_id in speaker_ids.items():
                    if _matches_named_character(speaker, text):
                        speaker_suffix = f" ({speaker_id})"
                        lookup[speaker] = f"Subject {subject_number}"
                        break
            grounding = f", grounded in {visual_sources}" if visual_sources else ""
            description = _prompt_text(text, limit=2400) if category == "character" else _limit_words(text, 36)
            entries.append(
                f"<Subject {subject_number}>{speaker_suffix} is the {category} described as {description}{grounding}."
            )
            subject_number += 1

    # Preserve explicitly supplied Subject references and avoid inventing
    # visual facts from opaque filenames.
    for reference in refs:
        if reference.kind == "subject":
            entries.append(f"<{reference.label}> is {reference.description}")
    if not entries:
        entries.append("<Subject 1> is the visible subject and environment defined by the detailed description.")
    for reference in refs:
        if reference.kind != "subject":
            source = f" from {reference.source}" if reference.source else ""
            role = {
                "picture": "the concrete image reference for appearance and composition",
                "video": "the temporal reference for motion, scene geography, and synchronized sound",
                "audio": "the audio reference for the specified sound or voice attributes",
            }[reference.kind]
            entries.append(f"<{reference.label}> is{source} {role} with role={reference.role}: {reference.description}")
    return entries, lookup


def _active_canonical_values(
    canonical: Sequence[str],
    active: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    canonical_values = tuple(value for value in canonical if str(value).strip())
    active_values = tuple(value for value in active if str(value).strip())
    if not active_values:
        return canonical_values
    selected: list[str] = []
    matched_active: set[int] = set()
    for canonical_value in canonical_values:
        matches = [
            index
            for index, active_value in enumerate(active_values)
            if _descriptions_overlap(str(canonical_value), str(active_value))
        ]
        if matches:
            selected.append(str(canonical_value))
            matched_active.update(matches)
    selected.extend(str(value) for index, value in enumerate(active_values) if index not in matched_active)
    return tuple(dict.fromkeys(selected))[:limit]


def _descriptions_overlap(left: str, right: str) -> bool:
    left_key = " ".join(left.casefold().split())
    right_key = " ".join(right.casefold().split())
    if not left_key or not right_key:
        return False
    if left_key in right_key or right_key in left_key:
        return True
    left_words = set(re.findall(r"[a-z0-9]+", left_key))
    right_words = set(re.findall(r"[a-z0-9]+", right_key))
    return bool(left_words and right_words and len(left_words & right_words) >= min(2, len(right_words)))


def _matches_named_character(speaker: str, description: str) -> bool:
    speaker_key = " ".join(speaker.casefold().split())
    description_key = " ".join(description.casefold().split())
    if not speaker_key:
        return False
    if speaker_key.isascii():
        return bool(re.search(rf"\b{re.escape(speaker_key)}\b", description_key))
    return speaker_key in description_key


def _style_lock(brief: ProjectBrief | None, world_bible: WorldBible | None) -> str:
    values: list[str] = []
    if brief:
        custom = getattr(brief, "style_instructions", "").strip()
        if custom:
            values.append(f"Creator style override: {custom}")
    if world_bible and world_bible.visual_style:
        values.append(world_bible.visual_style)
    elif brief:
        contract = get_style_contract(getattr(brief, "style_preset", None), getattr(brief, "style_instructions", ""))
        values.append(contract.compact())
    if brief:
        values.append(f"aspect ratio {brief.aspect_ratio}")
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).split()).strip(" ;")
        lowered = text.casefold()
        if text and lowered not in seen and not any(lowered in existing or existing in lowered for existing in seen):
            seen.add(text.casefold())
            deduped.append(text)
    return "; ".join(deduped) or "consistent cinematic visual language, physically plausible motion"


def _summary(
    shot: ShotSpec,
    refs: Sequence[ContextReference],
    *,
    previous_shot: ShotSpec | None,
) -> str:
    task_types: list[str] = []
    if any(reference.kind == "video" for reference in refs) or previous_shot:
        task_types.append("video continuation")
    if any(
        reference.kind == "picture" and reference.role in {"first_frame", "last_frame", "keyframe"}
        for reference in refs
    ):
        task_types.append("keyframe completion")
    if any(
        reference.kind == "subject" or reference.role in {"identity", "scene", "prop", "style", "reference"}
        for reference in refs
    ):
        task_types.append("reference generation")
    if any(reference.kind == "audio" for reference in refs):
        task_types.append("audio reference")
    prefix = " + ".join(task_types) or "reference generation"
    labels = ", ".join(f"<{reference.label}>" for reference in refs)
    video_reference = next((reference for reference in refs if reference.kind == "video"), None)
    video_label = f"<{video_reference.label}>" if video_reference else "the supplied video reference"
    opening = (
        f"The target clip is a continuation after the final moment of {video_label} using {labels}."
        if previous_shot or any(reference.kind == "video" for reference in refs)
        else f"The target clip uses {labels} as its reference material."
    )
    dramatic_beat = shot.hook.strip() or shot.prompt.strip()
    # Planner prose may be source-language creator copy. Keep the H3 summary
    # model-facing and concise; the full English visual description carries the
    # actual action details below.
    dramatic_beat = _summary_beat(dramatic_beat)
    return f"[{prefix}] {opening} The shot's new dramatic beat is {dramatic_beat}."


def _summary_beat(value: str) -> str:
    words = " ".join(value.split()).strip()
    if not words:
        return "the next observable action advances the story without replaying the reference beat"
    # Preserve non-ASCII planner prose instead of replacing it with a generic
    # sentence.  A later prompt-editor pass may translate it for H3, but this
    # compiler must not discard user-authored hook detail.
    return _prompt_text(words, limit=4000)


def _retention_analysis(
    shot: ShotSpec,
    refs: Sequence[ContextReference],
    subject_entries: Sequence[str],
    *,
    previous_shot: ShotSpec | None,
) -> list[str]:
    lines: list[str] = []
    target_marker = "[Shot 1]"
    for entry in subject_entries:
        match = re.match(r"(<Subject \d+>)", entry)
        if match:
            label = match.group(1)
            lines.append(
                f"{label} (appears in {target_marker}): fully_preserved - identity, wardrobe, and defined "
                "attributes remain stable."
            )
    if previous_shot:
        prior = _prompt_text(previous_shot.ending_state or _state_text(previous_shot.continuity_out))
        lines.append(
            "The supplied temporal reference ends at the prior shot's settled state; preserve its visible subjects, "
            f"screen geography, and audio phase, then advance only after that state is held: {prior}."
        )
    if shot.continuity_in.exited_characters:
        lines.append(
            "Exited-character lock: "
            + "; ".join(_prompt_text(value, limit=800) for value in shot.continuity_in.exited_characters)
            + ". Do not reintroduce an exited subject unless the shot explicitly says so."
        )
    for reference in refs:
        if reference.kind == "video":
            relation = (
                "the final state, motion direction, geography, and synchronized sound are retained "
                "without replaying earlier beats"
            )
        elif reference.kind == "picture" and reference.role in {"first_frame", "keyframe"}:
            relation = (
                "the first generated frame matches this boundary keyframe exactly; preserve subject identities, "
                "composition, screen geography, pose, and lighting before advancing"
            )
        elif reference.kind == "audio":
            relation = "its relevant timbre, rhythm, and sound texture guide the target without inventing a new source"
        else:
            relation = "its appearance, composition, and defining visual attributes are retained where they apply"
        marker = reference.relationship or ("reference" if reference.kind == "audio" else "fully_preserved")
        lines.append(f"<{reference.label}>: {marker} - {relation}.")
    return lines


def _continuity_lines(shot: ShotSpec, *, previous_shot: ShotSpec | None) -> list[str]:
    incoming = _state_text(shot.continuity_in)
    outgoing = _state_text(shot.continuity_out)
    lines = [f"Continuity input lock: {incoming or 'none supplied; preserve the reference state.'}"]
    if previous_shot:
        prior = _state_text(previous_shot.continuity_out)
        lines.append(
            f"Previous shot ending state: {prior or 'the final visible and audible moment of the reference video.'}"
        )
    lines.append(f"Continuity output target: {outgoing or 'end on a stable readable state for the next clip.'}")
    return lines


def _detailed_description(
    shot: ShotSpec,
    refs: Sequence[ContextReference],
    subject_entries: Sequence[str],
    subject_lookup: Mapping[str, str],
    style_lock: str,
    speaker_ids: Mapping[str, str],
    *,
    previous_shot: ShotSpec | None,
) -> str:
    labels = ", ".join(f"<{reference.label}>" for reference in refs)
    intro = (
        f"The target clip follows this immutable visual style lock: {_prompt_text(style_lock)}. "
        f"The reference material {labels} remains active where named below. Do not invent a new palette, lens, "
        "lighting direction, film grain, or character design between beats."
    )
    prior = ""
    if previous_shot:
        previous_ending = previous_shot.ending_state or _state_text(previous_shot.continuity_out)
        prior = (
            " The opening state is the first new moment after the reference video's final visible and audible "
            "state; do not replay it. Previous shot ending state: "
            f"{_prompt_text(previous_ending or 'the final visible and audible moment of the reference video')}."
        )
    first_frame_refs = [
        reference for reference in refs if reference.kind == "picture" and reference.role in {"first_frame", "keyframe"}
    ]
    if first_frame_refs:
        first_frame_labels = ", ".join(f"<{reference.label}>" for reference in first_frame_refs)
        prior += (
            f" The first generated frame must match {first_frame_labels} exactly as the boundary keyframe; "
            "hold the same subject identities, composition, screen geography, pose, and lighting before any "
            "new action begins."
        )
    visual_prompt = _clean_visual_prompt(shot.prompt, camera=shot.camera)
    visual = (
        f"[Shot 1] {visual_prompt}. "
        f"The camera {_prompt_text(shot.camera)}."
        f"{prior} The visual direction is non-spoken and must not be vocalized or rendered as text."
    )
    transition_kind = shot.transition_kind.value if hasattr(shot.transition_kind, "value") else shot.transition_kind
    visual += f" Boundary strategy: {transition_kind}."
    if any(reference.kind == "video" for reference in refs):
        visual += (
            " The first 0.5 to 1.0 seconds must match the supplied video's final readable state and keyframe; "
            "advance only after that hold, and never replay the preceding action or duplicate its rhythm."
        )
    if shot.hook:
        visual += f" The primary hook is {_prompt_text(shot.hook)}."
    anchors = [
        anchor
        for anchor in shot.reference_anchors
        if anchor and not re.match(r"(?i)^(?:global\s+)?style\s+(?:lock|continuity)", anchor.strip())
    ]
    if anchors:
        visual += " Preserve these active anchors exactly: " + _prompt_text("; ".join(anchors)) + "."
    subject_labels = _active_subject_labels(subject_entries, shot.continuity_in.characters)
    if subject_labels:
        visible = ", ".join(f"<{label}>" for label in subject_labels)
        visual = f"The active visible subjects are {visible}. " + visual
    if shot.continuity_in.fixed_landmarks:
        visual += (
            " Fixed landmarks and screen-relative positions: "
            + _prompt_text("; ".join(shot.continuity_in.fixed_landmarks))
            + "."
        )
    if shot.continuity_in.character_positions:
        visual += (
            " Character positions, facing, and initial poses: "
            + _prompt_text("; ".join(shot.continuity_in.character_positions))
            + "."
        )
    if shot.continuity_in.exited_characters:
        visual += (
            " These subjects remain off screen unless explicitly reintroduced: "
            + _prompt_text("; ".join(shot.continuity_in.exited_characters))
            + "."
        )
    timeline = _timeline(shot, speaker_ids=speaker_ids, subject_lookup=subject_lookup)
    return " ".join(value for value in (intro, visual, timeline) if value)


def _clean_visual_prompt(value: str, *, camera: str) -> str:
    """Remove compiler-owned boilerplate while retaining model-authored action."""

    text = _prompt_text(value)
    text = re.sub(r"(?i)\b(?:story\s+premise|global\s+style\s+lock|aspect\s+ratio):[^.。]*[.。]", "", text)
    text = re.sub(r"(?i)\bthis\s+is\s+visual\s+direction\s+only\.?", "", text)
    camera_text = " ".join(camera.split()).strip()
    if camera_text:
        text = re.sub(rf"(?i)\bcamera:\s*{re.escape(camera_text)}\.?", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" .。;") or "Advance the planned visual action"


def _active_subject_labels(entries: Sequence[str], active_values: Sequence[str]) -> list[str]:
    labels: list[str] = []
    active = tuple(" ".join(value.casefold().split()) for value in active_values if value.strip())
    for entry in entries:
        match = re.match(r"<(Subject \d+)>", entry)
        if not match:
            continue
        if not active or any(_descriptions_overlap(value, entry) for value in active):
            labels.append(match.group(1))
    # If the model used a semantic alias we cannot match, retaining the full
    # canonical set is safer than silently making every subject disappear.
    return labels or [match.group(1) for entry in entries if (match := re.match(r"<(Subject \d+)>", entry))]


def _timeline(
    shot: ShotSpec,
    *,
    speaker_ids: Mapping[str, str],
    subject_lookup: Mapping[str, str],
) -> str:
    pieces = ["The action unfolds continuously from its opening state to its ending state without a reset."]
    opening = shot.opening_state or shot.continuity_in.action
    ending = shot.ending_state or shot.continuity_out.action
    if opening:
        pieces.append(f"At the opening, {_prompt_text(opening)}.")
    for beat in shot.visual_beats:
        beat_text = f"From {beat.start_seconds:.3f}s to {beat.end_seconds:.3f}s, {_prompt_text(beat.visual_action)}."
        if beat.state_change:
            beat_text += f" The visible state changes so that {_prompt_text(beat.state_change)}."
        if beat.camera:
            beat_text += f" The camera {_prompt_text(beat.camera)}."
        if beat.performance:
            beat_text += f" Performance and expression: {_prompt_text(beat.performance)}."
        if beat.spatial_anchor:
            beat_text += f" Screen-space anchor: {_prompt_text(beat.spatial_anchor)}."
        if beat.sound:
            beat_text += f" Synchronized sound: {_prompt_text(beat.sound)}."
        if beat.handoff:
            beat_text += f" Beat handoff: {_prompt_text(beat.handoff)}."
        for line in shot.dialogue:
            line_start = line.start_seconds if line.start_seconds is not None else 0.0
            if beat.start_seconds <= line_start < beat.end_seconds or (
                line_start == shot.duration_seconds == beat.end_seconds
            ):
                beat_text += " " + _dialogue_line(line, speaker_ids, subject_lookup)
        pieces.append(beat_text)
    if not shot.visual_beats:
        pieces.append(_dialogue_lines(shot.dialogue, speaker_ids, subject_lookup))
    elif not shot.dialogue:
        pieces.append("No spoken dialogue, narration, or voice-over occurs in this clip.")
    if ending:
        pieces.append(f"By the end, {_prompt_text(ending)}.")
    if shot.continuity_handoff:
        pieces.append(
            "The final moment preserves this continuity for the next clip: "
            f"{_prompt_text(shot.continuity_handoff)}. Hold the final readable pose for approximately "
            "0.5 to 1.0 seconds "
            "before any new action; do not replay the preceding action."
        )
    return " ".join(pieces)


def _dialogue_line(
    line: DialogueLine,
    speaker_ids: Mapping[str, str],
    subject_lookup: Mapping[str, str],
) -> str:
    speaker_id = speaker_ids[line.speaker]
    subject = subject_lookup.get(line.speaker)
    subject_prefix = f"<{subject}> " if subject else ""
    timing = _dialogue_timing(line)
    language = _language_name(line.language)
    mode_instruction = {
        "on_screen": "The speaker remains visible and the mouth movement synchronizes naturally to the line.",
        "off_screen": "The speaker is off screen; every visible character keeps their lips closed.",
        "voice_over": "The line is voice-over; every visible character keeps their lips closed.",
    }[line.mode]
    ending_instruction = (
        "When the line ends, the speaker's lips close and the facial expression settles into the beat's ending state."
        if line.mode == "on_screen"
        else "The visible performance continues without lip movement after the voice ends."
    )
    return (
        f"{subject_prefix}({speaker_id}) {line.speaker} speaks in a {line.delivery} manner{timing}: "
        f"{mode_instruction} <d>[{language}] {line.text}</d>. {ending_instruction}"
    )


def _dialogue_lines(
    lines: Sequence[DialogueLine],
    speaker_ids: Mapping[str, str],
    subject_lookup: Mapping[str, str],
) -> str:
    if not lines:
        return "No spoken dialogue, narration, or voice-over occurs in this clip."
    rendered: list[str] = []
    for line in lines:
        rendered.append(_dialogue_line(line, speaker_ids, subject_lookup))
    return " ".join(rendered)


def _language_name(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    aliases = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
        "english": "English",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
    }
    return aliases.get(normalized, value.strip() or "English")


def _dialogue_timing(line: DialogueLine) -> str:
    if line.start_seconds is not None and line.end_seconds is not None:
        return f" from {line.start_seconds:g}s to {line.end_seconds:g}s"
    if line.start_seconds is not None:
        return f" starting at {line.start_seconds:g}s"
    return ""


def _state_text(state: object) -> str:
    fields = (
        "characters",
        "wardrobe",
        "props",
        "fixed_landmarks",
        "character_positions",
        "exited_characters",
        "performance",
        "spatial_anchor",
        "handoff",
        "location",
        "lighting",
        "camera",
        "action",
        "audio",
    )
    values: list[str] = []
    for field in fields:
        value = getattr(state, field, None)
        if isinstance(value, list | tuple):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        if value:
            values.append(f"{field}={value}")
    return "; ".join(values)


def _soundscape(shot: ShotSpec, refs: Sequence[ContextReference]) -> str:
    value = sanitize_audio_prompt(shot.audio_prompt, has_dialogue=bool(shot.dialogue))
    if not value:
        value = "Natural synchronized ambience and physical action sounds continue without a hard seam."
    audio_refs = [reference for reference in refs if reference.kind == "audio"]
    if audio_refs:
        value += " " + " ".join(
            f"<{reference.label}> supplies the referenced audio characteristics." for reference in audio_refs
        )
    return value


def sanitize_audio_prompt(value: str, *, has_dialogue: bool) -> str:
    """Remove contradictory no-speech clauses from an editable audio field."""

    text = value.strip()
    if not has_dialogue:
        return text
    text = re.sub(
        r"(?:no|without)\s+(?:spoken\s+)?(?:dialogue|narration|voice[- ]over)\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s{2,}", " ", text).strip(" .;,")
    return text or "Natural ambience and synchronized physical Foley continue beneath the dialogue track."
