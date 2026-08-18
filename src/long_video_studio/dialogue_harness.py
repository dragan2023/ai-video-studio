"""Deterministic dialogue checks for planner output.

The language model is useful for writing a line, but it is not a reliable
timeline or identity registry.  This module is the small, provider-neutral
boundary that turns model dialogue into a canonical, timed representation.
It deliberately does not render prompts or call a model; callers can use the
returned ``DialogueLine`` objects from either FL2VA or Ref2VA adapters.

The helpers are intentionally compatible with old projects.  When no
``WorldBible`` is available, speakers are registered from first appearance and
receive deterministic synthetic subject/speaker IDs.  Once a world bible is
available, aliases must resolve to exactly one ``SubjectCard`` and unknown or
ambiguous names fail closed.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .domain import DialogueLine, SubjectCard, WorldBible

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")
_PAUSE_RE = re.compile(r"[,，、;；:：]")
_LONG_PAUSE_RE = re.compile(r"[…⋯—–-]")
_SENTENCE_PAUSE_RE = re.compile(r"[.!?。！？]")
_NARRATOR_LABELS = {"narrator", "voice over", "voice-over", "旁白", "解说", "旁白者"}
_H3_SPEAKER_ID_RE = re.compile(r"^S[1-9]\d*$")

DEFAULT_SPEECH_MARGIN_SECONDS = 0.35
DEFAULT_LEAD_IN_SECONDS = 0.15
DEFAULT_INTERLINE_GAP_SECONDS = 0.12
DEFAULT_TAIL_OUT_SECONDS = 0.35
_TIMING_TOLERANCE = 1e-3


class DialogueHarnessError(ValueError):
    """A fail-closed dialogue error with machine-readable details."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.message = message
        self.details = {"code": code, **details}
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class SpeakerBinding:
    """Canonical identity and voice label for one speaking subject."""

    subject_id: str
    speaker_id: str | None
    label: str
    aliases: tuple[str, ...] = ()
    card: SubjectCard | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Return all accepted names, including the canonical label."""

        return tuple(dict.fromkeys((self.label, *self.aliases, self.subject_id)))


@dataclass(frozen=True)
class DialoguePreparation:
    """Canonical world bible, speaker roster, and timed dialogue lines."""

    world_bible: WorldBible | None
    roster: tuple[SpeakerBinding, ...]
    lines: tuple[DialogueLine, ...]


@dataclass(frozen=True)
class DialogueSchedule:
    """Timed lines returned by the planner-facing compatibility wrapper."""

    lines: tuple[DialogueLine, ...]
    shot_index: int | None = None


def estimate_speech_seconds(text: str, language: str = "Chinese") -> float:
    """Estimate natural spoken duration for ``text``.

    Mandarin/CJK characters and Latin words are counted separately so mixed
    lines such as ``"H100 十万美金"`` do not get treated as one opaque token.
    Punctuation contributes a small pause allowance.  The estimate is a
    conservative planning floor, not a claim about a particular actor's voice
    speed; callers should add a margin before accepting a hard window.
    """

    value = " ".join(str(text or "").split()).strip()
    if not value:
        return 0.0
    cjk_count = len(_CJK_RE.findall(value))
    latin_count = len(_LATIN_WORD_RE.findall(value))
    digit_count = len(_DIGIT_RE.findall(value))
    normalized_language = _normalise_language(language)

    if normalized_language in {"zh", "ja", "ko"}:
        character_rate = 4.2 if normalized_language == "zh" else 4.1
        word_rate = 2.7
    elif normalized_language == "en":
        character_rate = 4.2
        word_rate = 2.6
    else:
        character_rate = 4.2
        word_rate = 2.6

    # Digits and model names are usually articulated more slowly than ordinary
    # prose.  Count a small extra allowance, while avoiding double-counting
    # digits embedded in a Latin token (for example ``H100``).
    digit_allowance = 0.12 * max(0, digit_count - latin_count)
    spoken_units = (cjk_count / character_rate) + (latin_count / word_rate) + digit_allowance
    if spoken_units <= 0:
        spoken_units = max(1, len(re.sub(r"\s+", "", value))) / 10.0

    pause_seconds = (
        0.07 * len(_PAUSE_RE.findall(value))
        + 0.16 * len(_LONG_PAUSE_RE.findall(value))
        + 0.13 * len(_SENTENCE_PAUSE_RE.findall(value))
    )
    return round(max(0.45, spoken_units + pause_seconds), 3)


def minimum_dialogue_window(
    line: DialogueLine,
    *,
    speech_margin_seconds: float = DEFAULT_SPEECH_MARGIN_SECONDS,
) -> float:
    """Return the minimum safe interval for a line, including a small margin."""

    if speech_margin_seconds < 0:
        raise ValueError("speech_margin_seconds must be non-negative")
    return round(estimate_speech_seconds(line.text, line.language) + speech_margin_seconds, 3)


def build_speaker_roster(
    world_bible: WorldBible | None = None,
    lines: Sequence[DialogueLine] = (),
) -> tuple[SpeakerBinding, ...]:
    """Build a deterministic canonical roster and reject alias collisions.

    Existing ``SubjectCard.speaker_id`` values are preserved for subjects that
    actually speak. Missing IDs are assigned by first vocal-event order, which
    follows H3's S1/S2 contract; silent visual subjects consume no S ID. With
    no character cards, old callers get a
    synthetic card per first-seen speaker, which keeps the helper useful for
    legacy projects while still producing stable bindings.
    """

    cards = list(world_bible.subjects) if world_bible else []
    if not cards:
        cards = _synthetic_cards(lines)
    bindings: list[SpeakerBinding] = []
    seen_subject_ids: dict[str, str] = {}
    seen_speaker_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    card_by_subject: dict[str, SubjectCard] = {}
    card_by_name: dict[str, SubjectCard] = {}
    card_by_explicit_speaker: dict[str, SubjectCard] = {}
    for index, card in enumerate(cards, start=1):
        subject_id = _required_identifier(card.subject_id, "subject_id", index)
        _check_unique(seen_subject_ids, _normalise_key(subject_id), subject_id, "subject_id")
        if card.speaker_id and _H3_SPEAKER_ID_RE.fullmatch(card.speaker_id):
            _check_unique(seen_speaker_ids, _normalise_key(card.speaker_id), subject_id, "speaker_id")
        aliases = tuple(
            alias
            for alias in dict.fromkeys(str(value).strip() for value in card.aliases)
            if alias and _normalise_key(alias) != _normalise_key(card.label)
        )
        binding = SpeakerBinding(subject_id, None, card.label.strip(), aliases, card)
        if not binding.label:
            raise DialogueHarnessError(
                "invalid_speaker_card",
                "speaker card label is empty",
                subject_id=subject_id,
            )
        for name in binding.names:
            _check_unique(seen_names, _normalise_key(name), subject_id, "speaker_alias")
            card_by_name[_normalise_key(name)] = card
        card_by_subject[subject_id] = card
        if card.speaker_id and _H3_SPEAKER_ID_RE.fullmatch(card.speaker_id):
            card_by_explicit_speaker[card.speaker_id] = card
        bindings.append(binding)

    vocal_subject_ids: list[str] = []
    for line in lines:
        if line.mode == "voice_over" and _normalise_key(line.speaker) in _NARRATOR_LABELS:
            continue
        candidates = [
            card_by_name.get(_normalise_key(line.speaker)),
            card_by_subject.get(line.subject_id) if line.subject_id else None,
            card_by_explicit_speaker.get(line.speaker_id) if line.speaker_id else None,
        ]
        matched_ids = {card.subject_id for card in candidates if card is not None}
        if len(matched_ids) == 1:
            subject_id = next(iter(matched_ids))
            if subject_id not in vocal_subject_ids:
                vocal_subject_ids.append(subject_id)

    explicit_vocal_ids = {
        card.speaker_id
        for card in cards
        if card.subject_id in vocal_subject_ids and card.speaker_id and _H3_SPEAKER_ID_RE.fullmatch(card.speaker_id)
    }
    used_speaker_keys = {_normalise_key(value) for value in explicit_vocal_ids}
    assigned: dict[str, str] = {}
    next_speaker_number = 1
    for subject_id in vocal_subject_ids:
        card = card_by_subject[subject_id]
        speaker_id = card.speaker_id if card.speaker_id and _H3_SPEAKER_ID_RE.fullmatch(card.speaker_id) else None
        if not speaker_id:
            while _normalise_key(f"S{next_speaker_number}") in used_speaker_keys:
                next_speaker_number += 1
            speaker_id = f"S{next_speaker_number}"
            used_speaker_keys.add(_normalise_key(speaker_id))
            next_speaker_number += 1
        assigned[subject_id] = speaker_id
    return tuple(
        SpeakerBinding(
            binding.subject_id,
            assigned.get(binding.subject_id),
            binding.label,
            binding.aliases,
            binding.card,
        )
        for binding in bindings
    )


def canonicalize_world_bible(
    world_bible: WorldBible | None,
    lines: Sequence[DialogueLine] = (),
) -> WorldBible | None:
    """Return a copy with deterministic ``speaker_id`` values on subject cards."""

    if world_bible is None:
        return None
    # Rebuild H3 S IDs from the complete vocal-event order. Arbitrary Studio
    # voice keys and stale S ordering never leak into a model-facing prompt.
    base_world = world_bible.model_copy(
        update={"subjects": [card.model_copy(update={"speaker_id": None}) for card in world_bible.subjects]}
    )
    roster = build_speaker_roster(base_world, lines)
    by_subject = {binding.subject_id: binding.speaker_id for binding in roster if binding.speaker_id is not None}
    subjects = [
        card.model_copy(update={"speaker_id": by_subject.get(card.subject_id)}) for card in world_bible.subjects
    ]
    return world_bible.model_copy(update={"subjects": subjects})


def canonicalize_dialogue(
    lines: Sequence[DialogueLine],
    world_bible: WorldBible | None = None,
    *,
    roster: Sequence[SpeakerBinding] | None = None,
) -> list[DialogueLine]:
    """Resolve aliases and optional IDs to canonical ``DialogueLine`` copies."""

    bindings = tuple(roster) if roster is not None else build_speaker_roster(world_bible, lines)
    if not bindings:
        return [line.model_copy(deep=True) for line in lines]
    by_name = {_normalise_key(name): binding for binding in bindings for name in binding.names}
    by_subject = {binding.subject_id: binding for binding in bindings}
    by_speaker = {binding.speaker_id: binding for binding in bindings if binding.speaker_id is not None}
    if world_bible is not None:
        for card in world_bible.subjects:
            if card.speaker_id and _H3_SPEAKER_ID_RE.fullmatch(card.speaker_id):
                owner = by_subject.get(card.subject_id)
                if owner is not None:
                    by_speaker.setdefault(card.speaker_id, owner)
    result: list[DialogueLine] = []
    for index, line in enumerate(lines):
        by_name_match = by_name.get(_normalise_key(line.speaker))
        subject_match = by_subject.get(line.subject_id) if line.subject_id else None
        speaker_match = by_speaker.get(line.speaker_id) if line.speaker_id else None
        if line.subject_id and subject_match is None:
            raise DialogueHarnessError(
                "speaker_binding_conflict",
                f"dialogue line {index + 1} names an unknown subject_id",
                line_index=index,
                speaker=line.speaker,
                subject_id=line.subject_id,
                speaker_id=line.speaker_id,
            )
        matches = [match for match in (by_name_match, subject_match, speaker_match) if match is not None]
        if not matches:
            if line.mode == "voice_over" and _normalise_key(line.speaker) in _NARRATOR_LABELS:
                result.append(line.model_copy(update={"subject_id": None, "speaker_id": line.speaker_id or "VO1"}))
                continue
            raise DialogueHarnessError(
                "unknown_speaker",
                f"dialogue line {index + 1} does not resolve to a canonical subject",
                line_index=index,
                speaker=line.speaker,
                subject_id=line.subject_id,
                speaker_id=line.speaker_id,
            )
        binding = matches[0]
        if any(match.subject_id != binding.subject_id for match in matches[1:]):
            raise DialogueHarnessError(
                "speaker_binding_conflict",
                f"dialogue line {index + 1} names multiple subjects",
                line_index=index,
                speaker=line.speaker,
                subject_id=line.subject_id,
                speaker_id=line.speaker_id,
                matched_subject_ids=[match.subject_id for match in matches],
            )
        if binding.speaker_id is None:
            raise DialogueHarnessError(
                "speaker_binding_conflict",
                f"dialogue line {index + 1} resolved to a subject without a vocal-event speaker_id",
                line_index=index,
                subject_id=binding.subject_id,
            )
        result.append(
            line.model_copy(
                update={
                    "speaker": binding.label,
                    "subject_id": binding.subject_id,
                    "speaker_id": binding.speaker_id,
                }
            )
        )
    return result


def validate_active_speakers(
    lines: Sequence[DialogueLine],
    world_bible: WorldBible | None,
    active_characters: Sequence[str],
    *,
    active_subject_ids: Sequence[str] = (),
    shot_index: int | None = None,
) -> None:
    """Ensure an on-screen voice belongs to a character present in the shot."""

    on_screen = [line for line in lines if line.mode == "on_screen"]
    if not on_screen:
        return
    canonical_active_ids = {value for value in active_subject_ids if value.strip()}
    if canonical_active_ids:
        for line in on_screen:
            if line.subject_id in canonical_active_ids:
                continue
            raise DialogueHarnessError(
                "speaker_not_active",
                f"on-screen speaker {line.speaker!r} is not in continuity_in.active_subject_ids",
                shot_index=shot_index,
                subject_id=line.subject_id,
                active_subject_ids=list(active_subject_ids),
            )
        return
    active = [" ".join(value.casefold().split()) for value in active_characters if value.strip()]
    if not active or not world_bible or not world_bible.subjects:
        raise DialogueHarnessError(
            "missing_active_cast",
            "on-screen dialogue requires continuity_in.active_subject_ids or a named active character roster",
            shot_index=shot_index,
        )
    cards = {card.subject_id: card for card in world_bible.subjects}
    for line in on_screen:
        if not line.subject_id:
            continue
        card = cards.get(line.subject_id)
        names = [line.speaker, line.subject_id, *(card.aliases if card else [])]
        if any(
            name.casefold() in value or value in name.casefold() for name in names if name.strip() for value in active
        ):
            continue
        raise DialogueHarnessError(
            "speaker_not_active",
            f"on-screen speaker {line.speaker!r} is not present in continuity_in.characters",
            shot_index=shot_index,
            subject_id=line.subject_id,
            active_characters=list(active_characters),
        )


def normalize_dialogue_timing(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    *,
    world_bible: WorldBible | None = None,
    roster: Sequence[SpeakerBinding] | None = None,
    lead_in_seconds: float = DEFAULT_LEAD_IN_SECONDS,
    interline_gap_seconds: float = DEFAULT_INTERLINE_GAP_SECONDS,
    tail_out_seconds: float = DEFAULT_TAIL_OUT_SECONDS,
    speech_margin_seconds: float = DEFAULT_SPEECH_MARGIN_SECONDS,
    repair_explicit: bool = False,
) -> list[DialogueLine]:
    """Canonicalize and deterministically fill missing line timings.

    Explicit windows are never silently stretched.  A window shorter than the
    language-aware speech floor, an overlap, or an out-of-shot endpoint raises
    :class:`DialogueHarnessError` with a stable error code.
    """

    _validate_schedule_options(duration_seconds, lead_in_seconds, interline_gap_seconds, tail_out_seconds)
    canonical = canonicalize_dialogue(lines, world_bible, roster=roster)
    if repair_explicit:
        return _repair_lines(
            canonical,
            duration_seconds,
            lead_in_seconds=lead_in_seconds,
            interline_gap_seconds=interline_gap_seconds,
            tail_out_seconds=tail_out_seconds,
            speech_margin_seconds=speech_margin_seconds,
        )
    return _schedule_lines(
        canonical,
        duration_seconds,
        fill_missing=True,
        lead_in_seconds=lead_in_seconds,
        interline_gap_seconds=interline_gap_seconds,
        tail_out_seconds=tail_out_seconds,
        speech_margin_seconds=speech_margin_seconds,
    )


def validate_dialogue_timing(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    *,
    world_bible: WorldBible | None = None,
    roster: Sequence[SpeakerBinding] | None = None,
    tail_out_seconds: float = DEFAULT_TAIL_OUT_SECONDS,
    speech_margin_seconds: float = DEFAULT_SPEECH_MARGIN_SECONDS,
) -> tuple[DialogueLine, ...]:
    """Validate an already-timed dialogue list without filling omissions."""

    _validate_schedule_options(duration_seconds, 0.0, 0.0, tail_out_seconds)
    canonical = canonicalize_dialogue(lines, world_bible, roster=roster)
    checked = _schedule_lines(
        canonical,
        duration_seconds,
        fill_missing=False,
        lead_in_seconds=0.0,
        interline_gap_seconds=0.0,
        tail_out_seconds=tail_out_seconds,
        speech_margin_seconds=speech_margin_seconds,
    )
    return tuple(checked)


def prepare_dialogue(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    world_bible: WorldBible | None = None,
    **timing_options: float,
) -> DialoguePreparation:
    """Convenience API combining world-bible, speaker, and timing normalization."""

    roster = build_speaker_roster(world_bible, lines)
    canonical_bible = canonicalize_world_bible(world_bible, lines)
    timed = normalize_dialogue_timing(
        lines,
        duration_seconds,
        world_bible=canonical_bible,
        roster=roster,
        repair_explicit=True,
        **timing_options,
    )
    return DialoguePreparation(canonical_bible, roster, tuple(timed))


def bind_dialogue_lines(
    lines: Sequence[DialogueLine],
    world_bible: WorldBible | None,
    *,
    shot_index: int | None = None,
) -> list[DialogueLine]:
    """Planner-facing alias for :func:`canonicalize_dialogue`.

    ``shot_index`` is included only in structured failures so a UI can point
    at the offending storyboard shot without changing the domain model.
    """

    try:
        return canonicalize_dialogue(lines, world_bible)
    except DialogueHarnessError as error:
        raise _with_shot_index(error, shot_index) from error


def schedule_dialogue_lines(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    *,
    shot_index: int | None = None,
) -> DialogueSchedule:
    """Planner-facing alias that fills timings and returns a small envelope."""

    try:
        scheduled = normalize_dialogue_timing(lines, duration_seconds, repair_explicit=True)
    except DialogueHarnessError as error:
        raise _with_shot_index(error, shot_index) from error
    return DialogueSchedule(tuple(scheduled), shot_index=shot_index)


def _repair_lines(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    *,
    lead_in_seconds: float,
    interline_gap_seconds: float,
    tail_out_seconds: float,
    speech_margin_seconds: float,
) -> list[DialogueLine]:
    """Reflow fixable LLM timing mistakes without changing spoken content."""

    if not lines:
        return []
    required = [minimum_dialogue_window(line, speech_margin_seconds=speech_margin_seconds) for line in lines]
    available = duration_seconds - lead_in_seconds - tail_out_seconds
    required_total = sum(required) + interline_gap_seconds * max(0, len(lines) - 1)
    if required_total > available + _TIMING_TOLERANCE:
        raise DialogueHarnessError(
            "dialogue_schedule_overflow",
            "dialogue cannot fit this shot after deterministic reflow; Director must redistribute the cut",
            required_seconds=round(required_total, 3),
            available_seconds=round(available, 3),
            line_count=len(lines),
        )

    preferred: list[float] = []
    desired_starts: list[float | None] = []
    for line, minimum in zip(lines, required, strict=True):
        explicit_window = (
            line.end_seconds - line.start_seconds
            if line.start_seconds is not None and line.end_seconds is not None
            else minimum
        )
        preferred.append(max(minimum, explicit_window))
        if line.start_seconds is not None:
            desired_starts.append(line.start_seconds)
        elif line.end_seconds is not None:
            desired_starts.append(max(lead_in_seconds, line.end_seconds - minimum))
        else:
            desired_starts.append(None)

    preferred_total = sum(preferred) + interline_gap_seconds * max(0, len(lines) - 1)
    windows = preferred if preferred_total <= available + _TIMING_TOLERANCE else required

    starts: list[float] = []
    ends: list[float] = []
    cursor = lead_in_seconds
    for desired, window in zip(desired_starts, windows, strict=True):
        start = max(cursor, desired if desired is not None else cursor)
        end = start + window
        starts.append(start)
        ends.append(end)
        cursor = end + interline_gap_seconds

    maximum_end = duration_seconds - tail_out_seconds
    overflow = ends[-1] - maximum_end
    if overflow > _TIMING_TOLERANCE:
        shift = min(overflow, max(0.0, starts[0] - lead_in_seconds))
        if shift > 0:
            starts = [value - shift for value in starts]
            ends = [value - shift for value in ends]
            overflow -= shift
        if overflow > _TIMING_TOLERANCE:
            # Explicit timestamps contained idle gaps or oversized speech
            # windows. Compact them from the lead-in using only safe minimums.
            windows = required
            starts = []
            ends = []
            cursor = lead_in_seconds
            for window in windows:
                starts.append(cursor)
                cursor += window
                ends.append(cursor)
                cursor += interline_gap_seconds

    if ends[-1] > maximum_end + _TIMING_TOLERANCE:
        raise DialogueHarnessError(
            "dialogue_schedule_overflow",
            "dialogue reflow could not preserve the requested tail room",
            required_seconds=round(required_total, 3),
            available_seconds=round(available, 3),
            final_end_seconds=round(ends[-1], 3),
        )
    return [
        line.model_copy(
            update={
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
            }
        )
        for line, start, end in zip(lines, starts, ends, strict=True)
    ]


def _schedule_lines(
    lines: Sequence[DialogueLine],
    duration_seconds: float,
    *,
    fill_missing: bool,
    lead_in_seconds: float,
    interline_gap_seconds: float,
    tail_out_seconds: float,
    speech_margin_seconds: float,
) -> list[DialogueLine]:
    scheduled: list[DialogueLine] = []
    cursor = lead_in_seconds
    for index, line in enumerate(lines):
        required = minimum_dialogue_window(line, speech_margin_seconds=speech_margin_seconds)
        start = line.start_seconds
        end = line.end_seconds
        if not fill_missing and (start is None or end is None):
            raise DialogueHarnessError(
                "missing_timing",
                f"dialogue line {index + 1} has no explicit start/end window",
                line_index=index,
                speaker=line.speaker,
            )
        if start is None:
            start = cursor if end is None else end - required
            if start < cursor - _TIMING_TOLERANCE:
                raise DialogueHarnessError(
                    "dialogue_overlap",
                    f"dialogue line {index + 1} cannot fit before its explicit end",
                    line_index=index,
                    start_seconds=start,
                    previous_end_seconds=cursor,
                )
        if end is None:
            end = start + required
        assert start is not None and end is not None
        if start < cursor - _TIMING_TOLERANCE:
            raise DialogueHarnessError(
                "dialogue_overlap",
                f"dialogue line {index + 1} overlaps the previous line",
                line_index=index,
                start_seconds=start,
                previous_end_seconds=cursor,
            )
        actual_window = end - start
        if actual_window + _TIMING_TOLERANCE < required:
            raise DialogueHarnessError(
                "dialogue_window_too_short",
                f"dialogue line {index + 1} has only {actual_window:.3f}s for about {required:.3f}s of speech",
                line_index=index,
                speaker=line.speaker,
                available_seconds=round(actual_window, 3),
                required_seconds=required,
                text=line.text,
            )
        if start < -_TIMING_TOLERANCE or end > duration_seconds + _TIMING_TOLERANCE:
            raise DialogueHarnessError(
                "dialogue_overflow",
                f"dialogue line {index + 1} falls outside the shot duration",
                line_index=index,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=duration_seconds,
            )
        if index == len(lines) - 1 and end > duration_seconds - tail_out_seconds + _TIMING_TOLERANCE:
            raise DialogueHarnessError(
                "dialogue_tail_overflow",
                f"dialogue line {index + 1} leaves less than the requested tail room",
                line_index=index,
                end_seconds=end,
                duration_seconds=duration_seconds,
                tail_out_seconds=tail_out_seconds,
            )
        scheduled.append(
            line.model_copy(
                update={
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                }
            )
        )
        cursor = end + interline_gap_seconds
    return scheduled


def _with_shot_index(error: DialogueHarnessError, shot_index: int | None) -> DialogueHarnessError:
    if shot_index is None or "shot_index" in error.details:
        return error
    details = dict(error.details)
    details["shot_index"] = shot_index
    return DialogueHarnessError(
        error.code, error.message, **{key: value for key, value in details.items() if key != "code"}
    )


def _synthetic_cards(lines: Sequence[DialogueLine]) -> list[SubjectCard]:
    cards: list[SubjectCard] = []
    seen: set[str] = set()
    for line in lines:
        key = _normalise_key(line.subject_id or line.speaker)
        if not key or key in seen:
            continue
        seen.add(key)
        cards.append(
            SubjectCard(
                subject_id=line.subject_id or _slug_identifier(line.speaker),
                label=line.speaker,
                aliases=[],
                speaker_id=line.speaker_id,
            )
        )
    return cards


def _check_unique(seen: dict[str, str], key: str, subject_id: str, field: str) -> None:
    if not key:
        raise DialogueHarnessError("invalid_speaker_card", f"{field} is empty", subject_id=subject_id)
    previous = seen.get(key)
    if previous is not None and previous != subject_id:
        raise DialogueHarnessError(
            "speaker_alias_conflict",
            f"{field} maps to more than one subject",
            key=key,
            first_subject_id=previous,
            second_subject_id=subject_id,
        )
    seen[key] = subject_id


def _required_identifier(value: str, field: str, index: int) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise DialogueHarnessError("invalid_speaker_card", f"{field} is empty", index=index)
    return normalized


def _normalise_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.casefold().split())


def _normalise_language(value: str) -> str:
    normalized = _normalise_key(value).replace("_", "-")
    if normalized in {"zh", "zh-cn", "zh-hans", "chinese", "mandarin"}:
        return "zh"
    if normalized in {"ja", "ja-jp", "japanese"}:
        return "ja"
    if normalized in {"ko", "ko-kr", "korean"}:
        return "ko"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return normalized


def _slug_identifier(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", _normalise_key(value))
    if words:
        return "subject_" + "_".join(words)
    digest = hashlib.sha1(_normalise_key(value).encode("utf-8")).hexdigest()[:12]
    return "subject_" + digest


def _validate_schedule_options(
    duration_seconds: float,
    lead_in_seconds: float,
    interline_gap_seconds: float,
    tail_out_seconds: float,
) -> None:
    values = {
        "duration_seconds": duration_seconds,
        "lead_in_seconds": lead_in_seconds,
        "interline_gap_seconds": interline_gap_seconds,
        "tail_out_seconds": tail_out_seconds,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("dialogue timing options must be finite")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if any(float(value) < 0 for value in (lead_in_seconds, interline_gap_seconds, tail_out_seconds)):
        raise ValueError("dialogue timing options must be non-negative")
