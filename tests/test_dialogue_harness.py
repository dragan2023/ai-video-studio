from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from long_video_studio.dialogue_harness import (
    DialogueHarnessError,
    bind_dialogue_lines,
    build_speaker_roster,
    canonicalize_dialogue,
    canonicalize_world_bible,
    estimate_speech_seconds,
    normalize_dialogue_timing,
    prepare_dialogue,
    schedule_dialogue_lines,
    validate_dialogue_timing,
)
from long_video_studio.domain import (
    ContinuityState,
    DialogueLine,
    ProjectBrief,
    ShotSpec,
    StoryboardBeat,
    SubjectCard,
    WorldBible,
)
from long_video_studio.h3_prompt import render_h3_prompt
from long_video_studio.planner import DirectorPlan, PlannerOutput, PlannerService, ShotBlueprint
from long_video_studio.repository import StudioRepository
from long_video_studio.style_registry import style_prompt


def _world() -> WorldBible:
    return WorldBible(
        logline="A market confrontation",
        visual_style="cinematic realism",
        subjects=[
            SubjectCard(
                subject_id="liang",
                label="Liang Wenfeng",
                aliases=["梁文锋", "the buyer"],
                visual_identity="lean man in a charcoal coat",
                speaker_id="speaker_liang",
            ),
            SubjectCard(
                subject_id="huang",
                label="Huang Renxun",
                aliases=["黄仁勋", "the vendor"],
                visual_identity="compact man in a leather jacket",
            ),
        ],
    )


def test_estimate_speech_seconds_is_language_aware_and_accounts_for_pauses():
    chinese = estimate_speech_seconds("这是一句自然的中文对白。", "zh-CN")
    english = estimate_speech_seconds("This is a natural spoken line.", "English")
    mixed = estimate_speech_seconds("H100 十万美金——保熟。", "Chinese")

    assert chinese > 1
    assert english > 1
    assert mixed > estimate_speech_seconds("H100 十万美金。", "Chinese")
    assert estimate_speech_seconds("", "Chinese") == 0


def test_canonicalize_dialogue_resolves_alias_and_assigns_missing_card_id():
    lines = [DialogueLine(speaker="梁文锋", text="给我一片。")]

    canonical = canonicalize_dialogue(lines, _world())
    world = canonicalize_world_bible(_world(), lines)

    assert canonical[0].speaker == "Liang Wenfeng"
    assert canonical[0].subject_id == "liang"
    assert canonical[0].speaker_id == "S1"
    assert world is not None
    assert world.subjects[1].speaker_id is None


def test_missing_speaker_id_skips_an_existing_reserved_id():
    world = _world().model_copy(
        update={
            "subjects": [
                _world().subjects[0].model_copy(update={"speaker_id": "S2"}),
                _world().subjects[1].model_copy(update={"speaker_id": None}),
            ]
        }
    )

    canonical = canonicalize_world_bible(
        world,
        [DialogueLine(speaker="黄仁勋", text="十万美金。")],
    )

    assert canonical is not None
    assert [card.speaker_id for card in canonical.subjects] == [None, "S1"]


def test_h3_speaker_ids_follow_first_vocal_event_and_skip_silent_subjects():
    world = _world().model_copy(
        update={"subjects": [card.model_copy(update={"speaker_id": None}) for card in _world().subjects]}
    )
    lines = [
        DialogueLine(speaker="黄仁勋", text="先说。"),
        DialogueLine(speaker="梁文锋", text="后说。"),
    ]

    canonical = canonicalize_world_bible(world, lines)

    assert canonical is not None
    assert [card.speaker_id for card in canonical.subjects] == ["S2", "S1"]


def test_explicit_voice_over_narrator_does_not_steal_a_character_voice():
    line = DialogueLine(speaker="旁白", text="夜幕降临。", mode="voice_over")

    canonical = canonicalize_dialogue([line], _world())

    assert canonical[0].subject_id is None
    assert canonical[0].speaker_id == "VO1"


def test_canonicalize_dialogue_accepts_explicit_ids_when_display_name_is_alias():
    line = DialogueLine(
        speaker="vendor",
        subject_id="huang",
        speaker_id="S1",
        text="十万美金。",
    )

    canonical = canonicalize_dialogue([line], _world())

    assert canonical[0].speaker == "Huang Renxun"
    assert canonical[0].subject_id == "huang"
    assert canonical[0].speaker_id == "S1"


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (DialogueLine(speaker="unknown person", text="Hello."), "unknown_speaker"),
        (
            DialogueLine(speaker="梁文锋", subject_id="huang", text="Hello."),
            "speaker_binding_conflict",
        ),
    ],
)
def test_canonicalize_dialogue_fails_closed_with_structured_errors(line: DialogueLine, code: str):
    with pytest.raises(DialogueHarnessError) as raised:
        canonicalize_dialogue([line], _world())

    assert raised.value.code == code
    assert raised.value.details["code"] == code
    assert raised.value.details["line_index"] == 0


def test_canonical_speaker_id_pointing_at_another_subject_is_rejected():
    world = _world().model_copy(
        update={
            "subjects": [
                _world().subjects[0].model_copy(update={"speaker_id": "S1"}),
                _world().subjects[1].model_copy(update={"speaker_id": "S2"}),
            ]
        }
    )
    line = DialogueLine(speaker="Liang Wenfeng", subject_id="liang", speaker_id="S2", text="Hello.")

    with pytest.raises(DialogueHarnessError) as raised:
        canonicalize_dialogue([line], world)

    assert raised.value.code == "speaker_binding_conflict"


def test_duplicate_aliases_are_rejected_before_model_generation():
    world = _world().model_copy(
        update={
            "subjects": [
                *_world().subjects,
                SubjectCard(subject_id="other", label="Other", aliases=["梁文锋"]),
            ]
        }
    )

    with pytest.raises(DialogueHarnessError) as raised:
        build_speaker_roster(world)

    assert raised.value.code == "speaker_alias_conflict"
    assert raised.value.details["second_subject_id"] == "other"


def test_normalize_dialogue_timing_fills_missing_values_deterministically():
    lines = [
        DialogueLine(speaker="梁文锋", text="你好。"),
        DialogueLine(speaker="黄仁勋", text="十万美金。", start_seconds=2.0),
    ]

    normalized = normalize_dialogue_timing(lines, 15, world_bible=_world())

    assert normalized[0].speaker == "Liang Wenfeng"
    assert normalized[0].start_seconds == 0.15
    assert normalized[0].end_seconds is not None
    assert normalized[1].speaker == "Huang Renxun"
    assert normalized[1].start_seconds == 2.0
    assert normalized[1].end_seconds is not None
    assert normalized[1].end_seconds > normalized[1].start_seconds


def test_normalize_dialogue_timing_fills_an_explicit_end_without_moving_it():
    line = DialogueLine(speaker="梁文锋", text="你好。", end_seconds=2.0)

    normalized = normalize_dialogue_timing([line], 15, world_bible=_world())

    assert normalized[0].end_seconds == 2.0
    assert normalized[0].start_seconds is not None
    assert normalized[0].start_seconds < 2.0


def test_short_explicit_window_is_rejected_with_required_speech_budget():
    line = DialogueLine(
        speaker="Liang Wenfeng",
        text=(
            "你这 H100，十万美金一片，算力虚标，还带后门。"
            "我这 S5000 国产的，算力实打实，价格只要你零头。关键是——它保熟。"
        ),
        start_seconds=0.9,
        end_seconds=5.4,
    )

    with pytest.raises(DialogueHarnessError) as raised:
        normalize_dialogue_timing([line], 15, world_bible=_world())

    assert raised.value.code == "dialogue_window_too_short"
    assert raised.value.details["available_seconds"] == 4.5
    assert raised.value.details["required_seconds"] > 4.5


def test_explicit_overlap_is_rejected():
    lines = [
        DialogueLine(speaker="梁文锋", text="你好。", start_seconds=1, end_seconds=3),
        DialogueLine(speaker="黄仁勋", text="十万美金。", start_seconds=2.5, end_seconds=4.5),
    ]

    with pytest.raises(DialogueHarnessError) as raised:
        normalize_dialogue_timing(lines, 15, world_bible=_world())

    assert raised.value.code == "dialogue_overlap"
    assert raised.value.details["line_index"] == 1


def test_explicit_overflow_is_rejected():
    line = DialogueLine(
        speaker="梁文锋",
        text="你好。",
        start_seconds=14,
        end_seconds=16,
    )

    with pytest.raises(DialogueHarnessError) as raised:
        validate_dialogue_timing([line], 15, world_bible=_world())

    assert raised.value.code == "dialogue_overflow"
    assert raised.value.details["duration_seconds"] == 15


def test_final_line_reserves_a_short_natural_tail():
    line = DialogueLine(
        speaker="梁文锋",
        text="成交。",
        start_seconds=13,
        end_seconds=15,
    )

    with pytest.raises(DialogueHarnessError) as raised:
        validate_dialogue_timing([line], 15, world_bible=_world())

    assert raised.value.code == "dialogue_tail_overflow"


def test_planner_wrapper_repairs_tail_overflow_and_unnecessary_slack():
    lines = [
        DialogueLine(speaker="梁文锋", text="第一句。", start_seconds=0, end_seconds=5),
        DialogueLine(speaker="梁文锋", text="第二句。", start_seconds=5, end_seconds=10),
        DialogueLine(speaker="梁文锋", text="第三句。", start_seconds=10, end_seconds=15),
    ]

    repaired = schedule_dialogue_lines(lines, 15).lines

    assert [line.text for line in repaired] == [line.text for line in lines]
    assert repaired[-1].end_seconds is not None
    assert repaired[-1].end_seconds <= 14.65
    assert repaired[1].start_seconds is not None
    assert repaired[0].end_seconds is not None
    assert repaired[1].start_seconds >= repaired[0].end_seconds


def test_planner_wrapper_escalates_only_intrinsically_overfull_dialogue():
    lines = [
        DialogueLine(speaker="梁文锋", text="这是一个绝对不能被截断的非常长的句子。" * 4),
        DialogueLine(speaker="梁文锋", text="这是第二个同样绝对不能被截断的非常长的句子。" * 4),
    ]

    with pytest.raises(DialogueHarnessError) as raised:
        schedule_dialogue_lines(lines, 15)

    assert raised.value.code == "dialogue_schedule_overflow"
    assert raised.value.details["required_seconds"] > raised.value.details["available_seconds"]


def test_validate_requires_explicit_timing_but_prepare_combines_all_steps():
    line = DialogueLine(speaker="梁文锋", text="你好。")
    with pytest.raises(DialogueHarnessError) as raised:
        validate_dialogue_timing([line], 15, world_bible=_world())
    assert raised.value.code == "missing_timing"

    prepared = prepare_dialogue([line], 15, _world())
    assert prepared.world_bible is not None
    assert prepared.roster[0].subject_id == "liang"
    assert prepared.lines[0].start_seconds is not None
    assert prepared.lines[0].end_seconds is not None


def test_legacy_no_world_bible_callers_get_stable_synthetic_bindings():
    lines = [DialogueLine(speaker="Narrator", text="The market wakes.")]

    prepared = prepare_dialogue(lines, 15)

    assert prepared.world_bible is None
    assert prepared.roster[0].label == "Narrator"
    assert prepared.lines[0].subject_id == "subject_narrator"
    assert prepared.lines[0].speaker_id == "S1"


def test_planner_normalizer_binds_alias_and_fills_timing(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A buyer answers the vendor.", duration_seconds=15)
    project = planner._plan_heuristically(brief, [])
    project.world_bible = _world()
    project.shots[0].dialogue = [DialogueLine(speaker="梁文锋", text="这片芯片我买了。")]
    project.shots[0].continuity_in.active_subject_ids = ["liang"]

    output = planner._normalize_agent_output(
        PlannerOutput(world_bible=project.world_bible, shots=project.shots),
        brief,
        [],
    )

    line = output.shots[0].dialogue[0]
    assert line.speaker == "Liang Wenfeng"
    assert line.subject_id == "liang"
    assert line.speaker_id == "S1"
    assert line.start_seconds is not None
    assert line.end_seconds is not None
    assert output.world_bible.subjects[1].speaker_id is None


def test_shot_harness_rejects_on_screen_speaker_missing_from_active_cast(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    shot = ShotSpec(
        index=0,
        title="Wrong speaker",
        purpose="Catch a role swap",
        duration_seconds=8,
        prompt="Huang waits at the counter.",
        continuity_in=ContinuityState(characters=["Huang Renxun in a leather jacket"]),
        dialogue=[DialogueLine(speaker="梁文锋", text="我来回答。")],
    )

    with pytest.raises(DialogueHarnessError) as raised:
        planner._apply_dialogue_harness(shot, _world(), shot_index=0)

    assert raised.value.code == "speaker_not_active"


def test_planner_roles_define_closed_roster_timing_and_retry_contract(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A buyer answers the vendor.", duration_seconds=15)
    style = style_prompt(brief.style_preset, brief.style_instructions)
    blueprint = ShotBlueprint(
        source_section="Scene 1",
        duration_seconds=15,
        active_subjects=["Liang Wenfeng", "Huang Renxun"],
    )

    director = planner._director_system_prompt(brief, style)
    shot_director = planner._shot_director_system_prompt(
        brief,
        style,
        planner._h3_skill_contract(brief.style_preset),
        blueprint,
        is_first=True,
        needs_generated_anchor=True,
        world_bible=_world(),
    )
    critic = planner._continuity_critic_system_prompt(
        brief,
        style,
        planner._h3_skill_contract(brief.style_preset),
    )

    assert "canonical speaker roster" in director
    assert "subject_id" in shot_director and "speaker_id" in shot_director
    assert "Canonical roster snapshot" in shot_director
    assert "4.2 Chinese characters/s" in shot_director
    assert "Only the bound subject" in shot_director
    assert "reject invented or ambiguous speakers" in critic
    assert "windows shorter than the spoken text" in critic


def test_director_source_ledger_allows_only_complete_adjacent_screenplay_lines():
    prompt = """
人物：梁文锋、黄仁勋
梁文锋：（冷冷地）你这 H100，十万美金一片，算力虚标，还带后门。
梁文锋：我这 S5000 国产的，算力实打实，价格只要你零头。关键是——它保熟。【节拍：反击】
"""
    source = PlannerService._creator_dialogue_source(prompt)
    world = canonicalize_world_bible(_world(), source)
    assert world is not None
    combined = DialogueLine(
        speaker="Liang Wenfeng",
        subject_id="liang",
        speaker_id="S1",
        text=(
            "你这 H100，十万美金一片，算力虚标，还带后门。"
            "我这 S5000 国产的，算力实打实，价格只要你零头。关键是——它保熟。"
        ),
        start_seconds=0.5,
        end_seconds=14,
    )
    blueprint = ShotBlueprint(source_section="Climax", duration_seconds=15, dialogue=[combined])

    PlannerService._validate_creator_dialogue_selection([blueprint], source, world)

    second_only = blueprint.model_copy(
        update={
            "dialogue": [
                combined.model_copy(update={"text": "我这 S5000 国产的，算力实打实，价格只要你零头。关键是——它保熟。"})
            ]
        }
    )
    with pytest.raises(DialogueHarnessError) as partial_run:
        PlannerService._validate_creator_dialogue_selection([second_only], source, world)
    assert partial_run.value.code == "dialogue_source_mismatch"

    shortened = blueprint.model_copy(update={"dialogue": [combined.model_copy(update={"text": "关键是——它保熟。"})]})
    with pytest.raises(DialogueHarnessError) as raised:
        PlannerService._validate_creator_dialogue_selection([shortened], source, world)
    assert raised.value.code == "dialogue_source_mismatch"

    repeated = blueprint.model_copy(update={"dialogue": [combined, combined]})
    with pytest.raises(DialogueHarnessError) as repeated_error:
        PlannerService._validate_creator_dialogue_selection([repeated], source, world)
    assert repeated_error.value.code == "dialogue_source_mismatch"


def test_fl2va_prompt_locks_voice_to_the_canonical_visual_subject():
    world = canonicalize_world_bible(_world())
    assert world is not None
    shot = ShotSpec(
        index=0,
        title="The answer",
        purpose="The buyer answers without changing position",
        duration_seconds=8,
        prompt="Liang Wenfeng keeps eye contact and answers Huang Renxun.",
        opening_state="Both men face each other across the counter.",
        ending_state="Liang closes his lips and keeps eye contact.",
        visual_beats=[
            StoryboardBeat(
                start_seconds=0,
                end_seconds=8,
                visual_action="Liang answers while Huang listens without speaking",
                state_change="Liang closes his lips after the line",
            )
        ],
        dialogue=[
            DialogueLine(
                speaker="梁文锋",
                text="这片芯片我买了。",
                start_seconds=1,
                end_seconds=4,
            )
        ],
    )

    prompt = render_h3_prompt(
        shot,
        brief=ProjectBrief(prompt="A chip-market exchange."),
        world_bible=world,
    )

    assert "<Subject 1> is canonical label Liang Wenfeng" in prompt
    assert "exclusively owns voice (S1)" in prompt
    assert "<Subject 1> (S1) Liang Wenfeng" in prompt
    assert "speaker_liang" not in prompt
    assert "every other visible subject keeps their lips closed" in prompt


def test_hierarchical_planner_retries_only_the_rejected_shot(settings):
    configured = replace(
        settings,
        planner_base_url="http://planner.test/v1",
        planner_model="test-model",
        planner_allow_fallback=False,
        planner_pipeline_mode="hierarchical",
        planner_continuity_critic=False,
    )
    planner = PlannerService(configured, StudioRepository(configured.database_path))
    brief = ProjectBrief(prompt="A buyer pitches a domestic GPU.", duration_seconds=15)
    base = planner._plan_heuristically(brief, [])
    base.world_bible = _world()
    ledger_line = DialogueLine(
        speaker="梁文锋",
        text=(
            "你这 H100，十万美金一片，算力虚标，还带后门。"
            "我这 S5000 国产的，算力实打实，价格只要你零头。关键是——它保熟。"
        ),
        start_seconds=0.9,
        end_seconds=12.5,
    )
    blueprint = planner._blueprint_from_shot(base.shots[0]).model_copy(
        update={"dialogue": [ledger_line], "active_subject_ids": ["liang"]}
    )
    director_payload = DirectorPlan(world_bible=base.world_bible, shot_blueprints=[blueprint]).model_dump(mode="json")
    bad_shot = base.shots[0].model_dump(mode="json")
    bad_shot["dialogue"] = [ledger_line.model_copy(update={"end_seconds": 5.4}).model_dump(mode="json")]
    repaired_shot = json.loads(json.dumps(bad_shot))
    repaired_shot["dialogue"][0]["end_seconds"] = 12.5
    shot_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        if payload["stage"] == "creative_director":
            result = director_payload
        else:
            shot_requests.append(payload)
            result = {"shot": bad_shot if len(shot_requests) == 1 else repaired_shot}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    planner._transport = httpx.MockTransport(handler)
    output = asyncio.run(planner._plan_with_llm(brief, []))

    assert len(shot_requests) == 2
    feedback = shot_requests[1]["harness_feedback"]
    assert isinstance(feedback, dict)
    assert "dialogue_ledger_mismatch" in str(feedback["error"])
    assert output.shots[0].dialogue[0].speaker == "Liang Wenfeng"
    assert output.shots[0].dialogue[0].end_seconds == 12.5


def test_planner_wrappers_attach_shot_index_to_structured_errors():
    with pytest.raises(DialogueHarnessError) as raised:
        bind_dialogue_lines([DialogueLine(speaker="missing", text="Hello.")], _world(), shot_index=3)
    assert raised.value.details["shot_index"] == 3

    with pytest.raises(DialogueHarnessError) as raised:
        schedule_dialogue_lines(
            [DialogueLine(speaker="Liang Wenfeng", text="一" * 100, start_seconds=0, end_seconds=1)],
            15,
            shot_index=4,
        )
    assert raised.value.details["shot_index"] == 4
