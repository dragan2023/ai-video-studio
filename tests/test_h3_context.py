from __future__ import annotations

from long_video_studio.domain import (
    ContinuityState,
    DialogueLine,
    ProjectBrief,
    ShotSpec,
    StoryboardBeat,
    SubjectCard,
    WorldBible,
)
from long_video_studio.h3_context import (
    SECTION_ORDER,
    compile_ref2va_context,
    stable_speaker_ids,
)


def _shot(index: int = 1, *, dialogue: list[DialogueLine] | None = None) -> ShotSpec:
    return ShotSpec(
        index=index,
        title="Confrontation",
        purpose="Advance the confrontation with a controlled physical beat",
        duration_seconds=8,
        prompt="The woman crosses the hall, raises her hand, and stops beside the table.",
        camera="the camera tracks right with small amplitude at slow speed",
        audio_prompt="Footsteps, fabric movement, and the quiet room tone remain synchronized.",
        music_prompt="Sparse low strings at a slow tempo, fading near the end.",
        dialogue=dialogue or [],
        continuity_in=ContinuityState(
            characters=["young woman with a red robe", "older man in a dark coat"],
            wardrobe=["red embroidered robe", "dark wool coat"],
            props=["wooden table"],
            location="stone palace hall",
            lighting="warm window light from camera left",
            camera="eye-level tracking direction to the right",
            action="the woman is three steps from the table",
            audio="footsteps and room tone continue from the prior clip",
        ),
        continuity_out=ContinuityState(
            characters=["young woman with a red robe", "older man in a dark coat"],
            location="stone palace hall",
            action="the woman holds a tense hand gesture beside the table",
        ),
    )


def test_context_ir_uses_official_six_section_order_and_stable_labels():
    shot = _shot(
        dialogue=[
            DialogueLine(
                speaker="young woman",
                text="你终于来了。",
                language="Chinese",
                delivery="quiet but threatening",
                start_seconds=4,
                end_seconds=5,
            )
        ]
    )
    world = WorldBible(
        logline="A palace confrontation",
        visual_style="warm cinematic realism with restrained shadows",
        character_notes=["young woman with a red robe", "older man in a dark coat"],
        location_notes=["stone palace hall"],
        continuity_rules=["Keep the geography readable."],
    )
    context = compile_ref2va_context(
        shot,
        [
            {"kind": "image", "label": "Picture 7", "description": "identity and opening composition"},
            {"kind": "video", "label": "Video 3", "description": "the previous five seconds"},
        ],
        brief=ProjectBrief(prompt="A palace confrontation", style_preset="noir"),
        world_bible=world,
    )
    prompt = context.render()
    headings = [f"{section}:" for section in SECTION_ORDER]
    assert [prompt.index(heading) for heading in headings] == sorted(prompt.index(h) for h in headings)
    assert "<Picture 7>" in prompt
    assert "<Video 3>" in prompt
    assert "<Subject 1>" in prompt
    assert "retention_analysis:" in prompt
    assert "At the opening," in prompt
    assert "<d>[Chinese] 你终于来了。</d>" in prompt
    assert "young woman" in prompt
    assert "warm cinematic realism" in prompt


def test_context_ir_does_not_turn_visual_direction_into_speech():
    shot = _shot()
    prompt = compile_ref2va_context(
        shot,
        [("image", "Picture 1", "identity reference")],
    ).render()
    assert "The woman crosses the hall" in prompt
    assert "No spoken dialogue, narration, or voice-over occurs" in prompt
    assert "<d>" not in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt


def test_context_ir_renders_structured_storyboard_beats_and_creator_style_override():
    shot = _shot().model_copy(
        update={
            "opening_state": "The woman stands three steps from the table.",
            "ending_state": "Her raised hand settles beside the table.",
            "continuity_handoff": "Keep the palace axis, red robe, and room tone unchanged.",
            "reference_anchors": ["Subject identity: the red-robed woman", "Scene: stone palace hall"],
            "hook": "Her hand stops just before touching the table.",
            "visual_beats": [
                StoryboardBeat(
                    start_seconds=0,
                    end_seconds=4,
                    visual_action="She crosses the hall in one continuous walk",
                    state_change="She reaches the table",
                    camera="slow small-amplitude tracking shot",
                    sound="three synchronized footsteps and fabric movement",
                ),
                StoryboardBeat(
                    start_seconds=4,
                    end_seconds=8,
                    visual_action="She raises her hand and brakes before contact",
                    state_change="Her hand settles beside the table",
                    camera="the camera decelerates into a static hold",
                    sound="the final footstep decays into room tone",
                ),
            ],
        }
    )
    prompt = compile_ref2va_context(
        shot,
        [("video", "Video 1", "prior clip")],
        brief=ProjectBrief(
            prompt="A palace confrontation",
            style_preset="noir",
            style_instructions="Keep the palace geography readable.",
        ),
    ).render()
    assert "Creator style override: Keep the palace geography readable." in prompt
    assert "From 0.000s to 4.000s" in prompt
    assert "She crosses the hall in one continuous walk" in prompt
    assert "final moment preserves this continuity" in prompt
    assert "Keep the palace axis" in prompt
    assert "The primary hook is" in prompt
    assert "[Shot 1]" in prompt
    assert "At 00:00.000" not in prompt


def test_stable_speaker_ids_follow_film_order_and_can_be_reused():
    first = _shot(
        0,
        dialogue=[DialogueLine(speaker="woman", text="先走。")],
    )
    second = _shot(
        1,
        dialogue=[
            DialogueLine(speaker="man", text="等等。"),
            DialogueLine(speaker="woman", text="不。"),
        ],
    )
    ids = stable_speaker_ids([second, first])
    assert ids == {"woman": "S1", "man": "S2"}
    context = compile_ref2va_context(
        second,
        [("video", "Video 1", "prior clip")],
        speaker_ids=ids,
    )
    prompt = context.render()
    assert "(S2) man" in prompt
    assert "(S1) woman" in prompt


def test_reference_labels_are_normalized_without_losing_category_order():
    context = compile_ref2va_context(
        _shot(),
        [
            {"kind": "video", "label": "source", "description": "temporal source"},
            {"kind": "image", "label": "<Picture 1>", "description": "identity source"},
            {"kind": "audio", "label": "voice", "description": "voice texture"},
        ],
    )
    prompt = context.render()
    assert "<Video 1>" in prompt
    assert "<Picture 1>" in prompt
    assert "<Audio 1>" in prompt
    assert "<source>" not in prompt


def test_context_ir_keeps_retention_speaker_free_and_uses_reference_roles():
    shot = _shot(
        dialogue=[DialogueLine(speaker="young woman", text="等等。", language="zh-CN", start_seconds=4)]
    ).model_copy(
        update={
            "visual_beats": [
                StoryboardBeat(
                    start_seconds=0,
                    end_seconds=4,
                    visual_action="She walks toward the table",
                    state_change="She reaches the table",
                ),
                StoryboardBeat(
                    start_seconds=4,
                    end_seconds=8,
                    visual_action="She stops and speaks",
                    state_change="Her lips close after the line",
                ),
            ]
        }
    )
    prompt = compile_ref2va_context(
        shot,
        [
            {
                "kind": "picture",
                "label": "Picture 1",
                "description": "identity reference",
                "role": "identity",
                "relationship": "fully_preserved",
            },
            {
                "kind": "video",
                "label": "Video 1",
                "description": "prior clip",
                "role": "continuation",
                "relationship": "fully_preserved",
            },
        ],
    ).render()
    summary = prompt.split("summary:\n", 1)[1].split("\n\nretention_analysis:", 1)[0]
    retention = prompt.split("retention_analysis:\n", 1)[1].split("\n\ndetailed_description:", 1)[0]
    detailed = prompt.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    assert "keyframe completion" not in summary
    assert "reference generation" in summary
    assert "(S1)" not in retention
    assert "[Shot 1]" in detailed
    assert "[Shot 2]" not in detailed
    assert detailed.index("From 4.000s to 8.000s") < detailed.index("<d>[Chinese] 等等。</d>")
    assert "lips close" in detailed


def test_context_ir_voice_over_keeps_visible_lips_closed():
    shot = _shot(
        dialogue=[
            DialogueLine(
                speaker="Narrator",
                text="The hall remembers.",
                language="English",
                mode="voice_over",
                start_seconds=1,
            )
        ]
    ).model_copy(
        update={
            "visual_beats": [
                StoryboardBeat(
                    start_seconds=0,
                    end_seconds=8,
                    visual_action="The woman studies the empty hall",
                    state_change="Her gaze settles on the doorway",
                )
            ]
        }
    )
    prompt = compile_ref2va_context(shot, [("video", "Video 1", "prior clip")]).render()
    assert "The line is voice-over; every visible character keeps their lips closed." in prompt
    assert "<d>[English] The hall remembers.</d>" in prompt


def test_context_ir_removes_contradictory_no_dialogue_audio_clause():
    shot = _shot(
        dialogue=[DialogueLine(speaker="woman", text="Wait.", language="English", start_seconds=2)]
    ).model_copy(
        update={
            "audio_prompt": "Footsteps continue. No spoken dialogue, narration, or voice-over.",
            "visual_beats": [
                StoryboardBeat(
                    start_seconds=0,
                    end_seconds=8,
                    visual_action="The woman turns toward the doorway",
                    state_change="Her lips close after the line",
                )
            ],
        }
    )
    prompt = compile_ref2va_context(shot, [("video", "Video 1", "prior clip")]).render()
    assert "No spoken dialogue" not in prompt.split("overall_soundscape:\n", 1)[1]
    assert "<d>[English] Wait.</d>" in prompt


def test_context_ir_keeps_stable_subject_card_identity_binding():
    shot = _shot().model_copy(
        update={
            "continuity_in": ContinuityState(characters=["白鹿"], action="standing"),
        }
    )
    bible = WorldBible(
        logline="A palace exchange",
        visual_style="cinematic realism",
        subjects=[
            SubjectCard(
                subject_id="bailu",
                label="白鹿",
                aliases=["Bai Lu", "演员白鹿"],
                visual_identity="same face, hair, and body proportions",
                wardrobe="red court robe",
                speaker_id="S1",
            )
        ],
    )
    prompt = compile_ref2va_context(shot, [("video", "Video 1", "prior clip")], world_bible=bible).render()
    assert "<Subject 1> is the silent canonical character" in prompt
    assert "aliases=Bai Lu, 演员白鹿" in prompt
    assert "visual_identity=same face" in prompt
    assert "no speaker ID is assigned in this clip" in prompt


def test_context_ir_preserves_long_planner_fields_and_beat_details():
    """Normal planner detail must survive compilation instead of word slicing."""

    def long_field(label: str) -> str:
        return " ".join([f"{label}_detail_{index}" for index in range(70)])

    shot = _shot().model_copy(
        update={
            "prompt": long_field("shot_prompt") + " SHOT_PROMPT_TAIL_SENTINEL",
            "camera": long_field("camera") + " CAMERA_TAIL_SENTINEL",
            "opening_state": long_field("opening") + " OPENING_TAIL_SENTINEL",
            "ending_state": long_field("ending") + " ENDING_TAIL_SENTINEL",
            "continuity_handoff": long_field("handoff") + " HANDOFF_TAIL_SENTINEL",
            "hook": long_field("hook") + " HOOK_TAIL_SENTINEL",
            "reference_anchors": [long_field("anchor") + " ANCHOR_TAIL_SENTINEL"],
            "visual_beats": [
                StoryboardBeat(
                    start_seconds=0,
                    end_seconds=8,
                    visual_action=long_field("action") + " ACTION_TAIL_SENTINEL",
                    state_change=long_field("state") + " STATE_TAIL_SENTINEL",
                    camera=long_field("beat_camera") + " BEAT_CAMERA_TAIL_SENTINEL",
                    sound=long_field("sound") + " SOUND_TAIL_SENTINEL",
                )
            ],
        }
    )
    prompt = compile_ref2va_context(shot, [("video", "Video 1", "prior clip")]).render()

    for marker in (
        "SHOT_PROMPT_TAIL_SENTINEL",
        "CAMERA_TAIL_SENTINEL",
        "OPENING_TAIL_SENTINEL",
        "ENDING_TAIL_SENTINEL",
        "HANDOFF_TAIL_SENTINEL",
        "HOOK_TAIL_SENTINEL",
        "ANCHOR_TAIL_SENTINEL",
        "ACTION_TAIL_SENTINEL",
        "STATE_TAIL_SENTINEL",
        "BEAT_CAMERA_TAIL_SENTINEL",
        "SOUND_TAIL_SENTINEL",
    ):
        assert marker in prompt


def test_context_ir_emits_first_frame_lock_and_no_replay_boundary():
    previous = _shot(1).model_copy(update={"ending_state": "BOUNDARY_END_STATE_WITH_LOCKED_POSE_AND_PROP"})
    shot = _shot(2).model_copy(update={"opening_state": "BOUNDARY_OPEN_STATE_BEFORE_NEW_ACTION"})
    prompt = compile_ref2va_context(
        shot,
        [
            {
                "kind": "picture",
                "label": "Picture 1",
                "description": "the exact final frame of the previous shot",
                "role": "first_frame",
                "relationship": "fully_preserved",
            },
            {
                "kind": "video",
                "label": "Video 1",
                "description": "the previous clip",
                "role": "continuation",
                "relationship": "fully_preserved",
            },
        ],
        previous_shot=previous,
    ).render()
    detailed = prompt.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    assert "first generated frame must match <Picture 1> exactly" in detailed
    assert "do not replay" in detailed
    assert "BOUNDARY_END_STATE_WITH_LOCKED_POSE_AND_PROP" in detailed
