from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from test_assets import png_bytes

from long_video_studio.assets import AssetService
from long_video_studio.compiler import FilmCompiler
from long_video_studio.domain import (
    AssetRole,
    AssetUpdate,
    ContinuationMode,
    DialogueLine,
    FilmProject,
    ProjectBrief,
    ShotSpec,
    ShotTask,
    TransitionKind,
    UltraFastAnchorStrategy,
    WorldBible,
)
from long_video_studio.planner import DirectorPlan, PlannerOutput, PlannerService, ShotBlueprint
from long_video_studio.repository import StudioRepository
from long_video_studio.style_registry import STYLE_REGISTRY, get_style_contract, style_prompt


def test_planner_builds_continuity_aware_film_ir(settings):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    hero = assets.ingest_stream(
        png_bytes(),
        "hero.png",
        "image/png",
        roles=[AssetRole.CHARACTER, AssetRole.START_FRAME],
    )
    planner = PlannerService(settings, repository)
    project = asyncio.run(
        planner.plan(
            ProjectBrief(
                title="Cat story",
                prompt="A woman and her cat turn a rainy evening into a joyful game.",
                duration_seconds=60,
                reference_asset_ids=[hero.id],
            )
        )
    )
    assert len(project.shots) == 4
    assert sum(shot.duration_seconds for shot in project.shots) == 60
    assert all(4 <= shot.duration_seconds <= 15 for shot in project.shots)
    assert project.shots[0].start_frame_asset_id == hero.id
    assert project.shots[1].continuity_from_shot_id == project.shots[0].id
    assert all(shot.opening_state for shot in project.shots)
    assert all(shot.ending_state for shot in project.shots)
    assert all(shot.continuity_handoff for shot in project.shots)
    assert all(shot.reference_anchors for shot in project.shots)
    assert all(shot.hook for shot in project.shots)
    assert all(shot.visual_beats for shot in project.shots)
    assert all(
        shot.visual_beats[0].start_seconds == 0 and shot.visual_beats[-1].end_seconds == shot.duration_seconds
        for shot in project.shots
    )
    assert len(project.timeline) == len(project.shots)


def test_planner_never_retrieves_unselected_library_assets(settings):
    repository = StudioRepository(settings.database_path)
    asset = AssetService(settings, repository).ingest_stream(
        png_bytes("blue"),
        "private-character.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    asset = AssetService(settings, repository).update(
        asset.id,
        AssetUpdate(display_name="Unselected Character"),
    )
    planner = PlannerService(settings, repository)
    brief = ProjectBrief(
        prompt="A completely new character enters an empty studio.",
        duration_seconds=15,
        reference_asset_ids=[],
    )

    project = asyncio.run(planner.plan(brief))
    serialized = project.model_dump_json()
    assert project.brief.reference_asset_ids == []
    assert all(shot.reference_asset_ids == [] for shot in project.shots)
    assert asset.id not in serialized
    assert asset.display_name not in serialized


def test_planner_retrieves_only_explicitly_selected_assets(settings):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    unselected = assets.ingest_stream(
        png_bytes("blue"),
        "unselected.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    selected = assets.ingest_stream(
        png_bytes("red"),
        "selected.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    planner = PlannerService(settings, repository)
    brief = ProjectBrief(
        prompt="A character enters an empty studio.",
        duration_seconds=15,
        reference_asset_ids=[selected.id],
    )

    retrieved = planner._retrieve_assets(brief)
    assert [asset.id for asset in retrieved] == [selected.id]
    assert unselected.id not in {asset.id for asset in retrieved}


def test_style_registry_has_a_complete_contract_for_each_planner_preset():
    expected = {
        "cinematic",
        "documentary",
        "music_video",
        "commercial",
        "noir",
        "animation",
        "retro",
        "surreal",
        "energetic",
        "custom",
    }
    assert set(STYLE_REGISTRY) == expected
    for contract in STYLE_REGISTRY.values():
        rendered = contract.render()
        assert contract.label in rendered
        assert "Palette:" in rendered
        assert "Lighting:" in rendered
        assert "Lens and camera language:" in rendered
        assert "Motion rhythm:" in rendered
        assert "Global negative constraints:" in rendered
        assert contract.negative_constraints


def test_style_registry_falls_back_safely_and_honors_creator_override():
    assert get_style_contract("noir").id == "noir"
    assert get_style_contract("frontend-only").id == "cinematic"
    rendered = style_prompt("noir", "No red neon; keep the palace geography readable.")
    assert "neo-noir" in rendered
    assert "No red neon" in rendered


def test_heuristic_planner_projects_style_contract_into_world_bible_and_shots(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    project = planner._plan_heuristically(
        ProjectBrief(prompt="A palace confrontation.", duration_seconds=15, style_preset="noir"),
        [],
    )
    contract = STYLE_REGISTRY["noir"]
    assert contract.palette in project.world_bible.visual_style
    assert contract.lighting in project.shots[0].continuity_in.lighting
    assert contract.compact() in project.shots[0].prompt
    assert any("project premise" in anchor for anchor in project.shots[0].reference_anchors)
    assert any(contract.medium in anchor for anchor in project.shots[0].reference_anchors)


def test_llm_planner_prompt_contains_structured_style_contract(settings):
    planner = PlannerService(
        replace(
            settings,
            planner_base_url="http://planner.test/v1",
            planner_model="test-model",
            planner_allow_fallback=False,
        ),
        StudioRepository(settings.database_path),
    )
    brief = ProjectBrief(prompt="A noir palace mystery.", style_preset="noir", duration_seconds=15)
    expected = planner._plan_heuristically(brief, [])
    payload = PlannerOutput(world_bible=expected.world_bible, shots=expected.shots).model_dump_json()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        assert "Style DNA: 黑色电影" in system
        assert "Global negative constraints:" in system
        assert "Apply this contract to the world bible and every shot" in system
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": payload}}]},
        )

    planner._transport = httpx.MockTransport(handler)
    output = asyncio.run(planner._plan_with_llm(brief, []))
    assert output.shots


def test_hierarchical_planner_calls_director_shot_directors_and_critic(settings):
    """The LLM is split by responsibility while the public Film IR stays stable."""

    repository = StudioRepository(settings.database_path)
    planner = PlannerService(
        replace(
            settings,
            planner_base_url="http://planner.test/v1",
            planner_model="test-model",
            planner_allow_fallback=False,
            planner_pipeline_mode="hierarchical",
            planner_shot_concurrency=2,
            planner_continuity_critic=True,
        ),
        repository,
    )
    brief = ProjectBrief(prompt="A woman and a cat cross a rainy courtyard.", duration_seconds=30)
    spine = planner._plan_heuristically(brief, [])
    blueprints = [planner._blueprint_from_shot(shot) for shot in spine.shots]
    director_payload = DirectorPlan(
        world_bible=spine.world_bible,
        shot_blueprints=blueprints,
    ).model_dump(mode="json")
    shot_payloads = [shot.model_dump(mode="json") for shot in spine.shots]
    final_payload = PlannerOutput(
        world_bible=spine.world_bible,
        shots=spine.shots,
    ).model_dump(mode="json")
    critic_payload = json.loads(json.dumps(final_payload))
    critic_payload["shots"][1]["transition_kind"] = "hard_cut"
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "messages" in body:
            stage_payload = json.loads(body["messages"][1]["content"])
            stage = str(stage_payload["stage"])
        else:  # pragma: no cover - this test exercises the chat wire
            stage_payload = json.loads(body["input"][0]["content"][0]["text"])
            stage = str(stage_payload["stage"])
        calls.append((stage, stage_payload))
        if stage == "creative_director":
            result = director_payload
        elif stage == "shot_director":
            index = int(stage_payload["shot_index"])
            result = {"shot": shot_payloads[index]}
        else:
            result = critic_payload
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    planner._transport = httpx.MockTransport(handler)
    output = asyncio.run(planner._plan_with_llm(brief, []))

    assert [stage for stage, _ in calls] == [
        "creative_director",
        "shot_director",
        "shot_director",
        "continuity_critic",
    ]
    first_shot_payload = calls[1][1]
    assert first_shot_payload["previous_blueprint"] is None
    assert first_shot_payload["next_blueprint"] is not None
    second_shot_payload = calls[2][1]
    assert second_shot_payload["previous_blueprint"] is not None
    assert second_shot_payload["next_blueprint"] is None
    assert len(output.shots) == 2
    assert output.shots[0].transition_kind is TransitionKind.ANCHOR
    assert output.shots[1].transition_kind is TransitionKind.CONTINUOUS
    assert sum(shot.duration_seconds for shot in output.shots) == 30
    assert [shot.index for shot in output.shots] == [0, 1]


def test_hierarchical_planner_keeps_draft_when_critic_json_is_truncated(settings, monkeypatch):
    repository = StudioRepository(settings.database_path)
    planner = PlannerService(
        replace(
            settings,
            planner_base_url="http://planner.test/v1",
            planner_model="test-model",
            planner_allow_fallback=False,
            planner_pipeline_mode="hierarchical",
            planner_continuity_critic=True,
        ),
        repository,
    )
    brief = ProjectBrief(prompt="A woman crosses a rainy courtyard.", duration_seconds=24)
    spine = planner._plan_heuristically(brief, [])
    blueprints = [planner._blueprint_from_shot(shot) for shot in spine.shots]
    director_payload = DirectorPlan(
        world_bible=spine.world_bible,
        shot_blueprints=blueprints,
    ).model_dump(mode="json")
    shot_payloads = [shot.model_dump(mode="json") for shot in spine.shots]

    async def fake_request_json(
        _self,
        _client,
        _system_prompt,
        user_payload,
        *,
        schema,
        schema_name,
    ):
        del schema
        if schema_name == "nautilus_creative_director":
            return director_payload
        if schema_name.startswith("nautilus_shot_director_"):
            return {"shot": shot_payloads[int(user_payload["shot_index"])]}
        raise json.JSONDecodeError("truncated critic response", "{", 1)

    monkeypatch.setattr(PlannerService, "_request_json", fake_request_json)
    output = asyncio.run(planner._plan_with_llm(brief, []))

    assert len(output.shots) == len(spine.shots)
    assert [shot.index for shot in output.shots] == list(range(len(spine.shots)))
    assert all(shot.prompt for shot in output.shots)


def test_planner_retries_transient_provider_overload(settings, monkeypatch):
    configured = replace(
        settings,
        planner_retry_attempts=2,
        planner_retry_backoff_seconds=0,
    )
    repository = StudioRepository(configured.database_path)
    planner = PlannerService(configured, repository)
    calls = 0

    async def fake_wire(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Our servers are currently overloaded. Please try again later.")
        return {"ok": True}

    monkeypatch.setattr(planner, "_request_json_wire", fake_wire)

    async def request():
        async with httpx.AsyncClient() as client:
            return await planner._request_json(
                client,
                "system",
                {"stage": "test"},
                schema={"type": "object"},
                schema_name="test_retry",
            )

    result = asyncio.run(request())
    assert result == {"ok": True}
    assert calls == 2


def test_continuity_critic_cannot_erase_shot_director_anchor(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A creator addresses the camera.", duration_seconds=15)
    draft = planner._plan_heuristically(brief, [])
    assert draft.shots[0].anchor_prompt
    critic = PlannerOutput(
        world_bible=draft.world_bible,
        shots=[draft.shots[0].model_copy(update={"anchor_prompt": ""})],
    )

    restored = planner._restore_critic_dropped_fields(critic, draft)
    assert restored.shots[0].anchor_prompt == draft.shots[0].anchor_prompt


def test_normalizer_preserves_an_explicit_hard_cut(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A story with a deliberate scene cut.", duration_seconds=16)
    project = planner._plan_heuristically(brief, [])
    project.shots[1] = project.shots[1].model_copy(
        update={
            "transition_kind": TransitionKind.HARD_CUT,
            "continuity_from_shot_id": None,
            "task": ShotTask.FL2VA,
        }
    )
    normalized = planner._normalize_agent_output(
        PlannerOutput(world_bible=project.world_bible, shots=project.shots),
        brief,
        [],
    )
    assert normalized.shots[1].transition_kind is TransitionKind.HARD_CUT
    assert normalized.shots[1].continuity_from_shot_id is None


def test_shot_director_wire_schema_fragment_in_dialogue_is_sanitized(settings):
    blueprint = ShotBlueprint(
        index=0,
        title="Opening",
        purpose="A woman speaks",
        source_section="Scene 1",
        duration_seconds=8,
        opening_state="standing",
        ending_state="settled",
    )
    shot = PlannerService._coerce_shot_payload(
        {
            "shot": {
                "dialogue": [
                    {
                        "speaker": "woman",
                        "text": "Wait.",
                        "language": "English",
                        "delivery": {"title": "Delivery", "type": "string"},
                    }
                ],
                "prompt": "The woman turns toward the doorway.",
                "visual_beats": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 8,
                        "visual_action": "She turns",
                        "state_change": "She settles",
                        "camera": "The camera holds",
                        "sound": "Footsteps",
                    }
                ],
            }
        },
        0,
        blueprint,
    )
    assert shot.dialogue[0].delivery == "natural"


def test_compiler_hides_unavailable_model_in_warnings(settings):
    repository = StudioRepository(settings.database_path)
    planner = PlannerService(settings, repository)
    project = asyncio.run(planner.plan(ProjectBrief(prompt="A calm 30 second travel story.", duration_seconds=30)))
    plan = FilmCompiler(settings).compile(project)
    video_stages = [stage for stage in plan.stages if stage.kind == "video"]
    assert len(video_stages) == len(project.shots)
    assert any("endpoint is not configured" in warning for warning in plan.warnings)
    assert any("generated anchor frame" in warning for warning in plan.warnings)
    assert plan.stages[-1].kind == "assembly"
    assert plan.deployments[0].capability_id == "minimax-h3-fl2va"
    assert plan.deployments[0].status == "unconfigured"
    assert plan.estimated_seconds > 0


def test_compiler_exposes_configured_image_edit_provider(settings):
    configured = replace(
        settings,
        image_edit_provider="openai-compatible",
        image_edit_base_url="https://images.example.test",
        image_edit_model="Qwen-Image-Edit-2509",
        image_edit_max_references=4,
    )

    capability = FilmCompiler(configured).capabilities()[0]

    assert capability.available is True
    assert capability.endpoint == "https://images.example.test"
    assert capability.supports_multiple_references is True
    assert capability.recommended_gpus == 0


def test_compiler_rejects_multi_image_claim_for_base_qwen_image_edit(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="https://images.example.test",
        image_edit_model="Qwen/Qwen-Image-Edit",
        image_edit_max_references=4,
    )

    capability = FilmCompiler(configured).capabilities()[0]

    assert capability.available is False
    assert capability.supports_multiple_references is False
    assert any("single-image" in note for note in capability.notes)


def _anchor_mode_project() -> FilmProject:
    first = ShotSpec(
        index=0,
        title="Opening",
        purpose="Establish the scene",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        prompt="Establish the scene with the lead character.",
        anchor_prompt="Opening still: the lead character stands in the established scene.",
        reference_asset_ids=["scene"],
    )
    continuous = ShotSpec(
        index=1,
        title="Follow-through",
        purpose="Continue the action",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        prompt="Continue the action without a cut.",
        reference_asset_ids=["scene"],
        continuity_from_shot_id=first.id,
    )
    ref2va = ShotSpec(
        index=2,
        title="Audio beat",
        purpose="Use the audio-conditioned branch",
        duration_seconds=4,
        task=ShotTask.REF2VA,
        prompt="A distinct audio-conditioned beat.",
        reference_asset_ids=["scene", "audio"],
    )
    cut = ShotSpec(
        index=3,
        title="New scene",
        purpose="Start a new scene",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        prompt="Start the new scene with a clean visual anchor.",
        anchor_prompt="Opening still: establish the new scene and subject arrangement.",
        reference_asset_ids=["scene"],
    )
    return FilmProject(
        brief=ProjectBrief(prompt="A sixteen second story.", duration_seconds=16),
        world_bible=WorldBible(logline="A short story", visual_style="cinematic"),
        shots=[first, continuous, ref2va, cut],
    )


def _configured_image_edit(settings, mode: str):
    return replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen/Qwen-Image-Edit-2509",
        image_edit_anchor_mode=mode,
        h3_fl2va_url="http://fl2va.test",
        h3_ref2va_url="http://ref2va.test",
    )


@pytest.mark.parametrize(
    ("mode", "expected_shots"),
    [
        ("first-shot", {0}),
        ("scene-cuts", {0, 3}),
        ("every-shot", {0, 3}),
    ],
)
def test_compiler_image_edit_anchor_modes_apply_only_to_fl2va(settings, mode, expected_shots):
    project = _anchor_mode_project()
    plan = FilmCompiler(_configured_image_edit(settings, mode)).compile(project)

    keyframes = [stage for stage in plan.stages if stage.kind == "keyframe"]
    positions = {
        project.shots.index(next(shot for shot in project.shots if shot.id == stage.shot_id)) for stage in keyframes
    }

    assert positions == expected_shots
    assert all(stage.capability_id == "qwen-image-edit" for stage in keyframes)
    assert all(stage.inputs["anchor_mode"] == mode for stage in keyframes)
    assert not any(stage.shot_id == project.shots[2].id for stage in keyframes)


def test_compiler_routes_zero_material_anchors_to_text_to_image(settings):
    configured = replace(
        settings,
        text_to_image_provider="vllm-omni",
        text_to_image_base_url="http://text-to-image.test",
        text_to_image_model="Qwen/Qwen-Image-2512",
    )
    repository = StudioRepository(configured.database_path)
    planner = PlannerService(configured, repository)
    project = asyncio.run(
        planner.plan(
            ProjectBrief(
                prompt="A creator enters a completely new empty studio.",
                duration_seconds=15,
                reference_asset_ids=[],
            )
        )
    )

    plan = FilmCompiler(configured, repository=repository).compile(project)
    keyframes = [stage for stage in plan.stages if stage.kind == "keyframe"]
    assert keyframes
    assert all(stage.capability_id == "qwen-image-t2i" for stage in keyframes)
    assert any(item.capability_id == "qwen-image-t2i" for item in plan.deployments)
    assert not any(item.capability_id == "qwen-image-edit" for item in plan.deployments)


def test_compiler_rejects_missing_planner_anchor(settings):
    project = _anchor_mode_project()
    project.shots[0].anchor_prompt = ""

    with pytest.raises(ValueError, match="planner-authored anchor_prompt"):
        FilmCompiler(_configured_image_edit(settings, "first-shot")).compile(project)


def test_compiler_anchor_dependencies_follow_continuity_boundary(settings):
    project = _anchor_mode_project()
    plan = FilmCompiler(_configured_image_edit(settings, "every-shot")).compile(project)
    keyframes = {stage.shot_id: stage for stage in plan.stages if stage.kind == "keyframe"}
    videos = {stage.shot_id: stage for stage in plan.stages if stage.kind == "video"}

    first, continuous, _, cut = project.shots
    assert keyframes[first.id].depends_on == []
    assert continuous.id not in keyframes
    assert videos[continuous.id].depends_on == [videos[first.id].id]
    assert videos[continuous.id].capability_id == "minimax-h3-ref2va"
    assert videos[continuous.id].inputs["continuation_mode"] == "fast"
    assert keyframes[cut.id].depends_on == []
    assert videos[cut.id].depends_on == [keyframes[cut.id].id]
    # REF2VA never gets an Image Edit stage, even when the global mode is
    # every-shot.
    assert project.shots[2].id not in keyframes


def test_compiler_quality_continuation_uses_full_clip_ref2va(settings):
    project = _anchor_mode_project()
    fast_plan = FilmCompiler(_configured_image_edit(settings, "scene-cuts")).compile(project)
    fast_stage = next(
        stage for stage in fast_plan.stages if stage.kind == "video" and stage.shot_id == project.shots[1].id
    )
    project.brief.continuation_mode = ContinuationMode.QUALITY

    plan = FilmCompiler(_configured_image_edit(settings, "scene-cuts")).compile(project)
    stage = next(stage for stage in plan.stages if stage.kind == "video" and stage.shot_id == project.shots[1].id)

    assert stage.capability_id == "minimax-h3-ref2va"
    assert stage.inputs["continuation_mode"] == "quality"
    assert stage.inputs["continuity_from_shot_id"] == project.shots[0].id
    assert stage.estimated_seconds > fast_stage.estimated_seconds


def test_compiler_ultra_fast_continuation_routes_boundary_frame_to_fl2va(settings):
    project = _anchor_mode_project()
    project.brief.continuation_mode = ContinuationMode.ULTRA_FAST
    project.brief.ultra_fast_anchor_strategy = UltraFastAnchorStrategy.BOUNDARY

    plan = FilmCompiler(_configured_image_edit(settings, "every-shot")).compile(project)
    stage = next(item for item in plan.stages if item.kind == "video" and item.shot_id == project.shots[1].id)

    assert stage.capability_id == "minimax-h3-fl2va"
    assert stage.inputs["continuation_mode"] == "ultra_fast"
    assert stage.depends_on == [
        next(item for item in plan.stages if item.kind == "video" and item.shot_id == project.shots[0].id).id
    ]
    assert not any(item.kind == "keyframe" and item.shot_id == project.shots[1].id for item in plan.stages)
    assert any("极速续写" in warning for warning in plan.warnings)


def test_compiler_ultra_fast_defaults_to_independent_fl2va_anchors(settings):
    project = _anchor_mode_project()
    project.brief.continuation_mode = ContinuationMode.ULTRA_FAST
    for shot in project.shots:
        shot.anchor_prompt = f"Independent opening still for {shot.title}, cinematic 16:9."

    plan = FilmCompiler(_configured_image_edit(settings, "first-shot")).compile(project)
    keyframes = [stage for stage in plan.stages if stage.kind == "keyframe"]
    videos = [stage for stage in plan.stages if stage.kind == "video"]

    assert {stage.shot_id for stage in keyframes} == {shot.id for shot in project.shots}
    assert all(stage.capability_id == "minimax-h3-fl2va" for stage in videos)
    second_keyframe = next(stage for stage in keyframes if stage.shot_id == project.shots[1].id)
    first_video = next(stage for stage in videos if stage.shot_id == project.shots[0].id)
    assert second_keyframe.depends_on == [first_video.id]
    assert second_keyframe.capability_id == "qwen-image-edit"
    assert any("fresh opening frame" in warning for warning in plan.warnings)


def test_compiler_keeps_fl2va_as_unconfigured_ref2va_fallback(settings):
    project = _anchor_mode_project()
    configured = replace(settings, h3_fl2va_url="http://fl2va.test")

    plan = FilmCompiler(configured).compile(project)
    stage = next(stage for stage in plan.stages if stage.kind == "video" and stage.shot_id == project.shots[1].id)

    assert stage.capability_id == "minimax-h3-fl2va"
    assert any("internal FL2VA boundary fallback" in warning for warning in plan.warnings)


def test_compiler_explicit_start_frame_is_not_replaced_by_continuation_ref2va(settings):
    project = _anchor_mode_project()
    project.shots[1].start_frame_asset_id = "creator-start"

    plan = FilmCompiler(_configured_image_edit(settings, "scene-cuts")).compile(project)
    stage = next(stage for stage in plan.stages if stage.kind == "video" and stage.shot_id == project.shots[1].id)

    assert stage.capability_id == "minimax-h3-fl2va"
    assert stage.inputs["continuation_mode"] is None
    assert stage.depends_on == []


def test_compiler_first_shot_mode_does_not_promote_later_fl2va(settings):
    project = _anchor_mode_project()
    project.shots[0].task = ShotTask.REF2VA
    plan = FilmCompiler(_configured_image_edit(settings, "first-shot")).compile(project)

    keyframes = [stage for stage in plan.stages if stage.kind == "keyframe"]

    assert keyframes == []


def test_compiler_disabled_image_edit_keeps_direct_video_path(settings):
    project = _anchor_mode_project()
    project.shots[0].start_frame_asset_id = "scene"
    project.shots[3].start_frame_asset_id = "scene"
    plan = FilmCompiler(settings).compile(project)

    assert [stage for stage in plan.stages if stage.kind == "keyframe"] == []
    assert not any(deployment.capability_id == "qwen-image-edit" for deployment in plan.deployments)
    videos = {stage.shot_id: stage for stage in plan.stages if stage.kind == "video"}
    assert videos[project.shots[1].id].depends_on == [videos[project.shots[0].id].id]


def test_compiler_explicit_start_frame_bypasses_configured_image_edit(settings):
    project = _anchor_mode_project()
    project.shots[0].start_frame_asset_id = "creator-start"

    plan = FilmCompiler(_configured_image_edit(settings, "scene-cuts")).compile(project)

    keyframes = [stage for stage in plan.stages if stage.kind == "keyframe"]
    assert all(stage.shot_id != project.shots[0].id for stage in keyframes)


def test_planner_does_not_retrieve_matching_library_assets_when_none_are_selected(settings):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    cat = assets.ingest_stream(
        png_bytes("orange"),
        "orange-cat.png",
        "image/png",
        tags=["猫", "客厅"],
        roles=[AssetRole.CHARACTER, AssetRole.START_FRAME],
    )
    planner = PlannerService(settings, repository)
    project = asyncio.run(
        planner.plan(
            ProjectBrief(
                prompt="一个女孩在客厅逗猫，猫开心地追逐玩具。",
                duration_seconds=30,
            )
        )
    )
    assert project.brief.reference_asset_ids == []
    assert project.shots[0].reference_asset_ids == []
    assert project.shots[0].start_frame_asset_id is None
    assert cat.id not in project.model_dump_json()


def test_planner_keeps_reference_images_out_of_start_frame_and_writes_anchor_without_image_edit(settings):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    character = assets.ingest_stream(
        png_bytes("red"),
        "character.png",
        "image/png",
        tags=["主角"],
        roles=[AssetRole.CHARACTER],
    )
    location = assets.ingest_stream(
        png_bytes("blue"),
        "location.png",
        "image/png",
        tags=["场景"],
        roles=[AssetRole.LOCATION],
    )

    project = asyncio.run(
        PlannerService(settings, repository).plan(
            ProjectBrief(
                prompt="主角在场景中发现一盏发光的灯。",
                duration_seconds=15,
                reference_asset_ids=[character.id, location.id],
            )
        )
    )

    first = project.shots[0]
    assert first.start_frame_asset_id is None
    assert {character.id, location.id}.issubset(first.reference_asset_ids)
    assert first.anchor_prompt
    assert "still" in first.anchor_prompt.lower()


def test_shot_director_requires_generated_anchor_when_no_start_frame_is_selected(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    blueprint = ShotBlueprint(
        index=0,
        title="Opening",
        purpose="Establish the scene",
        source_section="Scene 1",
        duration_seconds=8,
        transition_kind=TransitionKind.ANCHOR,
    )

    prompt = planner._shot_director_system_prompt(
        ProjectBrief(prompt="A quiet opening."),
        "cinematic style",
        "H3 rules",
        blueprint,
        is_first=True,
        needs_generated_anchor=True,
    )

    assert "MUST populate anchor_prompt" in prompt
    assert "standalone zero-second T2I" in prompt
    assert "Never mention 参考图" in prompt

    continuous_prompt = planner._shot_director_system_prompt(
        ProjectBrief(prompt="A quiet opening."),
        "cinematic style",
        "H3 rules",
        blueprint.model_copy(update={"transition_kind": TransitionKind.CONTINUOUS}),
        is_first=True,
        needs_generated_anchor=True,
    )
    assert "MUST populate anchor_prompt" in continuous_prompt


def test_zero_material_heuristic_anchor_never_invents_visual_references(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A creator enters a new studio.", duration_seconds=15)
    shot = planner._plan_heuristically(brief, []).shots[0]
    prompt = planner._anchor_prompt(brief=brief, shot=shot, assets=[])

    assert "No source image or visual reference is supplied" in prompt
    assert "selected canonical visual references" not in prompt
    assert "参考图" not in prompt
    assert "asset_" not in prompt


def test_planner_leaves_first_frame_empty_for_image_edit_references(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen/Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    scene = assets.ingest_stream(
        png_bytes("green"),
        "workshop.png",
        "image/png",
        tags=["工作室"],
        roles=[AssetRole.LOCATION],
    )

    project = asyncio.run(
        PlannerService(configured, repository).plan(
            ProjectBrief(
                prompt="一位创作者走进工作室，拿起桌上的发光道具。",
                duration_seconds=30,
                reference_asset_ids=[scene.id],
            )
        )
    )

    assert scene.id in project.shots[0].reference_asset_ids
    assert project.shots[0].start_frame_asset_id is None
    assert project.shots[0].anchor_prompt
    assert "single" in project.shots[0].anchor_prompt.lower() or "still" in project.shots[0].anchor_prompt.lower()
    assert "dialogue" in project.shots[0].anchor_prompt.lower()
    assert project.shots[0].dialogue == []
    assert "No dialogue" in project.shots[0].audio_prompt


def test_planner_preserves_explicit_start_frame_with_image_edit_configured(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen/Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    start = assets.ingest_stream(
        png_bytes("purple"),
        "opening.png",
        "image/png",
        tags=["指定首帧"],
        roles=[AssetRole.START_FRAME],
    )

    project = asyncio.run(
        PlannerService(configured, repository).plan(
            ProjectBrief(
                prompt="从指定画面自然开始一个连续镜头。",
                duration_seconds=30,
                reference_asset_ids=[start.id],
            )
        )
    )

    assert project.shots[0].start_frame_asset_id == start.id
    assert project.shots[0].anchor_prompt == ""


def test_responses_sse_planner_path_returns_structured_agent_output(settings):
    repository = StudioRepository(settings.database_path)
    planner = PlannerService(
        replace(
            settings,
            planner_base_url="http://planner.test/v1",
            planner_api_key="test-key",
            planner_model="test-model",
            planner_wire_api="responses",
            planner_allow_fallback=False,
            planner_source="codex:test",
        ),
        repository,
    )
    brief = ProjectBrief(
        prompt="A woman and a cat share a joyful evening.",
        duration_seconds=30,
        style_preset="documentary",
        style_instructions="One continuous handheld shot with honest room tone.",
    )
    expected = planner._plan_heuristically(brief, [])
    payload = PlannerOutput(world_bible=expected.world_bible, shots=expected.shots).model_dump_json()
    sse = "\n".join(
        [
            'data: {"type":"response.created"}',
            *[
                f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': chunk})}"
                for chunk in (payload[:80], payload[80:])
            ],
            'data: {"type":"response.completed","response":{"status":"completed"}}',
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["text"]["format"]["type"] == "json_schema"
        assert "documentary" in body["input"][0]["content"][0]["text"]
        assert "honest room tone" in body["input"][0]["content"][0]["text"]
        shot_schema = body["text"]["format"]["schema"]["$defs"]["ShotSpec"]
        assert {
            "opening_state",
            "ending_state",
            "continuity_handoff",
            "reference_anchors",
            "hook",
            "visual_beats",
        } <= set(shot_schema["required"])
        assert shot_schema["properties"]["reference_anchors"]["minItems"] == 1
        assert shot_schema["properties"]["visual_beats"]["minItems"] == 1
        assert shot_schema["properties"]["continuity_handoff"]["minLength"] == 1
        beat_schema = body["text"]["format"]["schema"]["$defs"]["StoryboardBeat"]
        assert set(beat_schema["required"]) == set(beat_schema["properties"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    planner._transport = httpx.MockTransport(handler)
    output = asyncio.run(planner._plan_with_llm(brief, []))

    assert len(output.shots) == len(expected.shots)
    assert output.shots[0].prompt == expected.shots[0].prompt
    assert output.shots[0].audio_prompt == expected.shots[0].audio_prompt
    assert output.shots[0].dialogue == expected.shots[0].dialogue
    assert all(shot.fps == 24 and shot.flow_shift == 12.0 for shot in output.shots)


def test_agent_planner_rejects_storyboards_without_h3_timeline_contract(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A palace confrontation.", duration_seconds=15)
    project = planner._plan_heuristically(brief, [])
    incomplete = project.shots[0].model_copy(
        update={
            "opening_state": "",
            "ending_state": "",
            "continuity_handoff": "",
            "reference_anchors": [],
            "hook": "",
            "visual_beats": [],
        }
    )
    with pytest.raises(ValueError, match="omitted H3 storyboard fields"):
        planner._normalize_agent_output(
            PlannerOutput(world_bible=project.world_bible, shots=[incomplete]),
            brief,
            [],
        )


def test_agent_planner_rejects_gapped_h3_timeline(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A palace confrontation.", duration_seconds=15)
    project = planner._plan_heuristically(brief, [])
    shot = project.shots[0]
    beats = list(shot.visual_beats)
    beats[1] = beats[1].model_copy(update={"start_seconds": beats[1].start_seconds + 1})
    gapped = shot.model_copy(update={"visual_beats": beats})
    with pytest.raises(ValueError, match="timeline has a gap"):
        planner._normalize_agent_output(
            PlannerOutput(world_bible=project.world_bible, shots=[gapped]),
            brief,
            [],
        )


def test_agent_planner_limits_overlong_anchor_prompt_at_sentence_boundary(settings):
    prompt = "A" * 900 + ". " + "B" * 200
    limited = PlannerService._limit_anchor_prompt(prompt)
    assert limited == "A" * 900 + "."
    assert len(limited) <= 1000


def test_agent_planner_rejects_non_english_h3_model_prose(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(prompt="A palace confrontation.", duration_seconds=15)
    project = planner._plan_heuristically(brief, [])
    invalid = project.shots[0].model_copy(update={"audio_prompt": "真实连续的环境声与动作声。"})
    with pytest.raises(ValueError, match="non-English H3 model field"):
        planner._normalize_agent_output(
            PlannerOutput(world_bible=project.world_bible, shots=[invalid]),
            brief,
            [],
        )


def test_planner_parses_proxy_double_envelope_and_closes_duration(settings):
    repository = StudioRepository(settings.database_path)
    planner = PlannerService(settings, repository)
    base = planner._plan_heuristically(ProjectBrief(prompt="A palace story.", duration_seconds=30), [])
    shots = [shot.model_dump(mode="json") for shot in base.shots]
    nested = {"world_bible": base.world_bible.model_dump(mode="json"), "shots": shots}
    payload = {"world_bible": base.world_bible.model_dump(mode="json"), "shots": [*shots, nested]}

    parsed = planner._parse_planner_payload(payload)
    assert len(parsed.shots) == len(shots)
    normalized = planner._normalize_agent_output(parsed, base.brief, [])
    assert sum(shot.duration_seconds for shot in normalized.shots) == 30
    assert all(4 <= shot.duration_seconds <= 15 for shot in normalized.shots)


def test_planner_derives_missing_shot_indices_from_array_order(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    base = planner._plan_heuristically(ProjectBrief(prompt="A palace story.", duration_seconds=30), [])
    payload = PlannerOutput(world_bible=base.world_bible, shots=base.shots).model_dump(mode="json")
    for shot in payload["shots"]:
        shot.pop("index")

    parsed = planner._parse_planner_payload(payload)

    assert [shot.index for shot in parsed.shots] == list(range(len(parsed.shots)))


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "7秒，紧接上一镜头的连续电影写实画面。太和殿中，孟子义向龙椅走近。",
            "太和殿中，孟子义向龙椅走近。",
        ),
        (
            "7 seconds. Continue directly from the previous shot in a continuous cinematic image. "
            "Meng Ziyi walks toward the throne.",
            "Meng Ziyi walks toward the throne.",
        ),
    ],
)
def test_planner_strips_duration_and_reference_video_boilerplate(prompt, expected):
    assert PlannerService._clean_generation_prompt(prompt) == expected


def test_planner_rejects_llm_output_without_required_anchor(settings):
    configured = replace(
        settings,
        text_to_image_provider="vllm-omni",
        text_to_image_base_url="http://text-to-image.test",
        text_to_image_model="Qwen/Qwen-Image-2512",
    )
    repository = StudioRepository(configured.database_path)
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(prompt="A creator enters a workshop.", duration_seconds=15)
    output = planner._plan_heuristically(brief, [])
    output.shots[0].anchor_prompt = ""

    with pytest.raises(ValueError, match="omitted anchor_prompt"):
        planner._normalize_agent_output(output, brief, [])


def test_planner_repairs_anchor_missing_ordered_asset_binding(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    asset = AssetService(configured, repository).ingest_stream(
        png_bytes("green"),
        "palace.png",
        "image/png",
        roles=[AssetRole.LOCATION],
    )
    asset = AssetService(configured, repository).update(
        asset.id,
        AssetUpdate(display_name="太和殿"),
    )
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(
        prompt="宫殿中的开场。",
        duration_seconds=15,
        reference_asset_ids=[asset.id],
    )
    output = planner._plan_heuristically(brief, [asset])
    output.shots[0].anchor_prompt = "太和殿中的一幅电影感开场静帧。"

    normalized = planner._normalize_agent_output(output, brief, [asset])
    prompt = normalized.shots[0].anchor_prompt
    assert "参考图1 太和殿" in prompt
    assert "太和殿中的一幅电影感开场静帧" in prompt


def test_planner_repairs_anchor_bindings_using_final_reference_order(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    first = assets.ingest_stream(png_bytes("red"), "first.png", "image/png", roles=[AssetRole.CHARACTER])
    second = assets.ingest_stream(png_bytes("blue"), "second.png", "image/png", roles=[AssetRole.CHARACTER])
    first = assets.update(first.id, AssetUpdate(display_name="第一角色"))
    second = assets.update(second.id, AssetUpdate(display_name="第二角色"))
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(
        prompt="Two characters meet.",
        duration_seconds=15,
        reference_asset_ids=[first.id, second.id],
    )
    output = planner._plan_heuristically(brief, [first, second])
    shot = output.shots[0]
    shot.reference_asset_ids = [second.id, first.id]
    shot.anchor_prompt = (
        f"Use {second.id} and {first.id} for the two character identities in one coherent opening composition."
    )

    normalized = planner._normalize_agent_output(output, brief, [first, second])
    prompt = normalized.shots[0].anchor_prompt
    assert "参考图1 第二角色" in prompt
    assert "参考图2 第一角色" in prompt
    assert prompt.index("参考图1 第二角色") < prompt.index("参考图2 第一角色")
    assert second.id not in prompt
    assert first.id not in prompt


def test_ultra_fast_later_shot_binds_previous_boundary_before_assets(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    actor = assets.ingest_stream(
        png_bytes("blue"),
        "actor.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    actor = assets.update(actor.id, AssetUpdate(display_name="女主角"))
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(
        prompt="女主角从街道进入咖啡馆。",
        duration_seconds=28,
        reference_asset_ids=[actor.id],
        continuation_mode=ContinuationMode.ULTRA_FAST,
    )
    output = planner._plan_heuristically(brief, [actor])
    for index, shot in enumerate(output.shots):
        shot.anchor_prompt = f"保留女主角身份，为第 {index + 1} 镜建立新的电影中景构图。"

    normalized = planner._normalize_agent_output(output, brief, [actor])
    second = normalized.shots[1]

    assert second.continuity_from_shot_id == normalized.shots[0].id
    assert "参考图1 上一镜末帧" in second.anchor_prompt
    assert "参考图2 女主角" in second.anchor_prompt
    assert second.anchor_prompt.index("参考图1") < second.anchor_prompt.index("参考图2")


def test_ultra_fast_without_materials_uses_t2i_then_previous_boundary_edit(settings):
    planner = PlannerService(settings, StudioRepository(settings.database_path))
    brief = ProjectBrief(
        prompt="一名侦探从雨夜街道走进明亮咖啡馆。",
        duration_seconds=28,
        continuation_mode=ContinuationMode.ULTRA_FAST,
    )
    output = planner._plan_heuristically(brief, [])
    for index, shot in enumerate(output.shots):
        shot.anchor_prompt = f"第 {index + 1} 镜的完整零秒构图。"

    normalized = planner._normalize_agent_output(output, brief, [])
    first, second = normalized.shots[:2]

    assert first.continuity_from_shot_id is None
    assert "参考图1" not in first.anchor_prompt
    assert second.continuity_from_shot_id == first.id
    assert "参考图1 上一镜末帧" in second.anchor_prompt
    assert "参考图2" not in second.anchor_prompt
    assert first.task == ShotTask.FL2VA
    assert second.task == ShotTask.FL2VA


def test_planner_anchor_binding_repair_preserves_budget_and_is_idempotent(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    asset = AssetService(configured, repository).ingest_stream(
        png_bytes("blue"),
        "hero.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    asset = AssetService(configured, repository).update(
        asset.id,
        AssetUpdate(display_name="Hero"),
    )
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(prompt="A hero enters.", duration_seconds=15, reference_asset_ids=[asset.id])
    output = planner._plan_heuristically(brief, [asset])
    body = "A coherent opening still with precise lighting, pose, lens, and spatial detail. " + ("detail. " * 220)
    output.shots[0].anchor_prompt = body

    normalized = planner._normalize_agent_output(output, brief, [asset])
    prompt = normalized.shots[0].anchor_prompt
    assert len(prompt) <= 1000
    assert "参考图1 Hero" in prompt
    assert prompt.count("参考图1 Hero") == 1

    normalized_again = planner._normalize_agent_output(normalized, brief, [asset])
    assert normalized_again.shots[0].anchor_prompt == prompt


def test_planner_anchor_binding_scrubs_unreadable_asset_ids(settings):
    configured = replace(
        settings,
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://image-edit.test",
        image_edit_model="Qwen-Image-Edit-2511",
    )
    repository = StudioRepository(configured.database_path)
    asset = AssetService(configured, repository).ingest_stream(
        png_bytes("blue"),
        "hero.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
    )
    asset = AssetService(configured, repository).update(asset.id, AssetUpdate(display_name="Hero"))
    planner = PlannerService(configured, repository)
    brief = ProjectBrief(prompt="A hero enters.", duration_seconds=15, reference_asset_ids=[asset.id])
    output = planner._plan_heuristically(brief, [asset])
    output.shots[0].anchor_prompt = f"Use {asset.id} for identity only in the opening still."

    normalized = planner._normalize_agent_output(output, brief, [asset])
    prompt = normalized.shots[0].anchor_prompt
    assert asset.id not in prompt
    assert "参考图1 Hero" in prompt


def test_compiler_snapshots_structured_storyboard_fields(settings):
    project = _anchor_mode_project()
    project.shots[0].audio_prompt = "Rain and footsteps."
    project.shots[0].music_prompt = "A restrained string motif."
    project.shots[0].dialogue = [DialogueLine(speaker="Lead", text="I am here.", language="English", delivery="quiet")]
    plan = FilmCompiler(_configured_image_edit(settings, "first-shot")).compile(project)
    stage = next(item for item in plan.stages if item.kind == "video" and item.shot_id == project.shots[0].id)

    assert stage.inputs["anchor_prompt"] == project.shots[0].anchor_prompt
    assert stage.inputs["audio_prompt"] == "Rain and footsteps."
    assert stage.inputs["music_prompt"] == "A restrained string motif."
    assert stage.inputs["dialogue"][0]["text"] == "I am here."


def test_compiler_accepts_h3_maximum_fifteen_second_shot(settings):
    project = _anchor_mode_project()
    project.shots[0].duration_seconds = 15
    FilmCompiler(settings).compile(project)


def test_planner_structured_schema_caps_h3_shots_at_fifteen_seconds():
    schema = PlannerService._planner_json_schema()
    assert schema["$defs"]["ShotSpec"]["properties"]["duration_seconds"]["maximum"] == 15


def test_imported_shots_receive_complete_llm_h3_timeline(settings, monkeypatch):
    configured = replace(settings, planner_base_url="http://planner.test/v1", planner_model="test-model")
    planner = PlannerService(configured, StudioRepository(configured.database_path))
    source = planner._plan_heuristically(ProjectBrief(prompt="A lantern rises over a river.", duration_seconds=15), [])
    imported = source.model_copy(deep=True)
    original = imported.shots[0]
    imported.shots[0] = original.model_copy(update={
        "opening_state": "", "ending_state": "", "continuity_handoff": "",
        "reference_anchors": [], "hook": "", "visual_beats": [],
    })

    schema_names = []

    async def fake_request_json(_self, _client, _prompt, payload, *, schema, schema_name):
        del payload, schema
        schema_names.append(schema_name)
        if schema_name == "nautilus_imported_h3_repair_1":
            return {"shot": original.model_dump(mode="json")}
        assert schema_name == "nautilus_imported_h3_1"
        result = original.model_dump(mode="json")
        result["visual_beats"][0]["end_seconds"] = 0
        return {"shot": result}

    monkeypatch.setattr(PlannerService, "_request_json", fake_request_json)
    enriched = asyncio.run(planner.enrich_imported_shots(imported))
    assert schema_names == ["nautilus_imported_h3_1", "nautilus_imported_h3_repair_1"]
    shot = enriched.shots[0]
    assert shot.id == imported.shots[0].id
    assert shot.opening_state and shot.ending_state and shot.continuity_handoff
    assert shot.reference_anchors and shot.hook and shot.visual_beats
    assert shot.visual_beats[0].start_seconds == 0
    assert shot.visual_beats[-1].end_seconds == shot.duration_seconds


def test_imported_shot_enrichment_accepts_bare_shot_list_response(settings):
    configured = replace(
        settings,
        planner_base_url="http://planner.test/v1",
        planner_model="test-model",
    )
    planner = PlannerService(configured, StudioRepository(configured.database_path))
    source = planner._plan_heuristically(
        ProjectBrief(prompt="A lantern rises over a river.", duration_seconds=15), []
    )
    imported = source.model_copy(deep=True)
    original = imported.shots[0]
    imported.shots[0] = original.model_copy(update={
        "opening_state": "", "ending_state": "", "continuity_handoff": "",
        "reference_anchors": [], "hook": "", "visual_beats": [],
    })
    shot_json = original.model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_payload = json.loads(body["messages"][1]["content"])
        assert user_payload["stage"] == "imported_shot_h3_enrichment"
        # The provider answers a single-shot request with a bare JSON array
        # instead of a ShotSpec object; the planner must recover, not abort.
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps([shot_json])}}]},
        )

    planner._transport = httpx.MockTransport(handler)
    enriched = asyncio.run(planner.enrich_imported_shots(imported))
    shot = enriched.shots[0]
    assert shot.id == imported.shots[0].id
    assert shot.opening_state and shot.ending_state and shot.continuity_handoff
    assert shot.reference_anchors and shot.hook and shot.visual_beats
    assert shot.visual_beats[0].start_seconds == 0
    assert shot.visual_beats[-1].end_seconds == shot.duration_seconds


def test_imported_shot_enrichment_falls_back_when_provider_returns_scalar(settings):
    configured = replace(
        settings,
        planner_base_url="http://planner.test/v1",
        planner_model="test-model",
    )
    planner = PlannerService(configured, StudioRepository(configured.database_path))
    source = planner._plan_heuristically(
        ProjectBrief(prompt="A lantern rises over a river.", duration_seconds=15), []
    )
    imported = source.model_copy(deep=True)
    original = imported.shots[0]
    imported.shots[0] = original.model_copy(update={
        "opening_state": "", "ending_state": "", "continuity_handoff": "",
        "reference_anchors": [], "hook": "", "visual_beats": [],
    })
    shot_json = original.model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_payload = json.loads(body["messages"][1]["content"])
        assert user_payload["stage"] == "imported_shot_h3_enrichment"
        # A scalar JSON payload is not recoverable by wrapping; the planner must
        # retain source blueprint fields instead of aborting the whole plan.
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(shot_json["prompt"])}}]},
        )

    planner._transport = httpx.MockTransport(handler)
    enriched = asyncio.run(planner.enrich_imported_shots(imported))
    shot = enriched.shots[0]
    assert shot.id == imported.shots[0].id
    assert shot.opening_state and shot.ending_state and shot.continuity_handoff
    assert shot.reference_anchors and shot.hook and shot.visual_beats
    assert shot.visual_beats[0].start_seconds == 0
    assert shot.visual_beats[-1].end_seconds == shot.duration_seconds


def test_planner_extracts_json_from_prose_wrapper():
    payload = PlannerService._json_text('Provider preface: {\"shot\": {\"index\": 0}} trailing note')
    assert json.loads(payload) == {"shot": {"index": 0}}


def test_imported_h3_enrichment_preserves_creator_prompt_and_beats(settings, monkeypatch):
    """H3 enrichment must be incremental: it must not overwrite prompt/beats."""
    configured = replace(settings, planner_base_url="http://planner.test/v1", planner_model="test-model")
    planner = PlannerService(configured, StudioRepository(configured.database_path))
    source = planner._plan_heuristically(ProjectBrief(prompt="A lantern rises over a river.", duration_seconds=15), [])
    imported = source.model_copy(deep=True)
    original = imported.shots[0]
    imported.shots[0] = original.model_copy(update={
        "opening_state": "", "ending_state": "", "continuity_handoff": "",
        "reference_anchors": [], "hook": "",
    })

    async def fake_request_json(_self, _client, _prompt, payload, *, schema, schema_name):
        del payload, schema
        result = original.model_dump(mode="json")
        result["prompt"] = "SHORT REWRITTEN PROMPT"
        result["visual_beats"] = [{"start_seconds": 0, "end_seconds": 15, "visual_action": "one beat"}]
        result["opening_state"] = "A stable opening frame."
        result["ending_state"] = "A stable closing frame."
        result["continuity_handoff"] = "Keep the lantern center frame."
        result["reference_anchors"] = ["The lantern on the river."]
        result["hook"] = "The river stills."
        return {"shot": result}

    monkeypatch.setattr(PlannerService, "_request_json", fake_request_json)
    enriched = asyncio.run(planner.enrich_imported_shots(imported))
    shot = enriched.shots[0]
    assert shot.prompt == original.prompt
    assert shot.visual_beats == original.visual_beats
    assert shot.audio_prompt == original.audio_prompt
    assert shot.opening_state == "A stable opening frame."
    assert shot.ending_state == "A stable closing frame."
    assert shot.continuity_handoff == "Keep the lantern center frame."
    assert shot.reference_anchors == ["The lantern on the river."]
    assert shot.hook == "The river stills."
