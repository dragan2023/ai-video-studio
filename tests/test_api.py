from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_assets import png_bytes

from long_video_studio.app import create_app
from long_video_studio.domain import (
    FilmProject,
    ProjectBrief,
    RenderJob,
    RenderObservation,
    ShotSpec,
    ShotTask,
    WorldBible,
)
from long_video_studio.planner import PlannerError
from long_video_studio.repository import StudioRepository
from long_video_studio.runner import RenderManager


def test_creator_flow_upload_plan_edit_compile(settings):
    client = TestClient(create_app(settings))
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["planner"] == "heuristic"
    assert health.json()["fl2va_healthy"] is False
    assert health.json()["ref2va_healthy"] is False

    upload = client.post(
        "/api/assets/upload",
        data={"tags": "hero,warm", "roles": "character,start_frame"},
        files=[("files", ("hero.png", png_bytes().read(), "image/png"))],
    )
    assert upload.status_code == 200
    asset = upload.json()[0]

    planned = client.post(
        "/api/projects/plan",
        json={
            "title": "A creator flow",
            "prompt": "A woman plays with her cat and the mood grows increasingly joyful.",
            "duration_seconds": 30,
            "reference_asset_ids": [asset["id"]],
            "quality": "draft",
        },
    )
    assert planned.status_code == 200
    project = planned.json()
    assert project["shots"]
    assert project["shots"][0]["start_frame_asset_id"] == asset["id"]

    first_shot = project["shots"][0]
    edited = client.patch(
        f"/api/projects/{project['id']}/shots/{first_shot['id']}",
        json={"duration_seconds": 10, "prompt": "A smoother opening shot."},
    )
    assert edited.status_code == 200
    assert edited.json()["shots"][0]["prompt"] == "A smoother opening shot."

    compiled = client.post(f"/api/projects/{project['id']}/compile")
    assert compiled.status_code == 200
    assert compiled.json()["stages"][-1]["kind"] == "assembly"
    render = client.post(f"/api/projects/{project['id']}/render")
    assert render.status_code == 409
    assert "STUDIO_H3_FL2VA_URL" in render.json()["detail"]
    assert client.get("/").status_code == 200


def test_service_status_api_keeps_optional_services_explicit(settings):
    client = TestClient(create_app(settings))

    response = client.get("/api/services/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["status"] == "ready"
    services = {item["id"]: item for item in payload["services"]}
    assert services["planner"]["state"] == "ready"
    assert services["fl2va"]["state"] == "unconfigured"
    assert services["ref2va"]["state"] == "unconfigured"
    assert services["image_edit"]["state"] == "unconfigured"
    assert services["t2i"]["state"] == "unconfigured"
    assert payload["gpu_telemetry"]["state"] == "not_configured"


def test_style_presets_api_exposes_canonical_h3_contracts(settings):
    client = TestClient(create_app(settings))
    response = client.get("/api/style-presets")

    assert response.status_code == 200
    values = {item["id"]: item for item in response.json()}
    assert {
        "cinematic",
        "documentary",
        "music_video",
        "commercial",
        "noir",
        "animation",
        "retro",
        "surreal",
    } <= set(values)
    assert "Palette:" in values["cinematic"]["instructions"]
    assert values["cinematic"]["negative_constraints"]


def test_asset_content_rejects_paths_outside_media_roots(settings):
    app = create_app(settings)
    client = TestClient(app)
    uploaded = client.post(
        "/api/assets/upload",
        files=[("files", ("hero.png", png_bytes().read(), "image/png"))],
    )
    assert uploaded.status_code == 200
    asset_id = uploaded.json()[0]["id"]
    assert client.get(f"/api/assets/{asset_id}/content").status_code == 200

    outside = settings.data_dir.parent / "outside.png"
    outside.write_bytes(png_bytes().read())
    asset = app.state.services.repository.get_asset(asset_id)
    assert asset is not None
    app.state.services.repository.save_asset(asset.model_copy(update={"stored_path": str(outside)}))

    assert client.get(f"/api/assets/{asset_id}/content").status_code == 404


def test_asset_delete_succeeds_when_unreferenced(settings):
    client = TestClient(create_app(settings))
    upload = client.post(
        "/api/assets/upload",
        files=[("files", ("unused.png", png_bytes().read(), "image/png"))],
    )
    asset_id = upload.json()[0]["id"]

    deleted = client.delete(f"/api/assets/{asset_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert all(asset["id"] != asset_id for asset in client.get("/api/assets").json())


def test_asset_delete_is_blocked_while_project_references_it(settings):
    client = TestClient(create_app(settings))
    upload = client.post(
        "/api/assets/upload",
        data={"roles": "start_frame"},
        files=[("files", ("used.png", png_bytes().read(), "image/png"))],
    )
    asset_id = upload.json()[0]["id"]
    planned = client.post(
        "/api/projects/plan",
        json={
            "title": "Referenced asset",
            "prompt": "A short scene anchored by the uploaded frame.",
            "duration_seconds": 15,
            "reference_asset_ids": [asset_id],
            "quality": "draft",
        },
    )
    assert planned.status_code == 200

    deleted = client.delete(f"/api/assets/{asset_id}")

    assert deleted.status_code == 409
    assert "仍被项目或分镜引用" in deleted.json()["detail"]


def test_project_delete_removes_jobs_and_outputs_but_preserves_assets(settings):
    app = create_app(settings)
    client = TestClient(app)
    upload = client.post(
        "/api/assets/upload",
        files=[("files", ("shared.png", png_bytes().read(), "image/png"))],
    )
    asset_id = upload.json()[0]["id"]
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A disposable project.", reference_asset_ids=[asset_id]),
            world_bible=WorldBible(logline="Disposable", visual_style="Natural"),
            shots=[],
        )
    )
    job = app.state.services.repository.save_job(RenderJob(project_id=project.id, status="complete", progress=1))
    observation = app.state.services.repository.save_render_observation(
        RenderObservation(
            source_key="preserved-after-project-delete",
            project_id=project.id,
            shot_id="shot-history",
            render_profile=settings.render_profile,
            task=ShotTask.FL2VA,
            continuation_mode="initial",
            aspect_ratio="16:9",
            duration_seconds=10,
            inference_steps=50,
            elapsed_seconds=396.2,
        )
    )
    output_dir = settings.output_dir / project.id
    output_dir.mkdir(parents=True)
    (output_dir / "final.mp4").write_bytes(b"video")

    response = client.delete(f"/api/projects/{project.id}")

    assert response.status_code == 200
    assert response.json()["project_id"] == project.id
    assert app.state.services.repository.get_project(project.id) is None
    assert app.state.services.repository.get_job(job.id) is None
    assert not output_dir.exists()
    assert app.state.services.repository.get_asset(asset_id) is not None
    assert observation.id in {
        item.id for item in app.state.services.repository.list_render_observations(settings.render_profile)
    }


def test_project_delete_rejects_active_render_job(settings):
    app = create_app(settings)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="An active project."),
            world_bible=WorldBible(logline="Active", visual_style="Natural"),
            shots=[],
        )
    )
    app.state.services.repository.save_job(RenderJob(project_id=project.id, status="running"))

    response = TestClient(app).delete(f"/api/projects/{project.id}")

    assert response.status_code == 409
    assert app.state.services.repository.get_project(project.id) is not None


def test_async_planner_accepts_multiple_projects_before_either_finishes(settings, monkeypatch):
    app = create_app(settings)

    async def slow_plan(brief, project_id=None):
        await asyncio.sleep(0.08)
        project = app.state.services.repository.get_project(project_id)
        assert project is not None
        project.status = "planned"
        app.state.services.repository.save_project(project)
        return project

    monkeypatch.setattr(app.state.planning_manager.planner, "plan", slow_plan)
    with TestClient(app) as client:
        started = time.monotonic()
        first = client.post(
            "/api/projects/plan-async",
            json={"title": "Concurrent one", "prompt": "First background plan."},
        )
        second = client.post(
            "/api/projects/plan-async",
            json={"title": "Concurrent two", "prompt": "Second background plan."},
        )
        accepted_in = time.monotonic() - started
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["id"] != second.json()["id"]
        assert accepted_in < 0.08
        assert set(client.get("/api/planning/active").json()) == {
            first.json()["id"],
            second.json()["id"],
        }
        time.sleep(0.14)
        assert client.get(f"/api/projects/{first.json()['id']}").json()["status"] == "planned"
        assert client.get(f"/api/projects/{second.json()['id']}").json()["status"] == "planned"
        assert client.get("/api/planning/active").json() == []


def test_app_startup_marks_interrupted_planning_project_failed(settings):
    repository = StudioRepository(settings.database_path)
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="Interrupted planning."),
            world_bible=WorldBible(logline="Interrupted", visual_style="Natural"),
            shots=[],
            status="planning",
        )
    )

    app = create_app(settings)
    recovered = app.state.services.repository.get_project(project.id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.planner_trace[-1].status == "failed"
    assert "重启" in (recovered.planner_trace[-1].error or "")


def test_zero_material_render_requires_text_to_image_endpoint(settings):
    configured = replace(settings, h3_fl2va_url="http://fl2va.test")
    app = create_app(configured)
    shot = ShotSpec(
        index=0,
        title="Zero material opening",
        purpose="Establish a new world",
        prompt="A creator enters a sunlit empty studio.",
        anchor_prompt="A complete cinematic 16:9 opening still in a sunlit empty studio.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        reference_asset_ids=[],
    )
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A new studio.", reference_asset_ids=[]),
            world_bible=WorldBible(logline="Arrival", visual_style="cinematic"),
            shots=[shot],
            status="planned",
        )
    )

    with TestClient(app) as client:
        response = client.post(f"/api/projects/{project.id}/render")

    assert response.status_code == 409
    assert "STUDIO_T2I_BASE_URL" in response.json()["detail"]


def test_zero_material_render_accepts_configured_text_to_image_endpoint(settings):
    configured = replace(
        settings,
        h3_fl2va_url="http://fl2va.test",
        text_to_image_provider="vllm-omni",
        text_to_image_base_url="http://t2i.test",
        text_to_image_model="Qwen/Qwen-Image-2512",
    )
    app = create_app(configured)
    shot = ShotSpec(
        index=0,
        title="Zero material opening",
        purpose="Establish a new world",
        prompt="A creator enters a sunlit empty studio.",
        anchor_prompt="A complete cinematic 16:9 opening still in a sunlit empty studio.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        reference_asset_ids=[],
    )
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A new studio.", reference_asset_ids=[]),
            world_bible=WorldBible(logline="Arrival", visual_style="cinematic"),
            shots=[shot],
            status="planned",
        )
    )

    with TestClient(app) as client:
        response = client.post(f"/api/projects/{project.id}/render")

    assert response.status_code == 200


def test_render_force_query_is_forwarded_to_runner(settings, monkeypatch):
    app = create_app(settings)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="Render this project again."),
            world_bible=WorldBible(logline="Again", visual_style="Natural"),
            shots=[],
            status="complete",
        )
    )
    captured: dict[str, object] = {}

    def fake_submit(project_id: str, *, force: bool = False) -> RenderJob:
        captured.update(project_id=project_id, force=force)
        return RenderJob(project_id=project_id, force_rerender=force)

    monkeypatch.setattr(app.state.render_manager, "submit", fake_submit)

    response = TestClient(app).post(f"/api/projects/{project.id}/render?force=true")

    assert response.status_code == 200
    assert captured == {"project_id": project.id, "force": True}
    assert response.json()["force_rerender"] is True


def test_failed_planning_keeps_a_recoverable_project_draft(settings, monkeypatch):
    app = create_app(settings)

    async def fail_plan(brief, project_id=None):
        raise PlannerError("temporary planner failure")

    monkeypatch.setattr(app.state.services.planner, "plan", fail_plan)
    client = TestClient(app)

    response = client.post(
        "/api/projects/plan",
        json={
            "title": "Recoverable draft",
            "prompt": "A creator crosses a quiet station before dawn.",
            "duration_seconds": 30,
        },
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["message"] == "temporary planner failure"
    project_id = detail["project_id"]
    draft = client.get(f"/api/projects/{project_id}")
    assert draft.status_code == 200
    assert draft.json()["id"] == project_id
    assert draft.json()["status"] == "failed"
    assert draft.json()["shots"] == []
    assert any(project["id"] == project_id for project in client.get("/api/projects").json())


def test_render_endpoint_schedules_background_job_on_event_loop(settings, monkeypatch):
    async def no_op_run(self: RenderManager, job_id: str) -> None:
        return None

    monkeypatch.setattr(RenderManager, "_run", no_op_run)
    configured = replace(settings, h3_fl2va_url="http://fl2va.test")
    client = TestClient(create_app(configured))
    upload = client.post(
        "/api/assets/upload",
        data={"roles": "start_frame"},
        files=[("files", ("hero.png", png_bytes().read(), "image/png"))],
    )
    assert upload.status_code == 200
    asset_id = upload.json()[0]["id"]
    planned = client.post(
        "/api/projects/plan",
        json={
            "title": "Async render regression",
            "prompt": "A woman and a cat share a joyful moment.",
            "duration_seconds": 30,
            "reference_asset_ids": [asset_id],
            "quality": "draft",
        },
    )
    assert planned.status_code == 200

    response = client.post(f"/api/projects/{planned.json()['id']}/render")

    assert response.status_code == 200
    assert response.json()["status"] in {"queued", "running"}
    latest = client.get(f"/api/projects/{planned.json()['id']}/jobs/latest")
    assert latest.status_code == 200
    assert latest.json()["id"] == response.json()["id"]


def test_render_preflight_accepts_previous_clip_as_ref2va_continuation_input(settings, monkeypatch):
    async def no_op_run(self: RenderManager, job_id: str) -> None:
        return None

    monkeypatch.setattr(RenderManager, "_run", no_op_run)
    configured = replace(
        settings,
        h3_fl2va_url="http://fl2va.test",
        h3_ref2va_url="http://ref2va.test",
    )
    app = create_app(configured)
    client = TestClient(app)
    upload = client.post(
        "/api/assets/upload",
        data={"roles": "start_frame"},
        files=[("files", ("hero.png", png_bytes().read(), "image/png"))],
    )
    start_id = upload.json()[0]["id"]
    first = ShotSpec(
        index=0,
        title="Opening",
        purpose="Open",
        duration_seconds=7.5,
        task=ShotTask.FL2VA,
        prompt="Open on the creator.",
        start_frame_asset_id=start_id,
    )
    continuation = ShotSpec(
        index=1,
        title="Continue",
        purpose="Continue",
        duration_seconds=7.5,
        task=ShotTask.FL2VA,
        prompt="Continue with the next action.",
        continuity_from_shot_id=first.id,
    )
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A creator continues one flowing action."),
            world_bible=WorldBible(logline="One action", visual_style="realistic"),
            shots=[first, continuation],
        )
    )

    response = client.post(f"/api/projects/{project.id}/render")

    assert response.status_code == 200


def test_render_preflight_keeps_ordinary_ref2va_asset_contract(settings):
    configured = replace(settings, h3_ref2va_url="http://ref2va.test")
    app = create_app(configured)
    shot = ShotSpec(
        index=0,
        title="Asset reference",
        purpose="Use explicit references",
        duration_seconds=14,
        task=ShotTask.REF2VA,
        prompt="Animate the supplied reference media.",
    )
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="Animate creator-provided reference media."),
            world_bible=WorldBible(logline="Reference", visual_style="realistic"),
            shots=[shot],
        )
    )

    response = TestClient(app).post(f"/api/projects/{project.id}/render")

    assert response.status_code == 409
    assert "image plus audio/video references for shot 1" in response.json()["detail"]


def test_project_and_shot_dialog_updates_persist_and_invalidate_old_take(settings):
    client = TestClient(create_app(settings))
    planned = client.post(
        "/api/projects/plan",
        json={
            "title": "Dialog editing",
            "prompt": "A creator walks through a warm studio and finds a glowing prop.",
            "duration_seconds": 30,
        },
    )
    assert planned.status_code == 200
    project = planned.json()
    shot = project["shots"][0]

    project_update = client.patch(
        f"/api/projects/{project['id']}",
        json={
            "brief": {
                "title": "Edited title",
                "style": "A hand-crafted 16mm look",
                "style_preset": "custom",
                "style_instructions": "Soft tungsten pools, patient camera, tactile grain.",
                "continuation_mode": "quality",
            },
            "world_bible": {
                "logline": "The glowing prop changes the room's mood.",
                "character_notes": ["The creator wears a blue jacket."],
                "location_notes": ["A compact workshop at dusk."],
            },
        },
    )

    assert project_update.status_code == 200
    updated = project_update.json()
    assert updated["brief"]["title"] == "Edited title"
    assert updated["brief"]["style_instructions"].startswith("Soft tungsten")
    assert updated["brief"]["continuation_mode"] == "quality"
    assert updated["world_bible"]["character_notes"] == ["The creator wears a blue jacket."]

    shot_update = client.patch(
        f"/api/projects/{project['id']}/shots/{shot['id']}",
        json={
            "title": "Edited opening",
            "purpose": "Introduce the prop before the reveal.",
            "anchor_prompt": "A zero-second still of the creator facing the glowing prop.",
            "prompt": "A slow push-in toward the glowing prop, no jump cut.",
            "audio_prompt": "A low electrical hum and soft footsteps.",
            "music_prompt": "A sparse, audience-only cello texture.",
            "opening_state": "The creator stands at the workshop threshold.",
            "ending_state": "The creator settles beside the glowing prop.",
            "continuity_handoff": "Keep the blue jacket, warm light, prop position, and room tone stable.",
            "reference_anchors": [
                "Character identity: creator in a blue jacket",
                "Prop identity: the same glowing object",
            ],
            "hook": "The prop brightens as the creator reaches it.",
            "visual_beats": [
                {
                    "start_seconds": 0,
                    "end_seconds": 4,
                    "visual_action": "The creator approaches the prop.",
                    "state_change": "The distance closes.",
                    "camera": "Small-amplitude slow Push In.",
                    "sound": "Soft footsteps synchronize with each step.",
                },
                {
                    "start_seconds": 4,
                    "end_seconds": 8,
                    "visual_action": "The creator stops beside the prop.",
                    "state_change": "The prop becomes the visual focus.",
                    "camera": "The camera brakes into a Static Shot.",
                    "sound": "The final footstep decays into the electrical hum.",
                },
            ],
            "dialogue": [
                {
                    "speaker": "Creator",
                    "text": "There you are.",
                    "language": "English",
                    "delivery": "quietly relieved",
                    "start_seconds": 2,
                    "end_seconds": 4,
                }
            ],
            "negative_prompt": "text, logo, watermark",
            "duration_seconds": 8,
            "inference_steps": 50,
            "start_frame_asset_id": None,
            "continuation_mode": "fast",
        },
    )

    assert shot_update.status_code == 200
    edited_shot = shot_update.json()["shots"][0]
    assert edited_shot["title"] == "Edited opening"
    assert edited_shot["duration_seconds"] == 8
    assert edited_shot["start_frame_asset_id"] is None
    assert edited_shot["continuation_mode"] == "fast"
    assert edited_shot["anchor_prompt"].startswith("A zero-second still")
    assert edited_shot["audio_prompt"].startswith("A low electrical hum")
    assert edited_shot["music_prompt"].startswith("A sparse")
    assert edited_shot["opening_state"].startswith("The creator stands")
    assert edited_shot["ending_state"].startswith("The creator settles")
    assert len(edited_shot["reference_anchors"]) == 2
    assert edited_shot["hook"].startswith("The prop brightens")
    assert edited_shot["visual_beats"][-1]["end_seconds"] == 8
    assert edited_shot["dialogue"][0]["text"] == "There you are."
    assert edited_shot["status"] == "planned"
    assert edited_shot["selected_take_path"] is None
    assert edited_shot["render_started_at"] is None
    assert edited_shot["render_completed_at"] is None
    assert edited_shot["render_duration_seconds"] is None
    assert shot_update.json()["timeline"][1]["start_seconds"] == 8


def test_shot_dialogue_validation_rejects_invalid_lines(settings):
    client = TestClient(create_app(settings))
    project = client.post(
        "/api/projects/plan",
        json={"title": "Dialogue validation", "prompt": "A short conversation.", "duration_seconds": 15},
    ).json()
    shot = project["shots"][0]

    empty_speaker = client.patch(
        f"/api/projects/{project['id']}/shots/{shot['id']}",
        json={"dialogue": [{"speaker": "", "text": "Hello."}]},
    )
    invalid_timing = client.patch(
        f"/api/projects/{project['id']}/shots/{shot['id']}",
        json={
            "dialogue": [
                {
                    "speaker": "Lead",
                    "text": "Hello.",
                    "start_seconds": 5,
                    "end_seconds": 2,
                }
            ]
        },
    )
    past_shot = client.patch(
        f"/api/projects/{project['id']}/shots/{shot['id']}",
        json={
            "dialogue": [
                {
                    "speaker": "Lead",
                    "text": "Hello.",
                    "start_seconds": 1,
                    "end_seconds": shot["duration_seconds"] + 1,
                }
            ]
        },
    )

    assert empty_speaker.status_code == 422
    assert invalid_timing.status_code == 422
    assert past_shot.status_code == 422


def test_completed_video_is_inline_unless_download_is_requested(settings):
    app = create_app(settings)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A short preview film."),
            world_bible=WorldBible(logline="Preview", visual_style="Natural"),
            shots=[],
        )
    )
    output_path = settings.output_dir / project.id / "final.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"fake mp4")
    job = app.state.services.repository.save_job(
        RenderJob(
            project_id=project.id,
            status="complete",
            progress=1,
            output_path=str(output_path),
        )
    )
    client = TestClient(app)

    preview = client.get(f"/api/jobs/{job.id}/output")
    download = client.get(f"/api/jobs/{job.id}/output?download=true")

    assert preview.status_code == 200
    assert preview.headers.get("content-disposition", "").startswith("inline")
    assert download.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment")


def test_outputs_do_not_expose_absolute_paths(settings):
    app = create_app(settings)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A short preview film."),
            world_bible=WorldBible(logline="Preview", visual_style="Natural"),
            shots=[],
        )
    )
    output_path = settings.output_dir / project.id / "final.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"video")
    app.state.services.repository.save_job(
        RenderJob(
            project_id=project.id,
            status="complete",
            progress=1,
            output_path=str(output_path),
        )
    )

    response = TestClient(app).get("/api/outputs", params={"project_id": project.id})

    assert response.status_code == 200
    assert response.json() == [{"name": "final.mp4", "size_bytes": 5}]


def test_completed_video_cannot_escape_project_output_directory(settings, tmp_path):
    app = create_app(settings)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A short preview film."),
            world_bible=WorldBible(logline="Preview", visual_style="Natural"),
            shots=[],
        )
    )
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not a project output")
    job = app.state.services.repository.save_job(
        RenderJob(
            project_id=project.id,
            status="complete",
            progress=1,
            output_path=str(outside),
        )
    )

    response = TestClient(app).get(f"/api/jobs/{job.id}/output")

    assert response.status_code == 404


def test_react_web_root_serves_vite_assets(settings, tmp_path):
    assets = tmp_path / "web" / "assets"
    assets.mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text(
        '<script type="module" src="/assets/index-test.js"></script>', encoding="utf-8"
    )
    (assets / "index-test.js").write_text("export const ready = true;\n", encoding="utf-8")
    app = create_app(replace(settings, web_root=tmp_path / "web"))
    client = TestClient(app)

    assert client.get("/").status_code == 200
    asset = client.get("/assets/index-test.js")
    assert asset.status_code == 200
    assert "ready" in asset.text


def test_app_requires_a_built_react_ui(settings):
    with pytest.raises(RuntimeError, match="STUDIO_WEB_ROOT is required"):
        create_app(replace(settings, web_root=None))


def test_app_rejects_web_root_without_index(settings, tmp_path):
    empty_web_root = tmp_path / "empty-web"
    empty_web_root.mkdir()
    with pytest.raises(RuntimeError, match="missing index.html"):
        create_app(replace(settings, web_root=empty_web_root))
