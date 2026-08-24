from dataclasses import replace

from fastapi.testclient import TestClient

from long_video_studio.app import create_app
from long_video_studio.config import PlannerProfile
from long_video_studio.domain import FilmProject, ProjectBrief, ShotSpec, StoryboardBeat, WorldBible
from long_video_studio.planner import PlannerService


def test_preproduction_is_visible_and_gates_rendering(settings, monkeypatch):
    app = create_app(settings)

    async def fake_h3_enrichment(project):
        shot = project.shots[0].model_copy(update={
            "opening_state": "A black title card fills the first frame.",
            "ending_state": "The title card remains black and stable.",
            "continuity_handoff": "Hold black screen and silent ambience.",
            "reference_anchors": ["Scene geography: black title card"],
            "hook": "A single title emerges from darkness.",
            "visual_beats": [StoryboardBeat(
                start_seconds=0, end_seconds=5,
                visual_action="Black title card holds steady.",
                state_change="The title settles into a readable final frame.",
                camera="Locked-off frame.", sound="Near silence.",
            )],
        })
        return project.model_copy(update={"shots": [shot]})

    monkeypatch.setattr(app.state.services.planner, "enrich_imported_shots", fake_h3_enrichment)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A black title card."),
            world_bible=WorldBible(logline="Title", visual_style="cinematic"),
            shots=[
                ShotSpec(
                    index=0,
                    title="Black title",
                    purpose="Set tone",
                    duration_seconds=5,
                    prompt="Black screen with a title",
                    source_section="剪辑与动作：本镜黑场字幕镜。",
                )
            ],
        )
    )
    with TestClient(app) as client:
        created = client.post(f"/api/projects/{project.id}/preproduction")
        assert created.status_code == 200
        assert created.json()["status"] == "awaiting_approval"
        assert created.json()["shot_plans"][0]["start_frame_source"] == "system_black"

        blocked = client.post(f"/api/projects/{project.id}/render")
        assert blocked.status_code == 409
        assert "preproduction" in blocked.json()["detail"]

        approved = client.post(f"/api/projects/{project.id}/preproduction/approve")
        assert approved.status_code == 200
        assert approved.json()["status"] == "ready"

        rendered = client.post(f"/api/projects/{project.id}/render")
        assert rendered.status_code == 409
        assert "MiniMax-H3" in rendered.json()["detail"]


def test_planner_profiles_api_never_exposes_key(settings):
    configured = replace(settings, planner_profiles=(
        PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "private-key", "qwen-plus", "chat_completions"),
    ))
    with TestClient(create_app(configured)) as client:
        response = client.get("/api/planner-profiles")

    assert response.status_code == 200
    assert response.json()["profiles"] == [{
        "id": "qwen", "display_name": "Qwen", "model": "qwen-plus",
        "wire_api": "chat_completions", "available": True,
    }]
    assert "private-key" not in response.text


def test_preproduction_uses_selected_planner_profile(settings, monkeypatch):
    configured = replace(settings, planner_profiles=(
        PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "private-key", "qwen-plus", "chat_completions"),
    ))
    app = create_app(configured)
    project = app.state.services.repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A black title card."),
            world_bible=WorldBible(logline="Title", visual_style="cinematic"),
            shots=[
                ShotSpec(
                    index=0,
                    title="Black title",
                    purpose="Set tone",
                    duration_seconds=5,
                    prompt="Black screen with a title",
                    source_section="剪辑与动作：本镜黑场字幕镜。",
                )
            ],
        )
    )
    captured: dict[str, str] = {}

    async def fake_enrich(_self, incoming, shot_ids=None):
        captured["model"] = _self.settings.planner_model or ""
        captured["base_url"] = _self.settings.planner_base_url or ""
        return incoming

    monkeypatch.setattr(PlannerService, "enrich_imported_shots", fake_enrich)
    with TestClient(app) as client:
        response = client.post(
            f"/api/projects/{project.id}/preproduction",
            json={"profile_id": "qwen", "model": "qwen-plus"},
        )

    assert response.status_code == 200
    assert captured["model"] == "qwen-plus"
    assert captured["base_url"] == "https://llm.example/v1"


def test_active_planner_switch_persists_and_applies(settings):
    configured = replace(settings, planner_profiles=(
        PlannerProfile("qwen", "Qwen", "https://llm.example/v1", "private-key", "qwen-plus", "chat_completions"),
    ))
    app = create_app(configured)
    with TestClient(app) as client:
        updated = client.put(
            "/api/planner-profile/active",
            json={"profile_id": "qwen", "model": "qwen-max"},
        )
        assert updated.status_code == 200
        assert updated.json()["profile_id"] == "qwen"
        assert updated.json()["resolved_model"] == "qwen-max"

        profiles = client.get("/api/planner-profiles")
        assert profiles.status_code == 200
        assert profiles.json()["active"]["profile_id"] == "qwen"
        assert profiles.json()["active"]["model"] == "qwen-max"

    reloaded = create_app(configured)
    assert reloaded.state.services.active_planner_profile_id == "qwen"
    assert reloaded.state.services.active_planner_model == "qwen-max"
    planner = reloaded.state.services.resolve_planner()
    assert planner.settings.planner_model == "qwen-max"
    assert planner.settings.planner_base_url == "https://llm.example/v1"
