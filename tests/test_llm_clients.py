from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from long_video_studio.app import create_app
from long_video_studio.config import PlannerProfile


def test_create_llm_client_persists_and_hides_key(settings):
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/llm-clients",
            json={
                "id": "deepseek",
                "display_name": "DeepSeek V3",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-secret",
                "model": "deepseek-chat",
                "wire_api": "chat_completions",
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()["client"]
        assert payload["id"] == "deepseek"
        assert payload["model"] == "deepseek-chat"
        assert "api_key" not in created.text

        listed = client.get("/api/llm-clients").json()
        assert any(item["id"] == "deepseek" and item["source"] == "db" for item in listed["clients"])
        assert "sk-secret" not in client.get("/api/llm-clients").text


def test_set_default_llm_client_routes_resolve_planner(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        client.post(
            "/api/llm-clients",
            json={
                "id": "qwen",
                "display_name": "Qwen",
                "base_url": "https://llm.example/v1",
                "api_key": "private-key",
                "model": "qwen-plus",
                "wire_api": "chat_completions",
            },
        )
        active = client.post("/api/llm-clients/qwen/default", json={"model": "qwen-max"})
        assert active.status_code == 200, active.text
        assert active.json()["profile_id"] == "qwen"
        assert active.json()["resolved_model"] == "qwen-max"

    planner = app.state.services.resolve_planner()
    assert planner.settings.planner_model == "qwen-max"
    assert planner.settings.planner_base_url == "https://llm.example/v1"


def test_delete_active_default_is_rejected(settings):
    with TestClient(create_app(settings)) as client:
        client.post(
            "/api/llm-clients",
            json={
                "id": "lo",
                "display_name": "Local",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "local-model",
                "wire_api": "chat_completions",
            },
        )
        client.post("/api/llm-clients/lo/default", json={"model": "local-model"})
        deleted = client.delete("/api/llm-clients/lo")
        assert deleted.status_code == 409
        assert "default" in deleted.json()["detail"]

        # A second db client can be deleted freely.
        client.post(
            "/api/llm-clients",
            json={"id": "lite", "display_name": "Lite", "base_url": "http://127.0.0.1:8000/v1", "model": "lite"},
        )
        removed = client.delete("/api/llm-clients/lite")
        assert removed.status_code == 200
        assert removed.json()["deleted"] is True


def test_service_status_reports_default_llm_client(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        client.post(
            "/api/llm-clients",
            json={
                "id": "qwen",
                "display_name": "Qwen",
                "base_url": "https://llm.example/v1",
                "api_key": "private-key",
                "model": "qwen-plus",
                "wire_api": "chat_completions",
            },
        )
        client.post("/api/llm-clients/qwen/default", json={"model": "qwen-max"})

        status = client.get("/api/services/status").json()
        planner = next(item for item in status["services"] if item["id"] == "planner")
        assert planner["provider"] == "Qwen"
        assert planner["model"] == "qwen-max"


def test_env_profiles_and_db_clients_coexist(settings):
    configured = replace(settings, planner_profiles=(
        PlannerProfile("env_qwen", "EnvQwen", "https://llm.example/v1", "k", "env-qwen", "chat_completions"),
    ))
    app = create_app(configured)
    with TestClient(app) as client:
        client.post(
            "/api/llm-clients",
            json={
                "id": "db_deep",
                "display_name": "Deep",
                "base_url": "https://api.example/v1",
                "api_key": "x",
                "model": "deep",
            },
        )
        listed = client.get("/api/llm-clients").json()["clients"]
        ids = {item["id"] for item in listed}
        assert "env_qwen" in ids
        assert "db_deep" in ids
        assert all(item["source"] in {"env", "db"} for item in listed)
