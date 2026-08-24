"""End-to-end proof that the preproduction LLM switch actually routes to the
selected planner profile (e.g. qwen) instead of the default deepseek model.

A real local OpenAI-compatible stub records the ``model`` field of every
request; the test fails if the preproduction enrichment uses anything other
than the selected profile model.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fastapi.testclient import TestClient

from long_video_studio.app import create_app
from long_video_studio.config import PlannerProfile, Settings
from long_video_studio.domain import FilmProject, ProjectBrief, ShotSpec, WorldBible


class _StubHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        body = json.loads(raw)
        self.__class__.requests.append({"path": self.path, "model": body.get("model")})
        shot = {
            "shot": {
                "index": 0,
                "title": "Black title",
                "purpose": "Set tone",
                "duration_seconds": 5,
                "prompt": "Black screen with a title",
                "audio_prompt": "Near silence",
                "music_prompt": "",
                "dialogue": [],
                "opening_state": "A black title card fills the first frame.",
                "ending_state": "The title card remains black and stable.",
                "continuity_handoff": "Hold black screen and silent ambience.",
                "reference_anchors": ["Scene geography: black title card"],
                "hook": "A single title emerges from darkness.",
                "visual_beats": [{
                    "start_seconds": 0,
                    "end_seconds": 5,
                    "visual_action": "Black title card holds steady.",
                    "state_change": "The title settles into a readable final frame.",
                    "camera": "Locked-off frame.",
                    "sound": "Near silence.",
                }],
                "negative_prompt": "",
                "camera": "Locked-off frame.",
                "continuity_in": {"active_subject_ids": [], "characters": [], "fixed_landmarks": [], "handoff": ""},
                "continuity_out": {"active_subject_ids": [], "characters": [], "fixed_landmarks": [], "handoff": ""},
            }
        }
        content = json.dumps(shot, ensure_ascii=False)
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: Any) -> None:
        return


def _start_stub() -> tuple[ThreadingHTTPServer, str]:
    _StubHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/v1"


def test_preproduction_switch_uses_qwen_not_default(settings: Settings) -> None:
    server, base_url = _start_stub()
    try:
        configured = replace(
            settings,
            planner_profiles=(
                PlannerProfile("default", "DeepSeek", base_url, "dsk", "deepseek-flash", "chat_completions"),
                PlannerProfile("qwen", "Qwen", base_url, "qw", "qwen-plus", "chat_completions"),
            ),
        )
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

        with TestClient(app) as client:
            response = client.post(
                f"/api/projects/{project.id}/preproduction",
                json={"profile_id": "qwen", "model": ""},
            )
            assert response.status_code == 200, response.text

        models = [item["model"] for item in _StubHandler.requests]
        assert models, "preproduction never called the LLM provider stub"
        assert all(model == "qwen-plus" for model in models), f"used wrong model(s): {models}"
    finally:
        server.shutdown()
        server.server_close()


def test_active_planner_switch_routes_sync_plan_to_selected_profile(settings: Settings) -> None:
    server, base_url = _start_stub()
    try:
        configured = replace(
            settings,
            planner_profiles=(
                PlannerProfile("default", "DeepSeek", base_url, "dsk", "deepseek-flash", "chat_completions"),
                PlannerProfile("qwen", "Qwen", base_url, "qw", "qwen-plus", "chat_completions"),
            ),
        )
        app = create_app(configured)
        services = app.state.services
        services.set_active_planner("qwen", "")
        assert services.active_planner_profile_id == "qwen"
        planner = services.resolve_planner()
        assert planner.settings.planner_base_url == base_url
        assert planner.settings.planner_model == "qwen-plus"
    finally:
        server.shutdown()
        server.server_close()


def test_h3_batch_switch_uses_qwen_not_default(settings: Settings) -> None:
    server, base_url = _start_stub()
    try:
        configured = replace(
            settings,
            planner_profiles=(
                PlannerProfile("default", "DeepSeek", base_url, "dsk", "deepseek-flash", "chat_completions"),
                PlannerProfile("qwen", "Qwen", base_url, "qw", "qwen-plus", "chat_completions"),
            ),
        )
        app = create_app(configured)
        manager = app.state.planning_manager
        project = app.state.services.repository.save_project(
            FilmProject(
                brief=ProjectBrief(prompt="Imported long script", duration_seconds=15),
                world_bible=WorldBible(logline="Imported", visual_style="cinematic"),
                shots=[
                    ShotSpec(
                        index=0,
                        title="Shot 1",
                        purpose="test",
                        duration_seconds=5,
                        prompt="A stable scene",
                        source_section="source",
                    )
                ],
            )
        )

        async def run() -> None:
            await manager.start_imported_h3(project.id, profile_id="qwen")
            await manager._tasks[project.id]

        asyncio.run(run())
        models = [item["model"] for item in _StubHandler.requests]
        assert models, "H3 batch never called the LLM provider stub"
        assert all(model == "qwen-plus" for model in models), f"used wrong model(s): {models}"
    finally:
        server.shutdown()
        server.server_close()
