from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from fastapi.testclient import TestClient

from long_video_studio.app import create_app

MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


def _rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any],
    *,
    request_id: int,
    headers: dict[str, str] | None = None,
):
    return client.post(
        "/mcp/",
        headers=headers or MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        },
    )


def _initialize(client: TestClient, *, headers: dict[str, str] | None = None) -> None:
    response = _rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "nautilus-test", "version": "1"},
        },
        request_id=1,
        headers=headers,
    )
    assert response.status_code == 200
    assert _response_json(response)["result"]["serverInfo"]["name"] == "Nautilus Studio"


def _response_json(response):
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
    return response.json()


def test_mcp_lists_creator_safe_tools_and_calls_projects(settings):
    with TestClient(create_app(settings)) as client:
        _initialize(client)
        listed = _rpc(client, "tools/list", {}, request_id=2)
        llms_txt = client.get("/llms.txt").text
        called = _rpc(
            client,
            "tools/call",
            {"name": "studio_list_projects", "arguments": {}},
            request_id=3,
        )

    assert listed.status_code == 200
    names = {tool["name"] for tool in _response_json(listed)["result"]["tools"]}
    assert {
        "studio_status",
        "studio_list_projects",
        "studio_get_project",
        "studio_list_assets",
        "studio_import_asset",
        "studio_plan_project",
        "studio_render_project",
        "studio_render_status",
        "studio_cancel_planning",
    } <= names
    assert all(f"`{name}" in llms_txt for name in names)
    assert called.status_code == 200
    assert _response_json(called)["result"]["structuredContent"]["result"] == []


def test_mcp_cancel_planning_awaits_manager(settings):
    with TestClient(create_app(settings)) as client:
        _initialize(client)
        response = _rpc(
            client,
            "tools/call",
            {
                "name": "studio_cancel_planning",
                "arguments": {"project_id": "project_missing"},
            },
            request_id=2,
        )

    assert response.status_code == 200
    assert _response_json(response)["result"]["structuredContent"] == {
        "project_id": "project_missing",
        "cancelled": False,
    }


def test_mcp_can_start_heuristic_storyboard_planning(settings):
    with TestClient(create_app(settings)) as client:
        _initialize(client)
        response = _rpc(
            client,
            "tools/call",
            {
                "name": "studio_plan_project",
                "arguments": {
                    "prompt": "A detective finds a letter in a midnight cafe.",
                    "title": "Midnight Letter",
                    "duration_seconds": 28,
                    "continuation_mode": "ultra_fast",
                    "ultra_fast_transition": "fade_black",
                },
            },
            request_id=2,
        )

        assert response.status_code == 200
        result = _response_json(response)["result"]["structuredContent"]
        project = client.get(f"/api/projects/{result['project_id']}")

    assert result["status"] == "planning"
    assert project.status_code == 200
    assert project.json()["brief"]["continuation_mode"] == "ultra_fast"


def test_mcp_bearer_token_protects_all_protocol_requests(settings):
    protected = replace(settings, mcp_token="test-secret")
    authorized_headers = {**MCP_HEADERS, "authorization": "Bearer test-secret"}

    with TestClient(create_app(protected)) as client:
        unauthorized = _rpc(client, "tools/list", {}, request_id=1)
        _initialize(client, headers=authorized_headers)
        authorized = _rpc(
            client,
            "tools/list",
            {},
            request_id=2,
            headers=authorized_headers,
        )

    assert unauthorized.status_code == 401
    assert _response_json(unauthorized) == {"error": "unauthorized"}
    assert authorized.status_code == 200
