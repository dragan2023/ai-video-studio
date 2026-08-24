from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from long_video_studio.domain import RenderJob
from long_video_studio.service_status import (
    GpuSnapshotReader,
    ServiceStatusCollector,
    parse_vllm_omni_metrics,
)


def test_parse_vllm_omni_metrics_extracts_request_activity():
    payload = """
# HELP vllm_omni:num_requests_running active requests
vllm_omni:num_requests_running{model_name="demo"} 2.0
vllm_omni:num_requests_waiting{model_name="demo"} 1.0
vllm_omni:requests_success_total{finished_reason="stop",model_name="demo"} 7.0
vllm_omni:requests_success_total{finished_reason="abort",model_name="demo"} 1.0
process_resident_memory_bytes 1.25e+09
unrelated_metric 99
"""

    assert parse_vllm_omni_metrics(payload) == {
        "requests_running": 2,
        "requests_waiting": 1,
        "requests_succeeded": 8,
        "process_resident_memory_bytes": 1.25e9,
    }


def test_gpu_snapshot_reader_validates_and_marks_stale(tmp_path):
    path = tmp_path / "gpu.json"
    captured = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gpu_service_snapshot",
                "captured_at": captured.isoformat(),
                "devices": [
                    {
                        "service_id": "fl2va",
                        "node": "video-node",
                        "index": 0,
                        "name": "MTT S5000",
                        "utilization_percent": 87,
                        "memory_used_mib": 64000,
                        "memory_total_mib": 81920,
                        "temperature_c": 61,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reader = GpuSnapshotReader(path, max_age_seconds=20, max_bytes=1024 * 1024)

    fresh = reader.read(captured + timedelta(seconds=4))
    stale = reader.read(captured + timedelta(seconds=21))

    assert fresh["state"] == "ready"
    assert fresh["age_seconds"] == 4
    assert fresh["devices"][0]["utilization_percent"] == 87
    assert stale["state"] == "stale"
    assert stale["age_seconds"] == 21


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps({"schema_version": 2, "captured_at": "2026-08-15T08:00:00Z"}).encode(),
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gpu_service_snapshot",
                "captured_at": "2026-08-15T08:00:00Z",
                "devices": [
                    {
                        "service_id": "fl2va",
                        "index": 0,
                        "utilization_percent": 101,
                    }
                ],
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gpu_service_snapshot",
                "captured_at": "2026-08-15T08:00:00Z",
                "devices": [
                    {
                        "service_id": "fl2va",
                        "index": 0,
                        "memory_used_mib": 9,
                        "memory_total_mib": 8,
                    }
                ],
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gpu_service_snapshot",
                "captured_at": "2026-08-15T08:00:00Z",
                "devices": [
                    {"service_id": "fl2va", "node": "video", "index": 0},
                    {"service_id": "fl2va", "node": "video", "index": 0},
                ],
            }
        ).encode(),
    ],
)
def test_gpu_snapshot_reader_rejects_invalid_samples(tmp_path, payload):
    path = tmp_path / "gpu.json"
    path.write_bytes(payload)

    result = GpuSnapshotReader(path, max_age_seconds=20, max_bytes=1024 * 1024).read()

    assert result["state"] == "invalid"
    assert result["devices"] == []


def test_gpu_snapshot_reader_rejects_oversized_file(tmp_path):
    path = tmp_path / "gpu.json"
    path.write_bytes(b"x" * 2048)

    result = GpuSnapshotReader(path, max_age_seconds=20, max_bytes=1024).read()

    assert result["state"] == "invalid"
    assert "exceeds" in result["error"]


def test_service_status_collector_combines_health_metrics_and_gpu(settings, tmp_path):
    snapshot_path = tmp_path / "gpu.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gpu_service_snapshot",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "devices": [
                    {
                        "service_id": "fl2va",
                        "node": "video-node",
                        "index": 0,
                        "name": "MTT S5000",
                        "utilization_percent": 82,
                        "memory_used_mib": 67000,
                        "memory_total_mib": 81920,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    configured = replace(
        settings,
        h3_fl2va_url="http://provider.test/fl2va",
        h3_ref2va_url="http://provider.test/ref2va",
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://provider.test/edit/v1",
        image_edit_model="Qwen/Image-Edit",
        text_to_image_provider="vllm-omni",
        text_to_image_base_url="http://provider.test/t2i/v1",
        gpu_snapshot_path=snapshot_path,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/metrics"):
            running = 1 if request.url.path.startswith("/ref2va") else 0
            return httpx.Response(
                200,
                text=(
                    f'vllm_omni:num_requests_running{{model_name="demo"}} {running}\n'
                    'vllm_omni:num_requests_waiting{model_name="demo"} 0\n'
                    'vllm_omni:requests_success_total{finished_reason="stop"} 3\n'
                ),
            )
        return httpx.Response(404)

    result = asyncio.run(
        ServiceStatusCollector(
            configured,
            transport=httpx.MockTransport(handler),
        ).collect(planning_project_ids=["project-1"], active_jobs=[])
    )

    services = {item["id"]: item for item in result["services"]}
    assert result["status"] == "ready"
    assert services["planner"]["state"] == "busy"
    assert services["fl2va"]["state"] == "busy"
    assert services["fl2va"]["gpu"]["utilization_percent"] == 82
    assert services["ref2va"]["state"] == "busy"
    assert services["ref2va"]["requests_running"] == 1
    assert services["image_edit"]["state"] == "ready"
    assert services["t2i"]["state"] == "ready"


def test_service_status_collector_reports_unreachable_without_exposing_endpoint(settings):
    configured = replace(settings, h3_fl2va_url="http://secret-host.invalid/path?token=secret")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = asyncio.run(
        ServiceStatusCollector(
            configured,
            transport=httpx.MockTransport(handler),
        ).collect(planning_project_ids=[], active_jobs=[])
    )

    fl2va = next(item for item in result["services"] if item["id"] == "fl2va")
    assert result["status"] == "degraded"
    assert fl2va["state"] == "unreachable"
    assert "secret-host" not in json.dumps(result)
    assert "token=secret" not in json.dumps(result)


def test_service_status_includes_studio_outer_queue_for_all_model_services(settings):
    configured = replace(
        settings,
        h3_fl2va_url="http://provider.test/fl2va",
        h3_ref2va_url="http://provider.test/ref2va",
        image_edit_provider="vllm-omni",
        image_edit_base_url="http://provider.test/edit/v1",
        image_edit_model="Qwen/Image-Edit",
        text_to_image_provider="vllm-omni",
        text_to_image_base_url="http://provider.test/t2i/v1",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/metrics"):
            return httpx.Response(
                200,
                text=(
                    'vllm_omni:num_requests_running{model_name="demo"} 1\n'
                    'vllm_omni:num_requests_waiting{model_name="demo"} 0\n'
                ),
            )
        return httpx.Response(404)

    service_ids = ("fl2va", "ref2va", "image_edit", "t2i")
    jobs = [
        RenderJob(project_id=f"{service_id}-{index}", status="running", current_service_id=service_id)
        for service_id in service_ids
        for index in range(2)
    ]
    result = asyncio.run(
        ServiceStatusCollector(
            configured,
            transport=httpx.MockTransport(handler),
        ).collect(planning_project_ids=[], active_jobs=jobs)
    )

    services = {item["id"]: item for item in result["services"]}
    for service_id in service_ids:
        service = services[service_id]
        assert service["studio_active_requests"] == 2
        assert service["requests_running"] == 1
        assert service["requests_waiting"] == 1
        assert service["state"] == "busy"
    assert {item["current_service_id"] for item in result["activity"]["render"]["jobs"]} == set(service_ids)


def test_service_status_reports_comfyui_backend_ready(settings, tmp_path):
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    configured = replace(
        settings,
        h3_backend="comfyui",
        comfyui_url="http://comfy.test",
        comfyui_workflow=workflow,
    )
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        return httpx.Response(404)

    result = asyncio.run(
        ServiceStatusCollector(
            configured,
            transport=httpx.MockTransport(handler),
        ).collect(planning_project_ids=[], active_jobs=[])
    )

    services = {item["id"]: item for item in result["services"]}
    for service_id in ("fl2va", "ref2va"):
        assert services[service_id]["provider"] == "comfyui"
        assert services[service_id]["model"] == "Work-Fisher MiniMax-H3"
        assert services[service_id]["state"] == "ready"
        assert services[service_id]["healthy"] is True
    assert requests == ["/system_stats", "/system_stats"]
