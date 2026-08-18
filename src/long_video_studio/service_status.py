from __future__ import annotations

import asyncio
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from long_video_studio.config import Settings

ServiceState = Literal["ready", "busy", "queued", "unconfigured", "unreachable", "error", "unknown"]


class GpuDeviceTelemetry(BaseModel):
    """Provider-neutral GPU sample supplied by an operator-owned collector."""

    model_config = ConfigDict(extra="ignore")

    service_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    node: str | None = Field(default=None, max_length=96)
    index: int = Field(ge=0)
    name: str | None = Field(default=None, max_length=96)
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    memory_used_mib: int | None = Field(default=None, ge=0)
    memory_total_mib: int | None = Field(default=None, ge=0)
    temperature_c: float | None = None
    power_w: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_memory(self) -> GpuDeviceTelemetry:
        if (
            self.memory_used_mib is not None
            and self.memory_total_mib is not None
            and self.memory_used_mib > self.memory_total_mib
        ):
            raise ValueError("memory_used_mib cannot exceed memory_total_mib")
        return self


class GpuServiceSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1]
    kind: Literal["gpu_service_snapshot"]
    captured_at: datetime
    devices: list[GpuDeviceTelemetry] = Field(min_length=1)

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> GpuServiceSnapshot:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        identities = [(item.service_id, item.node, item.index) for item in self.devices]
        if len(identities) != len(set(identities)):
            raise ValueError("GPU snapshot contains duplicate service/node/device entries")
        return self


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_error(error: BaseException, limit: int = 240) -> str:
    value = " ".join(str(error).split())
    return value[:limit]


def _service_root(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in (
        "/v1/videos/sync",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1",
    ):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\n]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$",
    re.MULTILINE,
)


def parse_vllm_omni_metrics(payload: str) -> dict[str, float | int | None]:
    """Extract the small stable metric subset used by the creator UI."""

    values: dict[str, list[float]] = {}
    for match in _METRIC_LINE.finditer(payload):
        name = match.group("name")
        if name not in {
            "vllm_omni:num_requests_running",
            "vllm_omni:num_requests_waiting",
            "vllm_omni:requests_success_total",
            "process_resident_memory_bytes",
        }:
            continue
        value = float(match.group("value"))
        if math.isfinite(value):
            values.setdefault(name, []).append(value)
    return {
        "requests_running": int(sum(values.get("vllm_omni:num_requests_running", []))),
        "requests_waiting": int(sum(values.get("vllm_omni:num_requests_waiting", []))),
        "requests_succeeded": int(sum(values.get("vllm_omni:requests_success_total", []))),
        "process_resident_memory_bytes": (
            max(values["process_resident_memory_bytes"]) if values.get("process_resident_memory_bytes") else None
        ),
    }


class GpuSnapshotReader:
    def __init__(self, path: Path | None, *, max_age_seconds: float, max_bytes: int):
        self.path = path
        self.max_age_seconds = max_age_seconds
        self.max_bytes = max_bytes

    def read(self, now: datetime | None = None) -> dict[str, Any]:
        checked_at = now or _utc_now()
        if self.path is None:
            return {
                "configured": False,
                "state": "not_configured",
                "checked_at": checked_at.isoformat(),
                "age_seconds": None,
                "devices": [],
                "error": None,
            }
        try:
            size = self.path.stat().st_size
            if size > self.max_bytes:
                raise ValueError(f"GPU snapshot exceeds {self.max_bytes} bytes")
            raw = self.path.read_bytes()
            snapshot = GpuServiceSnapshot.model_validate_json(raw)
        except FileNotFoundError:
            return self._failure("missing", checked_at, "GPU snapshot file is missing")
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
            return self._failure("invalid", checked_at, _safe_error(error))

        captured_at = snapshot.captured_at.astimezone(timezone.utc)
        age_seconds = max(0.0, (checked_at.astimezone(timezone.utc) - captured_at).total_seconds())
        stale = age_seconds > self.max_age_seconds
        return {
            "configured": True,
            "state": "stale" if stale else "ready",
            "checked_at": checked_at.isoformat(),
            "captured_at": captured_at.isoformat(),
            "age_seconds": round(age_seconds, 3),
            "devices": [item.model_dump(mode="json") for item in snapshot.devices],
            "error": None,
        }

    def _failure(self, state: str, checked_at: datetime, error: str) -> dict[str, Any]:
        return {
            "configured": True,
            "state": state,
            "checked_at": checked_at.isoformat(),
            "age_seconds": None,
            "devices": [],
            "error": error,
        }


class ServiceStatusCollector:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self.transport = transport
        self.gpu_snapshots = GpuSnapshotReader(
            settings.gpu_snapshot_path,
            max_age_seconds=settings.gpu_snapshot_max_age_seconds,
            max_bytes=settings.gpu_snapshot_max_bytes,
        )

    async def collect(
        self,
        *,
        planning_project_ids: list[str],
        active_jobs: list[Any],
    ) -> dict[str, Any]:
        checked_at = _utc_now()
        specs = self._service_specs()
        timeout = httpx.Timeout(self.settings.service_probe_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            probed = await asyncio.gather(*(self._probe(spec, client, checked_at) for spec in specs))

        gpu = self.gpu_snapshots.read(checked_at)
        active_service_counts: dict[str, int] = {}
        for job in active_jobs:
            service_id = getattr(job, "current_service_id", None)
            if job.status == "running" and service_id:
                active_service_counts[service_id] = active_service_counts.get(service_id, 0) + 1
        devices_by_service: dict[str, list[dict[str, Any]]] = {}
        for device in gpu["devices"]:
            devices_by_service.setdefault(str(device["service_id"]), []).append(device)
        telemetry_fresh = gpu["state"] == "ready"
        for service in probed:
            studio_active = active_service_counts.get(service["id"], 0)
            service["studio_active_requests"] = studio_active
            engine_running = service.get("requests_running")
            engine_waiting = service.get("requests_waiting")
            if engine_running is not None:
                outer_waiting = max(0, studio_active - int(engine_running))
                service["requests_waiting"] = max(int(engine_waiting or 0), outer_waiting)
            service_gpu = self._gpu_summary(devices_by_service.get(service["id"], []), gpu)
            service["gpu"] = service_gpu
            if (
                telemetry_fresh
                and service_gpu
                and (service_gpu.get("utilization_percent") or 0) >= 5
                and service["state"] == "ready"
            ):
                service["state"] = "busy"
            if studio_active and service["state"] == "ready":
                service["state"] = "busy"

        planner = self._planner_status(planning_project_ids, checked_at)
        services = [planner, *probed]
        render_jobs = [
            {
                "id": item.id,
                "project_id": item.project_id,
                "status": item.status,
                "progress": item.progress,
                "current_shot_id": item.current_shot_id,
                "current_service_id": item.current_service_id,
                "message": item.message,
            }
            for item in active_jobs
        ]
        readiness = (
            "degraded"
            if any(item["configured"] and item["state"] in {"unreachable", "error"} for item in services)
            else "ready"
        )
        return {
            "schema_version": 1,
            "status": readiness,
            "checked_at": checked_at.isoformat(),
            "services": services,
            "activity": {
                "planning": {
                    "active_count": len(planning_project_ids),
                    "project_ids": planning_project_ids,
                },
                "render": {
                    "active_count": len(render_jobs),
                    "running_count": sum(item["status"] == "running" for item in render_jobs),
                    "queued_count": sum(item["status"] == "queued" for item in render_jobs),
                    "max_concurrency": self.settings.render_max_concurrency,
                    "profile": self.settings.render_profile,
                    "jobs": render_jobs,
                },
            },
            "gpu_telemetry": {key: value for key, value in gpu.items() if key != "devices"},
        }

    def _service_specs(self) -> list[dict[str, Any]]:
        image_edit_configured = bool(
            self.settings.image_edit_provider not in {"", "disabled", "none"}
            and self.settings.image_edit_base_url
            and self.settings.image_edit_model
        )
        t2i_configured = bool(
            self.settings.text_to_image_provider not in {"", "disabled", "none"}
            and self.settings.text_to_image_base_url
        )
        return [
            {
                "id": "fl2va",
                "display_name": "MiniMax-H3 FL2VA",
                "kind": "video",
                "provider": "vllm-omni",
                "model": "MiniMax-H3 / FL2VA",
                "configured": bool(self.settings.h3_fl2va_url),
                "endpoint": self.settings.h3_fl2va_url,
            },
            {
                "id": "ref2va",
                "display_name": "MiniMax-H3 Ref2VA",
                "kind": "video",
                "provider": "vllm-omni",
                "model": "MiniMax-H3 / Ref2VA",
                "configured": bool(self.settings.h3_ref2va_url),
                "endpoint": self.settings.h3_ref2va_url,
            },
            {
                "id": "image_edit",
                "display_name": "Qwen Image Edit",
                "kind": "image_edit",
                "provider": self.settings.image_edit_provider,
                "model": self.settings.image_edit_model,
                "configured": image_edit_configured,
                "endpoint": self.settings.image_edit_base_url,
            },
            {
                "id": "t2i",
                "display_name": "Qwen Image",
                "kind": "text_to_image",
                "provider": self.settings.text_to_image_provider,
                "model": self.settings.text_to_image_model,
                "configured": t2i_configured,
                "endpoint": self.settings.text_to_image_base_url,
            },
        ]

    async def _probe(
        self,
        spec: dict[str, Any],
        client: httpx.AsyncClient,
        checked_at: datetime,
    ) -> dict[str, Any]:
        result = {
            key: value
            for key, value in spec.items()
            if key in {"id", "display_name", "kind", "provider", "model", "configured"}
        }
        result.update(
            {
                "state": "unconfigured" if not spec["configured"] else "unknown",
                "healthy": None if not spec["configured"] else False,
                "checked_at": checked_at.isoformat(),
                "latency_ms": None,
                "http_status": None,
                "requests_running": None,
                "requests_waiting": None,
                "requests_succeeded": None,
                "error": None,
            }
        )
        if not spec["configured"]:
            return result

        root = _service_root(str(spec["endpoint"]))
        started = time.perf_counter()
        try:
            response = await client.get(f"{root}/health")
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            result["http_status"] = response.status_code
            if response.status_code >= 400:
                result["state"] = "error"
                result["error"] = f"health probe returned HTTP {response.status_code}"
                return result
            result["healthy"] = True
            result["state"] = "ready"
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            result["state"] = "unreachable"
            result["error"] = _safe_error(error) or error.__class__.__name__
            return result
        except httpx.HTTPError as error:
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            result["state"] = "error"
            result["error"] = _safe_error(error) or error.__class__.__name__
            return result

        try:
            metrics = await client.get(f"{root}/metrics")
            if metrics.status_code < 400:
                parsed = parse_vllm_omni_metrics(metrics.text)
                result.update(
                    {
                        key: parsed[key]
                        for key in (
                            "requests_running",
                            "requests_waiting",
                            "requests_succeeded",
                        )
                    }
                )
                if (result["requests_running"] or 0) > 0:
                    result["state"] = "busy"
                elif (result["requests_waiting"] or 0) > 0:
                    result["state"] = "queued"
        except httpx.HTTPError:
            # Metrics are an optional enhancement. A successful health probe
            # remains authoritative for service readiness.
            pass
        return result

    def _planner_status(self, project_ids: list[str], checked_at: datetime) -> dict[str, Any]:
        configured = bool(self.settings.planner_base_url) or self.settings.planner_allow_fallback
        return {
            "id": "planner",
            "display_name": "Storyboard Planner",
            "kind": "planner",
            "provider": self.settings.planner_source,
            "model": self.settings.planner_model or "deterministic fallback",
            "configured": configured,
            "state": "busy" if project_ids else ("ready" if configured else "unconfigured"),
            "healthy": None,
            "checked_at": checked_at.isoformat(),
            "latency_ms": None,
            "http_status": None,
            "requests_running": len(project_ids),
            "requests_waiting": 0,
            "requests_succeeded": None,
            "error": None,
            "gpu": None,
        }

    @staticmethod
    def _gpu_summary(devices: list[dict[str, Any]], telemetry: dict[str, Any]) -> dict[str, Any] | None:
        if not devices:
            return None
        utilization = [item["utilization_percent"] for item in devices if item["utilization_percent"] is not None]
        memory_used = [item["memory_used_mib"] for item in devices if item["memory_used_mib"] is not None]
        memory_total = [item["memory_total_mib"] for item in devices if item["memory_total_mib"] is not None]
        temperatures = [item["temperature_c"] for item in devices if item["temperature_c"] is not None]
        return {
            "state": telemetry["state"],
            "age_seconds": telemetry["age_seconds"],
            "device_count": len(devices),
            "utilization_percent": round(sum(utilization) / len(utilization), 1) if utilization else None,
            "memory_used_mib": sum(memory_used) if len(memory_used) == len(devices) else None,
            "memory_total_mib": sum(memory_total) if len(memory_total) == len(devices) else None,
            "temperature_c": max(temperatures) if temperatures else None,
            "devices": sorted(devices, key=lambda item: (str(item.get("node") or ""), int(item["index"]))),
        }
