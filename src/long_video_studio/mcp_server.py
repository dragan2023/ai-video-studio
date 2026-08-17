"""Model Context Protocol surface for Nautilus Studio.

The MCP server deliberately shares the Studio process and repository.  It
exposes creator-level operations only; it does not provide arbitrary shell,
filesystem, or provider-admin access.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings
from mcp.server.transport_security import TransportSecuritySettings

from .domain import (
    AssetRole,
    ContinuationMode,
    FilmProject,
    ProjectBrief,
    UltraFastAnchorStrategy,
    UltraFastTransition,
)

# MCP 1.27 exposes a generic FastMCP settings model whose lifespan annotation
# is unresolved until the module is imported.  Rebuilding it here prevents
# pydantic-settings from warning (or ignoring the field) when Studio loads
# environment configuration.
FastMCPSettings.model_rebuild()


class BearerAuthASGI:
    """Optional token gate for the mounted MCP app."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"authorization", b"").decode("latin-1")
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(supplied, expected):
            body = b'{"error":"unauthorized"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _project_summary(project: FilmProject, repository: Any) -> dict[str, Any]:
    job = repository.get_latest_job(project.id)
    return {
        "id": project.id,
        "title": project.brief.title,
        "prompt": project.brief.prompt,
        "duration_seconds": project.brief.duration_seconds,
        "shot_count": len(project.shots),
        "continuation_mode": project.brief.continuation_mode.value,
        "ultra_fast_anchor_strategy": project.brief.ultra_fast_anchor_strategy.value,
        "ultra_fast_transition": project.brief.ultra_fast_transition.value,
        "latest_job": _dump(job) if job else None,
    }


def _asset_summary(asset: Any) -> dict[str, Any]:
    data = _dump(asset)
    if isinstance(data, dict):
        # Keep the tool response useful to an agent without returning large
        # binary or internal bookkeeping fields.
        return {
            key: data.get(key)
            for key in (
                "id",
                "display_name",
                "original_name",
                "kind",
                "media_type",
                "roles",
                "tags",
                "caption",
                "width",
                "height",
                "duration_seconds",
                "source",
            )
            if key in data
        }
    return {"value": data}


def create_mcp_server(services: Any, planning_manager: Any, render_manager: Any) -> FastMCP:
    """Build the in-process MCP server around existing Studio services."""

    repository = services.repository
    mcp = FastMCP(
        "Nautilus Studio",
        stateless_http=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(services.settings.mcp_allowed_hosts),
            allowed_origins=[],
        ),
    )

    @mcp.tool(
        name="studio_status",
        description=(
            "Return Studio service health, active planning/render jobs, and the "
            "currently configured model capabilities."
        ),
    )
    async def studio_status() -> dict[str, Any]:
        active_jobs = [repository.get_latest_job(project_id) for project_id in render_manager.active_project_ids()]
        status = await services.service_status.collect(
            planning_project_ids=planning_manager.active_project_ids(),
            active_jobs=[job for job in active_jobs if job is not None],
        )
        return _dump(status)

    @mcp.tool(
        name="studio_list_projects",
        description="List creator projects with their latest render status.",
    )
    def studio_list_projects() -> list[dict[str, Any]]:
        return [_project_summary(project, repository) for project in repository.list_projects()]

    @mcp.tool(
        name="studio_get_project",
        description="Read one complete storyboard/project by id.",
    )
    def studio_get_project(project_id: str) -> dict[str, Any]:
        project = repository.get_project(project_id)
        if project is None:
            raise ValueError(f"project not found: {project_id}")
        return _dump(project)

    @mcp.tool(
        name="studio_list_assets",
        description="List the creator material library and its roles/tags.",
    )
    def studio_list_assets() -> list[dict[str, Any]]:
        return [_asset_summary(asset) for asset in repository.list_assets()]

    @mcp.tool(
        name="studio_import_asset",
        description=(
            "Import a local creator asset through Studio's configured allowed "
            "roots. The tool never reads arbitrary filesystem paths."
        ),
    )
    def studio_import_asset(
        path: str,
        tags: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            asset_roles = [AssetRole(role) for role in (roles or [AssetRole.REFERENCE.value])]
        except ValueError as exc:
            raise ValueError(f"invalid asset role: {exc}") from exc
        imported = services.assets.import_path(path, tags=tags or [], roles=asset_roles)
        return [_asset_summary(asset) for asset in imported]

    @mcp.tool(
        name="studio_plan_project",
        description=(
            "Create a project and start the AI storyboard planner. The call "
            "returns immediately with a project id; use studio_get_project to "
            "poll the storyboard."
        ),
    )
    async def studio_plan_project(
        prompt: str,
        title: str = "",
        duration_seconds: int = 30,
        continuation_mode: str = "quality",
        reference_asset_ids: list[str] | None = None,
        ultra_fast_anchor_strategy: str = "independent",
        ultra_fast_transition: str = "fade_black",
    ) -> dict[str, Any]:
        try:
            mode = ContinuationMode(continuation_mode)
            anchor_strategy = UltraFastAnchorStrategy(ultra_fast_anchor_strategy)
            transition = UltraFastTransition(ultra_fast_transition)
            brief = ProjectBrief(
                prompt=prompt.strip(),
                title=title.strip() or "未命名短片",
                duration_seconds=duration_seconds,
                continuation_mode=mode,
                reference_asset_ids=reference_asset_ids or [],
                ultra_fast_anchor_strategy=anchor_strategy,
                ultra_fast_transition=transition,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid project brief: {exc}") from exc
        missing = [asset_id for asset_id in brief.reference_asset_ids if repository.get_asset(asset_id) is None]
        if missing:
            raise ValueError(f"unknown reference_asset_ids: {', '.join(missing)}")
        project = await planning_manager.start(brief)
        return {
            "project_id": project.id,
            "status": "planning",
            "title": project.brief.title,
        }

    @mcp.tool(
        name="studio_render_project",
        description="Start rendering an approved storyboard and return the render job.",
    )
    def studio_render_project(project_id: str, force: bool = False) -> dict[str, Any]:
        project = repository.get_project(project_id)
        if project is None:
            raise ValueError(f"project not found: {project_id}")
        plan = services.compiler.compile(project)
        unconfigured = [
            deployment.capability_id
            for deployment in plan.deployments
            if deployment.status != "ready"
        ]
        if unconfigured:
            raise ValueError(
                "render requires configured endpoint(s): " + ", ".join(unconfigured)
            )
        job = render_manager.submit(project_id, force=force)
        return {
            "job": _dump(job),
            "estimated_seconds": plan.estimated_seconds,
            "deployments": _dump(plan.deployments),
        }

    @mcp.tool(
        name="studio_render_status",
        description="Read the latest render job for a project.",
    )
    def studio_render_status(project_id: str) -> dict[str, Any]:
        if repository.get_project(project_id) is None:
            raise ValueError(f"project not found: {project_id}")
        job = repository.get_latest_job(project_id)
        return _dump(job) if job else {"project_id": project_id, "status": "not_started"}

    @mcp.tool(
        name="studio_cancel_planning",
        description="Cancel an in-progress storyboard planning task.",
    )
    def studio_cancel_planning(project_id: str) -> dict[str, Any]:
        return {"project_id": project_id, "cancelled": planning_manager.cancel(project_id)}

    @mcp.resource("studio://projects")
    def projects_resource() -> str:
        return json.dumps(studio_list_projects(), ensure_ascii=False, indent=2)

    @mcp.resource("studio://project/{project_id}")
    def project_resource(project_id: str) -> str:
        return json.dumps(studio_get_project(project_id), ensure_ascii=False, indent=2)

    @mcp.prompt("creator_workflow")
    def creator_workflow() -> str:
        return (
            "Use studio_list_assets and studio_status first. For a new film, "
            "call studio_plan_project with the user's story and selected asset "
            "ids, inspect studio_get_project until the storyboard is complete, "
            "then call studio_render_project and poll studio_render_status. "
            "Prefer ultra_fast with independent anchors for short-drama shots "
            "when Ref2VA is not needed."
        )

    return mcp
