from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from long_video_studio import __version__
from long_video_studio.api import create_api_router
from long_video_studio.config import Settings
from long_video_studio.llms_txt import render_llms_txt
from long_video_studio.mcp_server import BearerAuthASGI, create_mcp_server
from long_video_studio.planning import PlanningManager
from long_video_studio.runner import RenderManager
from long_video_studio.services import StudioServices


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    services = StudioServices.create(resolved)
    planning_manager = PlanningManager(resolved, services.repository, services.planner)
    render_manager = RenderManager(
        resolved,
        services.repository,
        estimator=services.estimator,
    )
    mcp_server = create_mcp_server(services, planning_manager, render_manager) if resolved.mcp_enabled else None
    mcp_http_app = mcp_server.streamable_http_app() if mcp_server is not None else None
    if mcp_http_app is not None and resolved.mcp_token:
        mcp_http_app = BearerAuthASGI(mcp_http_app, resolved.mcp_token)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if mcp_server is None:
            try:
                yield
            finally:
                await planning_manager.shutdown()
                await render_manager.shutdown()
            return
        async with mcp_server.session_manager.run():
            try:
                yield
            finally:
                await planning_manager.shutdown()
                await render_manager.shutdown()

    app = FastAPI(
        title="Nautilus Studio",
        description="Nautilus Studio — creator-first agentic AI film workshop",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.services = services
    app.state.planning_manager = planning_manager
    app.state.render_manager = render_manager
    app.state.mcp_server = mcp_server
    app.include_router(create_api_router())

    if not resolved.web_root:
        raise RuntimeError(
            "STUDIO_WEB_ROOT is required. Build the React UI with "
            "`npm --prefix web run build` and point STUDIO_WEB_ROOT at web/dist."
        )
    static_dir = Path(resolved.web_root).expanduser().resolve()
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        raise RuntimeError(f"React UI build is missing index.html under STUDIO_WEB_ROOT: {static_dir}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    assets_dir = static_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="web-assets")
    if mcp_http_app is not None:
        app.mount(resolved.mcp_path.rstrip("/") or "/mcp", mcp_http_app, name="mcp")

    @app.get("/llms.txt", response_class=PlainTextResponse, include_in_schema=False)
    def llms_txt(request: Request) -> PlainTextResponse:
        return PlainTextResponse(
            render_llms_txt(
                str(request.base_url),
                mcp_enabled=resolved.mcp_enabled,
                mcp_requires_token=bool(resolved.mcp_token),
            ),
            headers={"Cache-Control": "public, max-age=300", "Vary": "Host"},
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(index_path)

    return app
