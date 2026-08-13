from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from long_video_studio import __version__
from long_video_studio.api import create_api_router
from long_video_studio.config import Settings
from long_video_studio.runner import RenderManager
from long_video_studio.services import StudioServices


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    services = StudioServices.create(resolved)
    app = FastAPI(
        title="Nautilus Studio",
        description="Nautilus Studio — creator-first agentic AI film workshop",
        version=__version__,
    )
    app.state.services = services
    app.state.render_manager = RenderManager(resolved, services.repository)
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

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(index_path)

    return app
