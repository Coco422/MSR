from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from msr.api.routes_admin import router as admin_router
from msr.api.routes_public import router as public_router
from msr.core.config import Settings, load_settings
from msr.core.logging import configure_logging
from msr.services.container import ServiceContainer
from msr.app.lifespan import app_lifespan


def create_app(settings: Settings | None = None, project_root: Path | None = None) -> FastAPI:
    resolved_settings = settings or load_settings(project_root=project_root)
    configure_logging(resolved_settings.app.log_level)

    app = FastAPI(
        title=resolved_settings.app.name,
        description=resolved_settings.app.service_name,
        version=resolved_settings.app.version,
        lifespan=app_lifespan,
    )
    app.state.container = ServiceContainer.from_settings(resolved_settings)

    web_dir = resolved_settings.project_root / "src" / "msr" / "web"
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    app.include_router(public_router)
    app.include_router(admin_router)

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    return app


def run() -> None:
    app = create_app()
    settings = app.state.container.settings
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)


app = create_app()
