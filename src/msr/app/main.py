from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, project_root: Path | None = None) -> FastAPI:
    resolved_settings = settings or load_settings(project_root=project_root)
    configure_logging(resolved_settings.app.log_level)
    logger.debug("Creating FastAPI application instance")

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

    console_pages = {
        "/": "index.html",
        "/models": "models.html",
        "/runtime": "runtime.html",
        "/transcribe": "transcribe.html",
    }

    def build_console_handler(page_path: Path, endpoint_name: str):
        async def serve_page() -> FileResponse:
            return FileResponse(page_path)

        serve_page.__name__ = endpoint_name
        return serve_page

    for route_path, filename in console_pages.items():
        endpoint_name = f"console_{filename.replace('.html', '').replace('-', '_')}"
        app.get(route_path, include_in_schema=False)(
            build_console_handler(web_dir / filename, endpoint_name)
        )

    return app


def run() -> None:
    settings = app.state.container.settings
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)


app = create_app()


if __name__ == "__main__":
    run()
