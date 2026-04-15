from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI

from msr.core.config import Settings
from msr.core.runtime_env import format_runtime_context


logger = logging.getLogger(__name__)


def _apply_offline_env(settings: Settings) -> None:
    if not settings.app.strict_offline:
        return
    for key, value in settings.offline_env.items():
        os.environ[key] = value


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    settings: Settings = app.state.container.settings
    _apply_offline_env(settings)
    Path(settings.app.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.runtime.data_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Application startup host=%s port=%s log_level=%s strict_offline=%s temp_dir=%s data_dir=%s %s",
        settings.app.host,
        settings.app.port,
        settings.app.log_level,
        settings.app.strict_offline,
        settings.app.temp_dir,
        settings.runtime.data_dir,
        format_runtime_context(),
    )
    await app.state.container.task_manager.start()
    try:
        yield
    finally:
        logger.info("Application shutdown requested")
        await app.state.container.task_manager.shutdown()
