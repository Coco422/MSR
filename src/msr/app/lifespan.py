from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI

from msr.core.config import Settings


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
    yield
