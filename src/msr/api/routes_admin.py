from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from msr.api.deps import get_container
from msr.api.schemas import (
    AuthCheckResponse,
    ModelInfo,
    RuntimeLimitsResponse,
    RuntimeLimitsUpdateRequest,
    RuntimeStateResponse,
    RuntimeTasksResponse,
)
from msr.core.errors import BackendLoadError, ModelBusyError, ModelNotFoundError, ModelNotLoadedError
from msr.core.security import require_api_key


router = APIRouter(prefix="/api/v1")


@router.get("/auth/check", response_model=AuthCheckResponse)
async def auth_check(_: str = Depends(require_api_key)):
    return {"authenticated": True}


@router.get("/models", response_model=list[ModelInfo])
async def list_models(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.model_manager.list_models()


@router.post("/models/{kind}/{model_id}/load")
async def load_model(kind: str, model_id: str, _: str = Depends(require_api_key), container=Depends(get_container)):
    try:
        return container.model_manager.load(kind, model_id)
    except (ModelBusyError, ModelNotLoadedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ModelNotFoundError, BackendLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{kind}/{model_id}/unload")
async def unload_model(kind: str, model_id: str, _: str = Depends(require_api_key), container=Depends(get_container)):
    try:
        container.model_manager.unload(kind, model_id)
        return {"status": "unloaded", "kind": kind, "model_id": model_id}
    except (ModelBusyError, ModelNotLoadedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runtime/active", response_model=RuntimeStateResponse)
async def runtime_active(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.model_manager.active_state()


@router.get("/runtime/tasks", response_model=RuntimeTasksResponse)
async def runtime_tasks(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.task_manager.task_snapshot()


@router.get("/runtime/limits", response_model=RuntimeLimitsResponse)
async def runtime_limits(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.task_manager.limits_snapshot()


@router.post("/runtime/limits", response_model=RuntimeLimitsResponse)
async def update_runtime_limits(
    payload: RuntimeLimitsUpdateRequest,
    _: str = Depends(require_api_key),
    container=Depends(get_container),
):
    return container.task_manager.update_limits(
        max_parallel_tasks=payload.max_parallel_tasks,
        max_queued_tasks=payload.max_queued_tasks,
        recent_task_limit=payload.recent_task_limit,
    )


@router.get("/system/resources")
async def system_resources(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.resource_monitor.snapshot()
