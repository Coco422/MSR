from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from msr.api.deps import get_container
from msr.api.schemas import AuthCheckResponse, ModelInfo, RuntimeStateResponse
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


@router.get("/system/resources")
async def system_resources(_: str = Depends(require_api_key), container=Depends(get_container)):
    return container.resource_monitor.snapshot()
