from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from msr.api.deps import get_container
from msr.api.schemas import AnalysisResponse, TaskStatusResponse
from msr.core.errors import InvalidAudioError, ModelNotLoadedError, QueueFullError, TranscriptionError
from msr.core.security import require_api_key


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/transcribe/", response_model=AnalysisResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="上传的音频文件(支持wav/mp3/ogg/flac/m4a)"),
    _: str = Depends(require_api_key),
    container=Depends(get_container),
):
    prepared = None
    try:
        container.model_manager.ensure_transcription_ready()
        prepared = await container.transcription_service.prepare_upload(audio)
        return await container.task_manager.submit(
            task_id=prepared.task_id,
            filename=prepared.original_filename,
            runner=lambda progress: _run_transcription(container, prepared, progress),
        )
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except QueueFullError as exc:
        if prepared:
            container.transcription_service.cleanup_prepared_upload(prepared)
        raise HTTPException(status_code=503, detail=exc.detail) from exc
    except TranscriptionError as exc:
        status_code, detail = _build_transcription_error_detail(exc, task_id=prepared.task_id if prepared else None)
        logger.warning("Sync transcription failed detail=%s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.post("/api/v1/transcriptions/submit", response_model=TaskStatusResponse)
async def submit_transcription_task(
    audio: UploadFile = File(..., description="上传的音频文件(支持wav/mp3/ogg/flac/m4a)"),
    _: str = Depends(require_api_key),
    container=Depends(get_container),
):
    prepared = None
    try:
        container.model_manager.ensure_transcription_ready()
        prepared = await container.transcription_service.prepare_upload(audio)
        return await container.task_manager.submit_async(
            task_id=prepared.task_id,
            filename=prepared.original_filename,
            runner=lambda progress: _run_transcription(container, prepared, progress),
        )
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except QueueFullError as exc:
        if prepared:
            container.transcription_service.cleanup_prepared_upload(prepared)
        raise HTTPException(status_code=503, detail=exc.detail) from exc
    except TranscriptionError as exc:
        if prepared:
            container.transcription_service.cleanup_prepared_upload(prepared)
        status_code, detail = _build_transcription_error_detail(exc, task_id=prepared.task_id if prepared else None)
        logger.warning("Async transcription submit failed detail=%s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.get("/api/v1/transcriptions/{task_id}", response_model=TaskStatusResponse)
async def get_transcription_task_status(task_id: str, _: str = Depends(require_api_key), container=Depends(get_container)):
    detail = container.task_manager.task_detail(task_id)
    if not detail:
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": f"Unknown task_id: {task_id}"})
    return detail


@router.get("/api/v1/transcriptions/{task_id}/result")
async def get_transcription_task_result(task_id: str, _: str = Depends(require_api_key), container=Depends(get_container)):
    record, payload = container.task_manager.task_result(task_id)
    if not record:
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": f"Unknown task_id: {task_id}"})
    if record.status == "completed" and payload is not None:
        return payload
    if record.status == "completed":
        return JSONResponse(
            status_code=410,
            content={
                "code": "task_result_expired",
                "task_id": task_id,
                "status": record.status,
                "message": "Task result is no longer retained locally.",
            },
        )
    if record.status in {"queued", "running"}:
        return JSONResponse(
            status_code=202,
            content={
                "code": "task_not_ready",
                "task_id": task_id,
                "status": record.status,
                "stage": record.stage,
                "message": "Task is still running.",
            },
        )
    failure_detail = _task_failure_detail(record)
    status_code = int(failure_detail.pop("http_status", 409))
    return JSONResponse(status_code=status_code, content=failure_detail)


@router.get("/health")
async def health_check(container=Depends(get_container)):
    return {
        "status": "healthy",
        "service": container.settings.app.service_name,
        "active_models": container.model_manager.active_state(),
    }


def _run_transcription(container, prepared, progress):
    try:
        return container.transcription_service.transcribe_prepared(prepared, progress_callback=progress)
    finally:
        container.transcription_service.cleanup_prepared_upload(prepared)


def _build_transcription_error_detail(exc: Exception, task_id: str | None = None) -> tuple[int, dict]:
    raw_message = str(exc)
    detail = {
        "code": "transcription_failed",
        "message": raw_message,
    }
    if task_id:
        detail["task_id"] = task_id

    lowered = raw_message.lower()
    if "cuda out of memory" in lowered or "out of memory" in lowered:
        detail["code"] = "cuda_oom"
        detail["message"] = "ASR 阶段显存不足。通常是单段音频过长、当前显存被其他进程占用，或同步转写叠加了大模型开销。"
        detail["raw_error"] = raw_message
        detail["hint"] = "建议优先使用异步任务接口，并检查当前 GPU 占用或进一步减小 ASR 切段长度。"
        return 507, detail
    if "no speech activity detected" in lowered:
        detail["code"] = "no_speech"
        detail["message"] = "音频中未检测到有效语音。"
        detail["raw_error"] = raw_message
        return 400, detail

    return 500, detail


def _task_failure_detail(record) -> dict:
    status_code, detail = _build_transcription_error_detail(record.error or "Task failed.", task_id=record.task_id)
    detail["status"] = record.status
    detail["stage"] = record.stage
    detail["http_status"] = status_code
    return detail
