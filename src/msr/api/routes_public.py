from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from msr.api.deps import get_container
from msr.api.schemas import AnalysisResponse
from msr.core.errors import InvalidAudioError, ModelNotLoadedError, QueueFullError, TranscriptionError
from msr.core.security import require_api_key


router = APIRouter()


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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
