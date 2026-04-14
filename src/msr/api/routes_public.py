from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from msr.api.deps import get_container
from msr.api.schemas import AnalysisResponse
from msr.core.errors import InvalidAudioError, ModelNotLoadedError, TranscriptionError
from msr.core.security import require_api_key


router = APIRouter()


@router.post("/transcribe/", response_model=AnalysisResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="上传的音频文件(支持wav/mp3/ogg/flac/m4a)"),
    _: str = Depends(require_api_key),
    container=Depends(get_container),
):
    started_at = time.time()
    try:
        result = await container.transcription_service.transcribe_upload(audio)
        elapsed = time.time() - started_at
        audio_duration = _duration_to_seconds(result["audio_duration"])
        speed = audio_duration / elapsed if elapsed > 0 else 0.0
        result["processing_time"] = _format_duration(elapsed)
        result["processing_speed"] = f"{speed:.2f}x"
        return result
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TranscriptionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health")
async def health_check(container=Depends(get_container)):
    return {
        "status": "healthy",
        "service": container.settings.app.service_name,
        "active_models": container.model_manager.active_state(),
    }


def _duration_to_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours = rounded // 3600
    minutes = (rounded % 3600) // 60
    secs = rounded % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
