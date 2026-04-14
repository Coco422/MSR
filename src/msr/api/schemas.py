from __future__ import annotations

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    speaker_id: str
    speaker_label: str
    text: str
    start_time: str
    end_time: str
    confidence: float = 1.0


class SpeakerInfo(BaseModel):
    speaker_id: str
    speaker_label: str
    total_duration: str
    segment_count: int


class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    transcripts: list[TranscriptionResult]
    speakers_info: list[SpeakerInfo]
    total_speakers: int
    audio_duration: str
    processing_time: str
    processing_speed: str


class ModelInfo(BaseModel):
    id: str
    kind: str
    backend: str
    local_path: str
    device: str
    enabled: bool
    default: bool
    path_exists: bool
    loaded: bool


class AuthCheckResponse(BaseModel):
    authenticated: bool


class RuntimeStateResponse(BaseModel):
    asr: dict | None
    diarization: dict | None
