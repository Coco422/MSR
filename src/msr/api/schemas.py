from __future__ import annotations

from pydantic import BaseModel, Field


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
    operation_status: str | None = None


class AuthCheckResponse(BaseModel):
    authenticated: bool


class RuntimeStateResponse(BaseModel):
    asr: dict | None
    diarization: dict | None


class RuntimeTaskRecord(BaseModel):
    task_id: str
    status: str
    stage: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queue_wait_ms: int | None = None
    run_ms: int | None = None
    filename: str
    audio_duration: str | None = None
    error: str | None = None


class TaskStatusResponse(RuntimeTaskRecord):
    result_available: bool = False


class RuntimeTaskCounts(BaseModel):
    active: int
    queued: int
    recent: int


class RuntimeTasksResponse(BaseModel):
    counts: RuntimeTaskCounts
    active: list[RuntimeTaskRecord]
    queued: list[RuntimeTaskRecord]
    recent: list[RuntimeTaskRecord]


class RuntimeLimitsResponse(BaseModel):
    max_parallel_tasks: int
    max_queued_tasks: int
    recent_task_limit: int


class RuntimeLimitsUpdateRequest(BaseModel):
    max_parallel_tasks: int | None = Field(default=None, ge=1)
    max_queued_tasks: int | None = Field(default=None, ge=0)
    recent_task_limit: int | None = Field(default=None, ge=1)
