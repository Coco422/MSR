from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import shutil
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Callable
import uuid

from fastapi import UploadFile

from msr.backends.vad.webrtc_backend import WebRTCVADBackend
from msr.core.config import Settings
from msr.core.errors import InvalidAudioError
from msr.core.types import SpeakerSegment, TextSegment
from msr.services.audio_io import load_audio, write_clip
from msr.services.model_manager import ModelManager


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
StageCallback = Callable[[str, str | None], None]


@dataclass(slots=True)
class PreparedAudioUpload:
    task_id: str
    original_filename: str
    language: str
    task_dir: Path
    source_path: Path


class TranscriptionService:
    def __init__(self, settings: Settings, model_manager: ModelManager):
        self.settings = settings
        self.model_manager = model_manager
        self.vad = WebRTCVADBackend()

    async def transcribe_upload(self, audio: UploadFile, language: str | None = None) -> dict:
        prepared = await self.prepare_upload(audio, language=language)
        try:
            return self.transcribe_prepared(prepared)
        finally:
            self.cleanup_prepared_upload(prepared)

    async def prepare_upload(
        self,
        audio: UploadFile,
        language: str | None = None,
        task_id: str | None = None,
    ) -> PreparedAudioUpload:
        file_extension = Path(audio.filename or "").suffix.lower()
        if file_extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise InvalidAudioError(f"Unsupported file format: {file_extension}")

        resolved_task_id = task_id or str(uuid.uuid4())
        self.settings.app.temp_dir.mkdir(parents=True, exist_ok=True)
        task_dir = Path(mkdtemp(prefix=f"msr_{resolved_task_id}_", dir=self.settings.app.temp_dir))
        source_path = task_dir / f"source{file_extension}"

        with source_path.open("wb") as handle:
            handle.write(await audio.read())

        return PreparedAudioUpload(
            task_id=resolved_task_id,
            original_filename=audio.filename or source_path.name,
            language=language or self.settings.app.default_language,
            task_dir=task_dir,
            source_path=source_path,
        )

    def cleanup_prepared_upload(self, prepared: PreparedAudioUpload) -> None:
        shutil.rmtree(prepared.task_dir, ignore_errors=True)

    def transcribe_prepared(
        self,
        prepared: PreparedAudioUpload,
        progress_callback: StageCallback | None = None,
    ) -> dict:
        started_at = time.perf_counter()

        def update_stage(stage: str, audio_duration: str | None = None) -> None:
            if progress_callback:
                progress_callback(stage, audio_duration)

        with self.model_manager.task_scope():
            update_stage("normalizing")
            audio, sample_rate = load_audio(prepared.source_path)
            audio_duration_seconds = len(audio) / sample_rate if sample_rate else 0.0
            audio_duration = format_time(audio_duration_seconds)

            update_stage("vad", audio_duration)
            vad_ranges = self.vad.detect(audio, sample_rate)
            if not vad_ranges:
                raise InvalidAudioError("No speech activity detected.")

            diarization_backend = self.model_manager.get_diarization_backend()
            asr_backend = self.model_manager.get_asr_backend()

            update_stage("diarization", audio_duration)
            speaker_segments = diarization_backend.diarize(prepared.source_path)
            if not speaker_segments:
                speaker_segments = [SpeakerSegment(speaker_id="0", start=0.0, end=audio_duration_seconds)]

            update_stage("asr", audio_duration)
            text_segments: list[TextSegment] = []
            for start, end in vad_ranges:
                clip = write_clip(audio, sample_rate, start, end, prepared.task_dir)
                for segment in asr_backend.transcribe(clip.path, language=prepared.language, timestamps=True):
                    segment_start = max(0.0, start + segment.start)
                    segment_end = min(end, start + segment.end)
                    if segment_end <= segment_start:
                        continue
                    if not str(segment.text).strip():
                        continue
                    text_segments.append(
                        TextSegment(
                            start=segment_start,
                            end=segment_end,
                            text=segment.text,
                            confidence=segment.confidence,
                        )
                    )

            update_stage("postprocess", audio_duration)
            speaker_texts = _match_text_to_speakers(text_segments, speaker_segments)
            speaker_presentations = _build_speaker_presentations(speaker_texts)
            speakers_info = _build_speakers_info(speaker_segments, speaker_texts, speaker_presentations)
            transcripts = _build_transcripts(speaker_texts, speaker_presentations)

        processing_seconds = max(time.perf_counter() - started_at, 0.0)
        processing_speed = audio_duration_seconds / processing_seconds if processing_seconds > 0 else 0.0

        return {
            "task_id": prepared.task_id,
            "status": "completed",
            "transcripts": transcripts,
            "speakers_info": speakers_info,
            "total_speakers": len(speakers_info),
            "audio_duration": audio_duration,
            "processing_time": _format_duration(processing_seconds),
            "processing_speed": f"{processing_speed:.2f}x",
        }


def _match_text_to_speakers(text_segments: list[TextSegment], speaker_segments: list[SpeakerSegment]) -> dict[str, list[TextSegment]]:
    grouped: dict[str, list[TextSegment]] = defaultdict(list)
    for segment in text_segments:
        best_speaker = "0"
        best_overlap = 0.0
        for speaker_segment in speaker_segments:
            overlap = min(segment.end, speaker_segment.end) - max(segment.start, speaker_segment.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_segment.speaker_id
        grouped[best_speaker].append(segment)
    return grouped


def _build_transcripts(
    speaker_texts: dict[str, list[TextSegment]],
    speaker_presentations: dict[str, dict[str, str]],
) -> list[dict]:
    items: list[tuple[float, dict]] = []
    for raw_speaker_id, segments in speaker_texts.items():
        presentation = speaker_presentations.get(raw_speaker_id)
        if not presentation:
            continue
        for segment in segments:
            if segment.end <= segment.start:
                continue
            items.append(
                (
                    segment.start,
                    {
                        "speaker_id": presentation["speaker_id"],
                        "speaker_label": presentation["speaker_label"],
                        "text": segment.text,
                        "start_time": format_time(segment.start),
                        "end_time": format_time(segment.end),
                        "confidence": segment.confidence,
                    },
                )
            )

    items.sort(key=lambda item: item[0])
    return [payload for _, payload in items]


def _build_speaker_presentations(speaker_texts: dict[str, list[TextSegment]]) -> dict[str, dict[str, str]]:
    valid_speakers = [
        speaker_id
        for speaker_id, segments in speaker_texts.items()
        if any(segment.end > segment.start and str(segment.text).strip() for segment in segments)
    ]
    valid_speakers.sort(key=lambda speaker_id: (_speaker_first_start(speaker_texts[speaker_id]), _speaker_sort_key(speaker_id)))

    return {
        raw_speaker_id: {
            "speaker_id": str(index),
            "speaker_label": _speaker_label(index),
        }
        for index, raw_speaker_id in enumerate(valid_speakers)
    }


def _build_speakers_info(
    speaker_segments: list[SpeakerSegment],
    speaker_texts: dict[str, list[TextSegment]],
    speaker_presentations: dict[str, dict[str, str]],
) -> list[dict]:
    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in speaker_segments:
        grouped[segment.speaker_id].append(segment)

    items = []
    for raw_speaker_id, presentation in speaker_presentations.items():
        segments = grouped.get(raw_speaker_id, [])
        if segments:
            total = sum(segment.end - segment.start for segment in segments)
            segment_count = len(segments)
        else:
            text_segments = speaker_texts.get(raw_speaker_id, [])
            total = sum(max(0.0, segment.end - segment.start) for segment in text_segments)
            segment_count = len(text_segments)
        items.append(
            {
                "speaker_id": presentation["speaker_id"],
                "speaker_label": presentation["speaker_label"],
                "total_duration": format_time(total),
                "segment_count": segment_count,
            }
        )
    return items


def _speaker_first_start(segments: list[TextSegment]) -> float:
    starts = [segment.start for segment in segments if segment.end > segment.start]
    if not starts:
        return float("inf")
    return min(starts)


def _speaker_sort_key(value: str) -> tuple[int, str]:
    numeric = str(value).replace("spk_", "").replace("speaker_", "")
    if numeric.isdigit():
        return (0, f"{int(numeric):08d}")
    return (1, str(value))


def _speaker_label(index: int) -> str:
    if index < 26:
        return f"说话人 {chr(65 + index)}"
    return f"说话人 {index + 1}"


def format_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _parse_time(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def _format_duration(seconds: float) -> str:
    return format_time(seconds)
