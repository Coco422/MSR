from __future__ import annotations

from collections import defaultdict
import shutil
from pathlib import Path
from tempfile import mkdtemp
import uuid

from fastapi import UploadFile

from msr.backends.vad.webrtc_backend import WebRTCVADBackend
from msr.core.config import Settings
from msr.core.errors import InvalidAudioError
from msr.core.types import SpeakerSegment, TextSegment
from msr.services.audio_io import load_audio, write_clip
from msr.services.model_manager import ModelManager


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


class TranscriptionService:
    def __init__(self, settings: Settings, model_manager: ModelManager):
        self.settings = settings
        self.model_manager = model_manager
        self.vad = WebRTCVADBackend()

    async def transcribe_upload(self, audio: UploadFile, language: str | None = None) -> dict:
        file_extension = Path(audio.filename or "").suffix.lower()
        if file_extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise InvalidAudioError(f"Unsupported file format: {file_extension}")

        task_id = str(uuid.uuid4())
        self.settings.app.temp_dir.mkdir(parents=True, exist_ok=True)
        task_dir = Path(mkdtemp(prefix=f"msr_{task_id}_", dir=self.settings.app.temp_dir))
        source_path = task_dir / f"source{file_extension}"

        try:
            with source_path.open("wb") as handle:
                handle.write(await audio.read())

            with self.model_manager.task_scope():
                return self._transcribe_path(task_id=task_id, source_path=source_path, language=language or self.settings.app.default_language, task_dir=task_dir)
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

    def _transcribe_path(self, task_id: str, source_path: Path, language: str, task_dir: Path) -> dict:
        audio, sample_rate = load_audio(source_path)
        audio_duration = len(audio) / sample_rate if sample_rate else 0.0
        vad_ranges = self.vad.detect(audio, sample_rate)
        if not vad_ranges:
            raise InvalidAudioError("No speech activity detected.")

        diarization_backend = self.model_manager.get_diarization_backend()
        asr_backend = self.model_manager.get_asr_backend()

        speaker_segments = diarization_backend.diarize(source_path)
        if not speaker_segments:
            speaker_segments = [SpeakerSegment(speaker_id="0", start=0.0, end=audio_duration)]

        text_segments: list[TextSegment] = []
        for start, end in vad_ranges:
            clip = write_clip(audio, sample_rate, start, end, task_dir)
            for segment in asr_backend.transcribe(clip.path, language=language, timestamps=True):
                text_segments.append(
                    TextSegment(
                        start=start + segment.start,
                        end=min(end, start + segment.end),
                        text=segment.text,
                        confidence=segment.confidence,
                    )
                )

        speaker_texts = _match_text_to_speakers(text_segments, speaker_segments)
        speakers_info = _build_speakers_info(speaker_segments)
        transcripts = []
        for speaker_id, segments in speaker_texts.items():
            speaker_label = _speaker_label(speaker_id)
            for segment in segments:
                transcripts.append(
                    {
                        "speaker_id": speaker_id,
                        "speaker_label": speaker_label,
                        "text": segment.text,
                        "start_time": format_time(segment.start),
                        "end_time": format_time(segment.end),
                        "confidence": segment.confidence,
                    }
                )

        transcripts.sort(key=lambda item: _parse_time(item["start_time"]))

        return {
            "task_id": task_id,
            "status": "completed",
            "transcripts": transcripts,
            "speakers_info": speakers_info,
            "total_speakers": len(speakers_info),
            "audio_duration": format_time(audio_duration),
            "processing_time": "0:00",
            "processing_speed": "0.00x",
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


def _build_speakers_info(speaker_segments: list[SpeakerSegment]) -> list[dict]:
    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in speaker_segments:
        grouped[segment.speaker_id].append(segment)

    items = []
    for speaker_id, segments in grouped.items():
        total = sum(segment.end - segment.start for segment in segments)
        items.append(
            {
                "speaker_id": speaker_id,
                "speaker_label": _speaker_label(speaker_id),
                "total_duration": format_time(total),
                "segment_count": len(segments),
            }
        )
    items.sort(key=lambda item: item["segment_count"], reverse=True)
    return items


def _speaker_label(speaker_id: str) -> str:
    normalized = str(speaker_id)
    numeric = normalized.replace("spk_", "").replace("speaker_", "")
    if numeric.isdigit():
        number = int(numeric)
        if number < 26:
            return f"说话人 {chr(65 + number)}"
        return f"说话人 {number + 1}"
    return f"说话人 {normalized}"


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
