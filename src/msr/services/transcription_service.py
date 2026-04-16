from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
import shutil
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Callable
import uuid

from fastapi import UploadFile

from msr.backends.vad.webrtc_backend import WebRTCVADBackend
from msr.core.config import Settings
from msr.core.errors import InvalidAudioError, TranscriptionError
from msr.core.types import AudioClip, SpeakerSegment, TextSegment, TimedToken
from msr.services.audio_io import load_audio, write_clip
from msr.services.model_manager import ModelManager


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
TOKEN_GROUP_PAUSE_SECONDS = 0.4
MAX_ASR_CHUNK_SECONDS = 20.0
CJK_RANGES = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
CHINESE_PUNCTUATION = "，。！？；：、“”‘’（）《》〈〉【】—…·"
StageCallback = Callable[[str, str | None], None]
logger = logging.getLogger(__name__)


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
        logger.info("Task %s started for %s", prepared.task_id, prepared.original_filename)

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
            asr_ranges = _split_ranges_for_asr(vad_ranges, MAX_ASR_CHUNK_SECONDS)
            if len(asr_ranges) != len(vad_ranges):
                logger.info(
                    "Task %s split %s VAD ranges into %s ASR chunks to limit peak memory",
                    prepared.task_id,
                    len(vad_ranges),
                    len(asr_ranges),
                )

            diarization_backend = self.model_manager.get_diarization_backend()
            asr_backend = self.model_manager.get_asr_backend()

            update_stage("diarization", audio_duration)
            speaker_segments = diarization_backend.diarize(prepared.source_path)
            if not speaker_segments:
                speaker_segments = [SpeakerSegment(speaker_id="0", start=0.0, end=audio_duration_seconds)]

            update_stage("asr", audio_duration)
            text_segments: list[TextSegment] = []
            clips = [write_clip(audio, sample_rate, start, end, prepared.task_dir) for start, end in asr_ranges]
            clip_segments = asr_backend.transcribe_many(
                [clip.path for clip in clips],
                language=prepared.language,
                timestamps=True,
            )
            if len(clip_segments) != len(clips):
                raise TranscriptionError("ASR backend returned a mismatched batch result.")
            for clip, segments in zip(clips, clip_segments):
                for segment in segments:
                    adjusted = _offset_text_segment(segment, clip)
                    if adjusted is not None:
                        text_segments.append(adjusted)

            update_stage("postprocess", audio_duration)
            speaker_texts = _match_text_to_speakers(text_segments, speaker_segments)
            speaker_presentations = _build_speaker_presentations(speaker_texts)
            speakers_info = _build_speakers_info(speaker_segments, speaker_texts, speaker_presentations)
            transcripts = _build_transcripts(speaker_texts, speaker_presentations)

        processing_seconds = max(time.perf_counter() - started_at, 0.0)
        processing_speed = audio_duration_seconds / processing_seconds if processing_seconds > 0 else 0.0
        logger.info(
            "Task %s completed in %.2fs for %s of audio",
            prepared.task_id,
            processing_seconds,
            audio_duration,
        )

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


def _split_ranges_for_asr(
    ranges: list[tuple[float, float]],
    max_chunk_seconds: float,
) -> list[tuple[float, float]]:
    if max_chunk_seconds <= 0:
        return list(ranges)

    chunks: list[tuple[float, float]] = []
    for start, end in ranges:
        current = start
        while end - current > max_chunk_seconds:
            chunk_end = current + max_chunk_seconds
            chunks.append((current, chunk_end))
            current = chunk_end
        if end > current:
            chunks.append((current, end))
    return chunks


def _offset_text_segment(segment: TextSegment, clip: AudioClip) -> TextSegment | None:
    text = str(segment.text).strip()
    if not text:
        return None

    tokens = tuple(
        adjusted
        for adjusted in (
            _offset_token(token, clip)
            for token in segment.tokens
        )
        if adjusted is not None
    )

    if tokens:
        start = tokens[0].start
        end = tokens[-1].end
    else:
        start = max(0.0, clip.start + segment.start)
        end = min(clip.end, clip.start + segment.end)

    if end <= start:
        return None

    return TextSegment(
        start=start,
        end=end,
        text=text,
        confidence=segment.confidence,
        tokens=tokens,
    )


def _offset_token(token: TimedToken, clip: AudioClip) -> TimedToken | None:
    start = max(clip.start, clip.start + token.start)
    end = min(clip.end, clip.start + token.end)
    if end <= start:
        return None
    return TimedToken(text=token.text, start=start, end=end, confidence=token.confidence)


def _match_text_to_speakers(text_segments: list[TextSegment], speaker_segments: list[SpeakerSegment]) -> dict[str, list[TextSegment]]:
    ordered_speakers = sorted(
        speaker_segments,
        key=lambda segment: (segment.start, segment.end, _speaker_sort_key(segment.speaker_id)),
    )
    grouped: dict[str, list[TextSegment]] = defaultdict(list)
    for segment in text_segments:
        for speaker_id, matched_segment in _split_text_segment_by_speaker(segment, ordered_speakers):
            grouped[speaker_id].append(matched_segment)

    for segments in grouped.values():
        segments.sort(key=lambda item: (item.start, item.end))
    return grouped


def _split_text_segment_by_speaker(
    segment: TextSegment,
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[str, TextSegment]]:
    if not segment.tokens:
        return [(_best_speaker_for_span(segment.start, segment.end, speaker_segments), segment)]

    items: list[tuple[str, TextSegment]] = []
    current_speaker: str | None = None
    current_tokens: list[TimedToken] = []

    for token in segment.tokens:
        speaker_id = _best_speaker_for_span(token.start, token.end, speaker_segments)
        if current_tokens and (
            speaker_id != current_speaker or token.start - current_tokens[-1].end >= TOKEN_GROUP_PAUSE_SECONDS
        ):
            token_segment = _build_token_text_segment(current_tokens, segment.confidence)
            if token_segment and current_speaker is not None:
                items.append((current_speaker, token_segment))
            current_tokens = []

        if not current_tokens:
            current_speaker = speaker_id
        current_tokens.append(token)

    token_segment = _build_token_text_segment(current_tokens, segment.confidence)
    if token_segment and current_speaker is not None:
        items.append((current_speaker, token_segment))

    if items:
        return items
    return [(_best_speaker_for_span(segment.start, segment.end, speaker_segments), segment)]


def _best_speaker_for_span(start: float, end: float, speaker_segments: list[SpeakerSegment]) -> str:
    if not speaker_segments:
        return "0"

    midpoint = (start + end) / 2
    best_speaker = "0"
    best_overlap = -1.0
    best_distance = float("inf")
    best_sort_key = _speaker_sort_key(best_speaker)

    for speaker_segment in speaker_segments:
        overlap = max(0.0, min(end, speaker_segment.end) - max(start, speaker_segment.start))
        distance = _distance_to_segment(midpoint, speaker_segment)
        sort_key = _speaker_sort_key(speaker_segment.speaker_id)
        if overlap > best_overlap + 1e-9:
            best_speaker = speaker_segment.speaker_id
            best_overlap = overlap
            best_distance = distance
            best_sort_key = sort_key
            continue
        if abs(overlap - best_overlap) <= 1e-9 and (distance, sort_key) < (best_distance, best_sort_key):
            best_speaker = speaker_segment.speaker_id
            best_distance = distance
            best_sort_key = sort_key

    return best_speaker


def _distance_to_segment(point: float, speaker_segment: SpeakerSegment) -> float:
    if speaker_segment.start <= point <= speaker_segment.end:
        return 0.0
    if point < speaker_segment.start:
        return speaker_segment.start - point
    return point - speaker_segment.end


def _build_token_text_segment(tokens: list[TimedToken], confidence: float) -> TextSegment | None:
    if not tokens:
        return None

    start = tokens[0].start
    end = tokens[-1].end
    if end <= start:
        return None

    text = _normalize_text(" ".join(token.text for token in tokens))
    if not text:
        return None

    return TextSegment(start=start, end=end, text=text, confidence=confidence, tokens=tuple(tokens))


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


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    normalized = re.sub(fr"(?<=[{CJK_RANGES}])\s+(?=[{CJK_RANGES}])", "", normalized)
    normalized = re.sub(fr"\s+(?=[{CHINESE_PUNCTUATION}])", "", normalized)
    normalized = re.sub(fr"(?<=[{CHINESE_PUNCTUATION}])\s+", "", normalized)
    return normalized


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
