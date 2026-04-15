from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from msr.backends.asr.base import ASRBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.types import TextSegment
from msr.services.audio_io import probe_duration

_FUNASR_PAUSE_SPLIT_SECONDS = 0.4
_FUNASR_MAX_SEGMENT_SECONDS = 6.0
_CJK_RANGES = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CHINESE_PUNCTUATION = "，。！？；：、“”‘’（）《》〈〉【】—…·"


class FunASRBackend(ASRBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._model = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=str(local_path),
                device=_normalize_device(device),
                disable_update=bool(load_options.get("disable_update", True)),
            )
            self.options = load_options
            self.loaded = True
        except Exception as exc:
            raise BackendLoadError(f"Failed to load FunASR model from {local_path}: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
        if not self._model:
            raise TranscriptionError("FunASR backend is not loaded.")

        try:
            batch_size_s = self.options.get("batch_size_s", 60)
            raw = self._model.generate(
                input=str(audio_path),
                cache={},
                language=language,
                batch_size_s=batch_size_s,
            )
        except TypeError:
            raw = self._model.generate(input=str(audio_path), cache={}, batch_size_s=self.options.get("batch_size_s", 60))
        except Exception as exc:
            raise TranscriptionError(f"FunASR transcription failed: {exc}") from exc

        return _parse_funasr_result(raw, audio_path)


def _normalize_device(device: str) -> str:
    if device.startswith("cuda") and ":" not in device:
        return "cuda:0"
    return device


def _parse_funasr_result(raw: Any, audio_path: Path) -> list[TextSegment]:
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    duration = probe_duration(audio_path)
    segments: list[TextSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list):
            sentence_segments = _parse_sentence_info(sentence_info, duration)
            if sentence_segments:
                segments.extend(sentence_segments)
                continue

        timestamp = item.get("timestamp")
        raw_text = str(item.get("text", "")).strip()
        if isinstance(timestamp, list) and raw_text:
            timestamp_segments = _parse_timestamp_segments(raw_text, timestamp, duration)
            if timestamp_segments:
                segments.extend(timestamp_segments)
                continue

        text = _normalize_text(raw_text)
        if text:
            segments.append(TextSegment(start=0.0, end=duration, text=text))

    if segments:
        return segments

    return [TextSegment(start=0.0, end=duration, text="[Empty transcription]")]


def _parse_sentence_info(sentence_info: list[Any], audio_duration: float) -> list[TextSegment]:
    pairs = []
    for sentence in sentence_info:
        if not isinstance(sentence, dict):
            continue
        pairs.append((sentence.get("start", 0.0), sentence.get("end", audio_duration)))

    scale = _infer_timestamp_scale(pairs, audio_duration)
    segments: list[TextSegment] = []
    for sentence in sentence_info:
        if not isinstance(sentence, dict):
            continue
        text = _normalize_text(str(sentence.get("text", "")).strip())
        if not text:
            continue
        start = _timestamp_to_seconds(sentence.get("start", 0.0), scale)
        end = _timestamp_to_seconds(sentence.get("end", audio_duration), scale)
        if end <= start:
            continue
        segments.append(TextSegment(start=start, end=end, text=text))
    return segments


def _parse_timestamp_segments(raw_text: str, timestamp: list[Any], audio_duration: float) -> list[TextSegment]:
    pairs = _timestamp_pairs(timestamp)
    if not pairs:
        return []

    tokens = [token for token in raw_text.split() if token]
    scale = _infer_timestamp_scale(pairs, audio_duration)

    if len(tokens) != len(pairs):
        text = _normalize_text(raw_text)
        if not text:
            return []
        start = _timestamp_to_seconds(pairs[0][0], scale)
        end = _timestamp_to_seconds(pairs[-1][1], scale)
        if end <= start:
            return []
        return [TextSegment(start=start, end=end, text=text)]

    segments: list[TextSegment] = []
    current_tokens: list[str] = []
    current_start = 0.0
    current_end = 0.0

    for token, (start_raw, end_raw) in zip(tokens, pairs):
        start = _timestamp_to_seconds(start_raw, scale)
        end = _timestamp_to_seconds(end_raw, scale)
        if end <= start:
            continue

        if not current_tokens:
            current_tokens = [token]
            current_start = start
            current_end = end
            continue

        gap = max(0.0, start - current_end)
        duration = current_end - current_start
        if gap >= _FUNASR_PAUSE_SPLIT_SECONDS or duration >= _FUNASR_MAX_SEGMENT_SECONDS:
            segment = _build_text_segment(current_tokens, current_start, current_end)
            if segment:
                segments.append(segment)
            current_tokens = [token]
            current_start = start
            current_end = end
            continue

        current_tokens.append(token)
        current_end = end

    segment = _build_text_segment(current_tokens, current_start, current_end)
    if segment:
        segments.append(segment)
    return segments


def _build_text_segment(tokens: list[str], start: float, end: float) -> TextSegment | None:
    text = _normalize_text(" ".join(tokens))
    if not text or end <= start:
        return None
    return TextSegment(start=start, end=end, text=text)


def _timestamp_pairs(timestamp: list[Any]) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for item in timestamp:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((item[0], item[1]))
    return pairs


def _infer_timestamp_scale(pairs: list[tuple[Any, Any]], audio_duration: float) -> float:
    values = []
    for start, end in pairs:
        for value in (start, end):
            try:
                values.append(abs(float(value)))
            except (TypeError, ValueError):
                continue

    if not values:
        return 1.0

    max_value = max(values)
    if max_value > max(audio_duration * 2 + 1.0, 10.0):
        return 0.001
    return 1.0


def _timestamp_to_seconds(value: Any, scale: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric * scale


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    normalized = re.sub(fr"(?<=[{_CJK_RANGES}])\s+(?=[{_CJK_RANGES}])", "", normalized)
    normalized = re.sub(fr"\s+(?=[{_CHINESE_PUNCTUATION}])", "", normalized)
    normalized = re.sub(fr"(?<=[{_CHINESE_PUNCTUATION}])\s+", "", normalized)
    return normalized
