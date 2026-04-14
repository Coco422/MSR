from __future__ import annotations

from pathlib import Path
from typing import Any

from msr.backends.asr.base import ASRBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.types import TextSegment
from msr.services.audio_io import probe_duration


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
            for sentence in sentence_info:
                text = str(sentence.get("text", "")).strip()
                if not text:
                    continue
                start = _to_seconds(sentence.get("start", 0.0))
                end = _to_seconds(sentence.get("end", duration))
                segments.append(TextSegment(start=start, end=end, text=text))
            continue

        timestamp = item.get("timestamp")
        text = str(item.get("text", "")).strip()
        if isinstance(timestamp, list) and text:
            if timestamp and isinstance(timestamp[0], (list, tuple)) and len(timestamp[0]) >= 2:
                start = _to_seconds(timestamp[0][0])
                end = _to_seconds(timestamp[-1][1])
                segments.append(TextSegment(start=start, end=end, text=text))
                continue

        if text:
            segments.append(TextSegment(start=0.0, end=duration, text=text))

    if segments:
        return segments

    return [TextSegment(start=0.0, end=duration, text="[Empty transcription]")]


def _to_seconds(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1000:
        return numeric / 1000.0
    return numeric
