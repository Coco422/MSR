from __future__ import annotations

from pathlib import Path

from msr.backends.asr.base import ASRBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.types import TextSegment
from msr.services.audio_io import probe_duration


class FasterWhisperBackend(ASRBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._model = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        try:
            from faster_whisper import WhisperModel

            compute_type = load_options.get("compute_type", "float16" if device.startswith("cuda") else "int8")
            self._model = WhisperModel(
                model_size_or_path=str(local_path),
                device="cuda" if device.startswith("cuda") else "cpu",
                compute_type=compute_type,
                local_files_only=True,
            )
            self.options = load_options
            self.loaded = True
        except Exception as exc:
            raise BackendLoadError(f"Failed to load faster-whisper model from {local_path}: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
        if not self._model:
            raise TranscriptionError("faster-whisper backend is not loaded.")

        try:
            segments, _ = self._model.transcribe(
                str(audio_path),
                beam_size=int(self.options.get("beam_size", 5)),
                language=None if language == "auto" else language,
                vad_filter=False,
                word_timestamps=False,
            )
            items = [
                TextSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text).strip(),
                )
                for segment in segments
                if str(segment.text).strip()
            ]
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper transcription failed: {exc}") from exc

        if items:
            return items

        duration = probe_duration(audio_path)
        return [TextSegment(start=0.0, end=duration, text="[Empty transcription]")]
