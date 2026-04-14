from __future__ import annotations

from pathlib import Path

from msr.backends.diarization.base import DiarizationBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.types import SpeakerSegment


class PyannoteBackend(DiarizationBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._pipeline = None
        self._torch = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        try:
            import torch
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(str(local_path))
            if device.startswith("cuda") and hasattr(pipeline, "to"):
                pipeline.to(torch.device("cuda"))

            self._pipeline = pipeline
            self._torch = torch
            self.options = load_options
            self.loaded = True
        except Exception as exc:
            raise BackendLoadError(f"Failed to load pyannote pipeline from {local_path}: {exc}") from exc

    def unload(self) -> None:
        self._pipeline = None
        self._torch = None
        self.loaded = False

    def diarize(self, audio_path: Path, speaker_hints: dict | None = None) -> list[SpeakerSegment]:
        if not self._pipeline:
            raise TranscriptionError("pyannote backend is not loaded.")

        diarize_kwargs = {}
        hints = speaker_hints or {}
        if "min_speakers" in hints:
            diarize_kwargs["min_speakers"] = hints["min_speakers"]
        if "max_speakers" in hints:
            diarize_kwargs["max_speakers"] = hints["max_speakers"]

        try:
            annotation = self._pipeline(str(audio_path), **diarize_kwargs)
        except TypeError:
            annotation = self._pipeline({"audio": str(audio_path)}, **diarize_kwargs)
        except Exception as exc:
            raise TranscriptionError(f"pyannote diarization failed: {exc}") from exc

        segments = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker_id=str(label),
                    start=float(segment.start),
                    end=float(segment.end),
                )
            )
        return segments
