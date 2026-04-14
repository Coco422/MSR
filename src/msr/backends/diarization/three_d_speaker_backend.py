from __future__ import annotations

from inspect import signature
from pathlib import Path

from msr.backends.diarization.base import DiarizationBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.types import SpeakerSegment
from msr.services.audio_io import load_audio


class ThreeDSpeakerBackend(DiarizationBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._model = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        try:
            from speakerlab.bin.infer_diarization import Diarization3Dspeaker

            ctor = signature(Diarization3Dspeaker)
            kwargs = {}
            if "model_cache_dir" in ctor.parameters:
                kwargs["model_cache_dir"] = str(local_path)
            elif "model_dir" in ctor.parameters:
                kwargs["model_dir"] = str(local_path)
            elif "model_path" in ctor.parameters:
                kwargs["model_path"] = str(local_path)

            self._model = Diarization3Dspeaker(**kwargs)
            self.options = load_options
            self.loaded = True
        except Exception as exc:
            raise BackendLoadError(f"Failed to load 3D-Speaker model from {local_path}: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def diarize(self, audio_path: Path, speaker_hints: dict | None = None) -> list[SpeakerSegment]:
        if not self._model:
            raise TranscriptionError("3D-Speaker backend is not loaded.")

        try:
            waveform, sample_rate = load_audio(audio_path)
            raw_segments = self._model(waveform.copy(), wav_fs=sample_rate)
        except Exception as exc:
            raise TranscriptionError(f"3D-Speaker diarization failed: {exc}") from exc

        segments = []
        for raw in raw_segments or []:
            if len(raw) < 3:
                continue
            start, end, speaker_id = raw[0], raw[1], raw[2]
            segments.append(
                SpeakerSegment(
                    speaker_id=str(speaker_id),
                    start=float(start),
                    end=float(end),
                )
            )
        return segments
