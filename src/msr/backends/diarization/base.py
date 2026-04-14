from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from msr.core.types import SpeakerSegment


class DiarizationBackend(ABC):
    def __init__(self, model_id: str, options: dict | None = None):
        self.model_id = model_id
        self.options = options or {}
        self.loaded = False

    @abstractmethod
    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def diarize(self, audio_path: Path, speaker_hints: dict | None = None) -> list[SpeakerSegment]:
        raise NotImplementedError
