from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from msr.core.types import TextSegment


class ASRBackend(ABC):
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
    def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
        raise NotImplementedError

    def transcribe_many(
        self,
        audio_paths: Sequence[Path],
        language: str | Sequence[str | None],
        timestamps: bool = True,
    ) -> list[list[TextSegment]]:
        if isinstance(language, str) or language is None:
            languages = [language] * len(audio_paths)
        else:
            languages = list(language)
            if len(languages) != len(audio_paths):
                raise ValueError("language count must match audio_paths in transcribe_many().")

        return [
            self.transcribe(audio_path, language=str(item or "auto"), timestamps=timestamps)
            for audio_path, item in zip(audio_paths, languages)
        ]
