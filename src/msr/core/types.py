from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TextSegment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass(slots=True)
class SpeakerSegment:
    speaker_id: str
    start: float
    end: float


@dataclass(slots=True)
class AudioClip:
    path: Path
    start: float
    end: float
