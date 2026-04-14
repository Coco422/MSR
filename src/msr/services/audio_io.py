from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import soundfile as sf

from msr.core.types import AudioClip


def load_audio(path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    if not audio.flags.c_contiguous:
        audio = np.ascontiguousarray(audio)
    if len(audio):
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
    return audio, sr


def write_clip(audio: np.ndarray, sample_rate: int, start: float, end: float, directory: Path) -> AudioClip:
    start_idx = max(0, int(start * sample_rate))
    end_idx = max(start_idx + 1, int(end * sample_rate))
    clip_audio = audio[start_idx:end_idx]
    with NamedTemporaryFile(prefix="clip_", suffix=".wav", dir=directory, delete=False) as handle:
        clip_path = Path(handle.name)
    sf.write(str(clip_path), clip_audio, sample_rate)
    return AudioClip(path=clip_path, start=start, end=end)


def probe_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)
