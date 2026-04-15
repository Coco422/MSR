from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import warnings

from msr.backends.diarization.base import DiarizationBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.runtime_env import format_runtime_context
from msr.core.types import SpeakerSegment
from msr.services.audio_io import load_audio


logger = logging.getLogger(__name__)


class PyannoteBackend(DiarizationBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._pipeline = None
        self._torch = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        try:
            import torch

            _ensure_torchaudio_pyannote_compat()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"\n?torchcodec is not installed correctly so built-in audio decoding will fail\.",
                    category=UserWarning,
                )
                from pyannote.audio import Pipeline

            checkpoint_path = _resolve_pyannote_checkpoint(local_path)
            logger.info(
                "Initializing pyannote backend model_id=%s device=%s checkpoint=%s",
                self.model_id,
                device,
                checkpoint_path,
            )
            pipeline = Pipeline.from_pretrained(str(checkpoint_path))
            if device.startswith("cuda") and hasattr(pipeline, "to"):
                pipeline.to(torch.device("cuda"))

            self._pipeline = pipeline
            self._torch = torch
            self.options = load_options
            self.loaded = True
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name in {"pyannote", "pyannote.audio"} or "pyannote.audio" in str(exc):
                raise BackendLoadError(
                    "Failed to load pyannote pipeline from "
                    f"{local_path}: missing dependency 'pyannote.audio'. "
                    f"Current runtime: {format_runtime_context()}. "
                    "请使用 `bash tools/runtime_env.sh run pyannote` 启动服务，或先执行 "
                    "`bash tools/runtime_env.sh setup pyannote` 创建独立环境。"
                ) from exc
            raise BackendLoadError(f"Failed to load pyannote pipeline from {local_path}: {exc}") from exc
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
            waveform, sample_rate = load_audio(audio_path)
            annotation = self._pipeline(
                {
                    "waveform": self._torch.from_numpy(waveform).unsqueeze(0),
                    "sample_rate": sample_rate,
                },
                **diarize_kwargs,
            )
        except TypeError:
            annotation = self._pipeline(
                {
                    "waveform": self._torch.from_numpy(waveform).unsqueeze(0),
                    "sample_rate": sample_rate,
                },
                **diarize_kwargs,
            )
        except Exception as exc:
            raise TranscriptionError(f"pyannote diarization failed: {exc}") from exc

        annotation = _extract_pyannote_annotation(annotation)
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


@dataclass(slots=True)
class _AudioMetaDataCompat:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def _ensure_torchaudio_pyannote_compat() -> None:
    import soundfile as sf
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _AudioMetaDataCompat

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "info"):

        def _info(file: Any, backend: str | None = None) -> _AudioMetaDataCompat:
            audio_source = file.get("audio") if isinstance(file, dict) else file
            metadata = sf.info(str(audio_source))
            subtype_info = str(metadata.subtype_info or "")
            bit_depth = "".join(char for char in subtype_info if char.isdigit())
            return _AudioMetaDataCompat(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
                bits_per_sample=int(bit_depth) if bit_depth else 0,
                encoding=str(metadata.subtype or metadata.format or "UNKNOWN"),
            )

        torchaudio.info = _info


def _resolve_pyannote_checkpoint(local_path: Path) -> Path:
    resolved_path = local_path.expanduser().resolve()
    if resolved_path.is_dir():
        return resolved_path / "config.yaml"
    return resolved_path


def _extract_pyannote_annotation(result: Any) -> Any:
    if hasattr(result, "itertracks"):
        return result
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    return result
