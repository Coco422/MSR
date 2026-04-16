from __future__ import annotations

import importlib.util
from inspect import signature
from pathlib import Path
import sys
from types import ModuleType

from msr.backends.diarization.base import DiarizationBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.runtime_env import format_runtime_context
from msr.core.types import SpeakerSegment
from msr.services.audio_io import load_audio


class ThreeDSpeakerBackend(DiarizationBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._model = None

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        stubbed_modules: list[str] = []
        try:
            if not load_options.get("include_overlap", False):
                stubbed_modules = _install_pyannote_stub_if_missing()
            from speakerlab.bin.infer_diarization import Diarization3Dspeaker

            ctor = signature(Diarization3Dspeaker)
            kwargs = {}
            if "device" in ctor.parameters:
                kwargs["device"] = device
            if "include_overlap" in ctor.parameters:
                kwargs["include_overlap"] = bool(load_options.get("include_overlap", False))
            if "hf_access_token" in ctor.parameters and load_options.get("hf_access_token"):
                kwargs["hf_access_token"] = load_options["hf_access_token"]
            if "speaker_num" in ctor.parameters and load_options.get("speaker_num") is not None:
                kwargs["speaker_num"] = load_options["speaker_num"]
            if "model_cache_dir" in ctor.parameters:
                kwargs["model_cache_dir"] = str(local_path)
            elif "model_dir" in ctor.parameters:
                kwargs["model_dir"] = str(local_path)
            elif "model_path" in ctor.parameters:
                kwargs["model_path"] = str(local_path)

            self._model = Diarization3Dspeaker(**kwargs)
            self.options = load_options
            self.loaded = True
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name == "speakerlab" or missing_name.startswith("speakerlab.") or "speakerlab" in str(exc):
                raise BackendLoadError(
                    "Failed to load 3D-Speaker model from "
                    f"{local_path}: missing dependency 'speakerlab'. "
                    f"Current runtime: {format_runtime_context()}. "
                    "默认链请使用 `bash tools/runtime_env.sh run default`，"
                    "如果你要跑 Qwen + 3D-Speaker，请重新执行 `bash tools/runtime_env.sh setup qwen`。"
                ) from exc
            raise BackendLoadError(f"Failed to load 3D-Speaker model from {local_path}: {exc}") from exc
        except Exception as exc:
            raise BackendLoadError(f"Failed to load 3D-Speaker model from {local_path}: {exc}") from exc
        finally:
            _remove_stubbed_modules(stubbed_modules)

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


def _install_pyannote_stub_if_missing() -> list[str]:
    try:
        if importlib.util.find_spec("pyannote.audio") is not None:
            return []
    except ModuleNotFoundError:
        pass

    message = "pyannote.audio is required when include_overlap=True."
    stubbed_modules: list[str] = []

    pyannote_module = sys.modules.get("pyannote")
    if pyannote_module is None:
        pyannote_module = ModuleType("pyannote")
        sys.modules["pyannote"] = pyannote_module
        stubbed_modules.append("pyannote")

    audio_module = ModuleType("pyannote.audio")

    class _UnavailablePyannote:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(message)

        def __init__(self, *args, **kwargs):
            raise RuntimeError(message)

    audio_module.Inference = _UnavailablePyannote
    audio_module.Model = _UnavailablePyannote
    pyannote_module.audio = audio_module
    sys.modules["pyannote.audio"] = audio_module
    stubbed_modules.append("pyannote.audio")
    return stubbed_modules


def _remove_stubbed_modules(module_names: list[str]) -> None:
    for module_name in reversed(module_names):
        sys.modules.pop(module_name, None)
