from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from msr.backends.asr.base import ASRBackend
from msr.backends.asr.faster_whisper_backend import FasterWhisperBackend
from msr.backends.asr.funasr_backend import FunASRBackend
from msr.backends.diarization.base import DiarizationBackend
from msr.backends.diarization.pyannote_backend import PyannoteBackend
from msr.backends.diarization.three_d_speaker_backend import ThreeDSpeakerBackend
from msr.core.config import ModelConfig, Settings
from msr.core.errors import ModelBusyError, ModelNotFoundError, ModelNotLoadedError


ASR_FACTORIES = {
    "funasr": FunASRBackend,
    "faster_whisper": FasterWhisperBackend,
}

DIARIZATION_FACTORIES = {
    "3d_speaker": ThreeDSpeakerBackend,
    "pyannote": PyannoteBackend,
}


@dataclass(slots=True)
class LoadedModel:
    config: ModelConfig
    backend: Any


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._active_tasks = 0
        self._registry = self._build_registry(settings.models)
        self._loaded: dict[str, LoadedModel | None] = {
            "asr": None,
            "diarization": None,
        }

    def _build_registry(self, models: list[ModelConfig]) -> dict[str, dict[str, ModelConfig]]:
        registry: dict[str, dict[str, ModelConfig]] = {"asr": {}, "diarization": {}}
        for model in models:
            registry.setdefault(model.kind, {})
            registry[model.kind][model.id] = model
        return registry

    @contextmanager
    def task_scope(self):
        with self._lock:
            if not self._loaded["asr"] or not self._loaded["diarization"]:
                raise ModelNotLoadedError("Both ASR and diarization models must be loaded before transcription.")
            self._active_tasks += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_tasks = max(0, self._active_tasks - 1)

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for kind, kind_models in self._registry.items():
                active = self._loaded.get(kind)
                for model in kind_models.values():
                    items.append(
                        {
                            "id": model.id,
                            "kind": model.kind,
                            "backend": model.backend,
                            "local_path": str(model.local_path),
                            "device": model.device,
                            "enabled": model.enabled,
                            "default": model.default,
                            "path_exists": model.local_path.exists(),
                            "loaded": bool(active and active.config.id == model.id),
                        }
                    )
            return items

    def active_state(self) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            return {
                kind: (
                    {
                        "id": loaded.config.id,
                        "backend": loaded.config.backend,
                        "local_path": str(loaded.config.local_path),
                        "device": loaded.config.device,
                    }
                    if loaded
                    else None
                )
                for kind, loaded in self._loaded.items()
            }

    def load(self, kind: str, model_id: str) -> dict[str, Any]:
        with self._lock:
            if self._active_tasks:
                raise ModelBusyError("Cannot load models while transcription is running.")
            config = self._get_model(kind, model_id)

            current = self._loaded.get(kind)
            if current and current.config.id == config.id:
                return self.active_state()[kind] or {}

            if current:
                current.backend.unload()
                self._loaded[kind] = None

            backend = self._instantiate(config)
            backend.load(config.local_path, config.device, config.options)
            self._loaded[kind] = LoadedModel(config=config, backend=backend)
            return self.active_state()[kind] or {}

    def unload(self, kind: str, model_id: str) -> None:
        with self._lock:
            if self._active_tasks:
                raise ModelBusyError("Cannot unload models while transcription is running.")
            loaded = self._loaded.get(kind)
            if not loaded or loaded.config.id != model_id:
                raise ModelNotLoadedError(f"Model {model_id} is not currently loaded for kind {kind}.")
            loaded.backend.unload()
            self._loaded[kind] = None

    def get_asr_backend(self) -> ASRBackend:
        loaded = self._loaded.get("asr")
        if not loaded:
            raise ModelNotLoadedError("ASR model is not loaded.")
        return loaded.backend

    def get_diarization_backend(self) -> DiarizationBackend:
        loaded = self._loaded.get("diarization")
        if not loaded:
            raise ModelNotLoadedError("Diarization model is not loaded.")
        return loaded.backend

    def _get_model(self, kind: str, model_id: str) -> ModelConfig:
        config = self._registry.get(kind, {}).get(model_id)
        if not config:
            raise ModelNotFoundError(f"Unknown {kind} model: {model_id}")
        if not config.enabled:
            raise ModelNotFoundError(f"Model {model_id} is disabled.")
        if not config.local_path.exists():
            raise ModelNotFoundError(f"Configured model path does not exist: {config.local_path}")
        return config

    def _instantiate(self, config: ModelConfig):
        factories = ASR_FACTORIES if config.kind == "asr" else DIARIZATION_FACTORIES
        factory = factories.get(config.backend)
        if not factory:
            raise ModelNotFoundError(f"Unsupported backend: {config.backend}")
        return factory(model_id=config.id, options=config.options)
