from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from msr.backends.asr.base import ASRBackend
from msr.backends.asr.faster_whisper_backend import FasterWhisperBackend
from msr.backends.asr.funasr_backend import FunASRBackend
from msr.backends.asr.qwen_asr_backend import QwenASRBackend
from msr.backends.diarization.base import DiarizationBackend
from msr.backends.diarization.pyannote_backend import PyannoteBackend
from msr.backends.diarization.three_d_speaker_backend import ThreeDSpeakerBackend
from msr.core.config import ModelConfig, Settings
from msr.core.errors import ModelBusyError, ModelNotFoundError, ModelNotLoadedError
from msr.core.runtime_env import format_runtime_context


ASR_FACTORIES = {
    "funasr": FunASRBackend,
    "faster_whisper": FasterWhisperBackend,
    "qwen_asr": QwenASRBackend,
}

DIARIZATION_FACTORIES = {
    "3d_speaker": ThreeDSpeakerBackend,
    "pyannote": PyannoteBackend,
}
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedModel:
    config: ModelConfig
    backend: Any


@dataclass(slots=True)
class ModelOperation:
    kind: str
    action: str
    model_id: str


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._active_tasks = 0
        self._inflight_tasks_provider: Callable[[], int] | None = None
        self._registry = self._build_registry(settings.models)
        self._loaded: dict[str, LoadedModel | None] = {
            "asr": None,
            "diarization": None,
        }
        self._operation: ModelOperation | None = None

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
                operation = self._operation if self._operation and self._operation.kind == kind else None
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
                            "operation_status": operation.action if operation and operation.model_id == model.id else None,
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
                        "operation_status": self._operation.action
                        if self._operation and self._operation.kind == kind
                        else None,
                    }
                    if loaded
                    else (
                        {
                            "operation_status": self._operation.action,
                            "target_model_id": self._operation.model_id,
                        }
                        if self._operation and self._operation.kind == kind
                        else None
                    )
                )
                for kind, loaded in self._loaded.items()
            }

    def bind_inflight_tasks_provider(self, provider: Callable[[], int]) -> None:
        self._inflight_tasks_provider = provider

    def ensure_transcription_ready(self) -> None:
        with self._lock:
            if not self._loaded["asr"] or not self._loaded["diarization"]:
                raise ModelNotLoadedError("Both ASR and diarization models must be loaded before transcription.")

    def load(self, kind: str, model_id: str) -> dict[str, Any]:
        previous_loaded: LoadedModel | None = None
        operation = ModelOperation(kind=kind, action="loading", model_id=model_id)
        with self._lock:
            self._ensure_model_operation_allowed_locked(operation)
            config = self._get_model(kind, model_id)

            current = self._loaded.get(kind)
            if current and current.config.id == config.id:
                logger.info(
                    "Model load skipped kind=%s model_id=%s because it is already active",
                    kind,
                    model_id,
                )
                return self.active_state()[kind] or {}

            self._operation = operation
            previous_loaded = current
            if current:
                logger.info(
                    "Unloading previous %s model model_id=%s before loading model_id=%s",
                    kind,
                    current.config.id,
                    model_id,
                )
                self._loaded[kind] = None

        previous_unloaded = previous_loaded is None
        try:
            if previous_loaded:
                previous_loaded.backend.unload()
                previous_unloaded = True

            logger.info(
                "Loading model kind=%s model_id=%s backend=%s device=%s path=%s %s",
                config.kind,
                config.id,
                config.backend,
                config.device,
                config.local_path,
                format_runtime_context(),
            )
            backend = self._instantiate(config)
            try:
                backend.load(config.local_path, config.device, config.options)
            except Exception:
                logger.exception(
                    "Model load failed kind=%s model_id=%s backend=%s path=%s",
                    config.kind,
                    config.id,
                    config.backend,
                    config.local_path,
                )
                raise
            loaded_model = LoadedModel(config=config, backend=backend)
            with self._lock:
                self._loaded[kind] = loaded_model
                self._clear_model_operation_locked(operation)
                state = self.active_state()[kind] or {}
            logger.info(
                "Model loaded kind=%s model_id=%s active_state=%s",
                kind,
                model_id,
                state,
            )
            return state
        except Exception:
            with self._lock:
                if previous_loaded and not previous_unloaded:
                    self._loaded[kind] = previous_loaded
                self._clear_model_operation_locked(operation)
            raise

    def unload(self, kind: str, model_id: str) -> None:
        operation = ModelOperation(kind=kind, action="unloading", model_id=model_id)
        with self._lock:
            self._ensure_model_operation_allowed_locked(operation)
            loaded = self._loaded.get(kind)
            if not loaded or loaded.config.id != model_id:
                raise ModelNotLoadedError(f"Model {model_id} is not currently loaded for kind {kind}.")
            self._operation = operation
            self._loaded[kind] = None
        try:
            logger.info(
                "Unloading model kind=%s model_id=%s backend=%s path=%s",
                kind,
                model_id,
                loaded.config.backend,
                loaded.config.local_path,
            )
            loaded.backend.unload()
            with self._lock:
                self._clear_model_operation_locked(operation)
            logger.info("Model unloaded kind=%s model_id=%s", kind, model_id)
        except Exception:
            with self._lock:
                self._loaded[kind] = loaded
                self._clear_model_operation_locked(operation)
            raise

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

    def _busy_task_count(self) -> int:
        provider_count = self._inflight_tasks_provider() if self._inflight_tasks_provider else 0
        return max(self._active_tasks, provider_count)

    def _ensure_model_operation_allowed_locked(self, operation: ModelOperation) -> None:
        if self._busy_task_count():
            raise ModelBusyError("Cannot load or unload models while transcription tasks are active or queued.")
        if self._operation is not None:
            raise ModelBusyError(
                "Another model operation is already in progress: "
                f"{self._operation.action} {self._operation.kind}:{self._operation.model_id}."
            )

    def _clear_model_operation_locked(self, operation: ModelOperation) -> None:
        if (
            self._operation
            and self._operation.kind == operation.kind
            and self._operation.action == operation.action
            and self._operation.model_id == operation.model_id
        ):
            self._operation = None
