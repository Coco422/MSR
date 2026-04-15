from __future__ import annotations

from dataclasses import dataclass

from msr.core.config import Settings
from msr.services.model_manager import ModelManager
from msr.services.resource_monitor import ResourceMonitor
from msr.services.task_manager import TaskManager
from msr.services.transcription_service import TranscriptionService


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    model_manager: ModelManager
    resource_monitor: ResourceMonitor
    task_manager: TaskManager
    transcription_service: TranscriptionService

    @classmethod
    def from_settings(cls, settings: Settings) -> "ServiceContainer":
        model_manager = ModelManager(settings)
        resource_monitor = ResourceMonitor()
        task_manager = TaskManager(settings)
        model_manager.bind_inflight_tasks_provider(task_manager.inflight_count)
        transcription_service = TranscriptionService(
            settings=settings,
            model_manager=model_manager,
        )
        return cls(
            settings=settings,
            model_manager=model_manager,
            resource_monitor=resource_monitor,
            task_manager=task_manager,
            transcription_service=transcription_service,
        )
