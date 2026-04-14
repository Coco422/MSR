from __future__ import annotations

from dataclasses import dataclass

from msr.core.config import Settings
from msr.services.model_manager import ModelManager
from msr.services.resource_monitor import ResourceMonitor
from msr.services.transcription_service import TranscriptionService


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    model_manager: ModelManager
    resource_monitor: ResourceMonitor
    transcription_service: TranscriptionService

    @classmethod
    def from_settings(cls, settings: Settings) -> "ServiceContainer":
        model_manager = ModelManager(settings)
        resource_monitor = ResourceMonitor()
        transcription_service = TranscriptionService(
            settings=settings,
            model_manager=model_manager,
        )
        return cls(
            settings=settings,
            model_manager=model_manager,
            resource_monitor=resource_monitor,
            transcription_service=transcription_service,
        )
