from __future__ import annotations

from abc import ABC, abstractmethod


class VADBackend(ABC):
    @abstractmethod
    def detect(self, audio, sample_rate: int) -> list[tuple[float, float]]:
        raise NotImplementedError
