class MSRError(Exception):
    """Base error for the MSR service."""


class ConfigurationError(MSRError):
    """Raised when application configuration is invalid."""


class ModelBusyError(MSRError):
    """Raised when a model operation conflicts with an active task."""


class ModelNotFoundError(MSRError):
    """Raised when a configured model cannot be found."""


class ModelNotLoadedError(MSRError):
    """Raised when a required model has not been loaded."""


class BackendLoadError(MSRError):
    """Raised when a backend fails to initialize."""


class InvalidAudioError(MSRError):
    """Raised when uploaded audio is invalid."""


class TranscriptionError(MSRError):
    """Raised when transcription fails."""


class QueueFullError(MSRError):
    """Raised when the runtime queue cannot accept more work."""

    def __init__(self, detail: dict):
        super().__init__(detail.get("message", "Task queue is full."))
        self.detail = detail
