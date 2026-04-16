from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any

from msr.core.errors import ConfigurationError


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


@dataclass(slots=True)
class SecurityConfig:
    api_key: str
    api_key_header: str = "X-API-Key"


@dataclass(slots=True)
class AppConfig:
    name: str
    service_name: str
    version: str
    host: str
    port: int
    log_level: str
    default_language: str
    temp_dir: Path
    strict_offline: bool


@dataclass(slots=True)
class WebConfig:
    title: str
    resource_refresh_seconds: int = 3


@dataclass(slots=True)
class RuntimeConfig:
    max_parallel_tasks: int = 1
    max_queued_tasks: int = 2
    recent_task_limit: int = 50
    data_dir: Path = Path("data")

    @property
    def override_path(self) -> Path:
        return self.data_dir / "runtime_overrides.toml"

    @property
    def recent_tasks_path(self) -> Path:
        return self.data_dir / "recent_tasks.json"

    @property
    def task_results_dir(self) -> Path:
        return self.data_dir / "task_results"

    def task_result_path(self, task_id: str) -> Path:
        return self.task_results_dir / f"{task_id}.json"


@dataclass(slots=True)
class ModelConfig:
    id: str
    kind: str
    backend: str
    local_path: Path
    device: str = "cpu"
    enabled: bool = True
    default: bool = False
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Settings:
    project_root: Path
    app: AppConfig
    security: SecurityConfig
    web: WebConfig
    runtime: RuntimeConfig
    models: list[ModelConfig]

    @property
    def offline_env(self) -> dict[str, str]:
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "MS_SDK_OFFLINE": "1",
            "AIMEETING_STRICT_OFFLINE": "1",
        }


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or _project_root()
    app_config_path = root / "config" / "app.toml"
    models_config_path = root / "config" / "models.toml"

    app_raw = _load_toml(app_config_path)
    models_raw = _load_toml(models_config_path)

    app_section = app_raw.get("app", {})
    security_section = app_raw.get("security", {})
    web_section = app_raw.get("web", {})
    runtime_section = dict(app_raw.get("runtime", {}))

    temp_dir = Path(app_section.get("temp_dir", "tmp"))
    if not temp_dir.is_absolute():
        temp_dir = root / temp_dir

    data_dir = Path(runtime_section.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    runtime_override = _load_toml(data_dir / "runtime_overrides.toml").get("runtime", {})
    runtime_section.update(runtime_override)

    api_key = os.getenv("MSR_API_KEY", security_section.get("api_key", ""))
    if not api_key:
        raise ConfigurationError("Missing API key. Set MSR_API_KEY or security.api_key.")

    models = []
    for raw in models_raw.get("models", []):
        model_path = Path(raw["local_path"])
        if not model_path.is_absolute():
            model_path = root / model_path
        options = _normalize_model_options(raw.get("options", {}), root)
        models.append(
            ModelConfig(
                id=raw["id"],
                kind=raw["kind"],
                backend=raw["backend"],
                local_path=model_path,
                device=raw.get("device", "cpu"),
                enabled=bool(raw.get("enabled", True)),
                default=bool(raw.get("default", False)),
                options=options,
            )
        )

    return Settings(
        project_root=root,
        app=AppConfig(
            name=app_section.get("name", "MSR"),
            service_name=app_section.get("service_name", "Multi Speaker Recognization"),
            version=app_section.get("version", "0.1.0"),
            host=app_section.get("host", "0.0.0.0"),
            port=int(app_section.get("port", 8011)),
            log_level=app_section.get("log_level", "INFO"),
            default_language=app_section.get("default_language", "zh"),
            temp_dir=temp_dir,
            strict_offline=bool(app_section.get("strict_offline", True)),
        ),
        security=SecurityConfig(
            api_key=api_key,
            api_key_header=security_section.get("api_key_header", "X-API-Key"),
        ),
        web=WebConfig(
            title=web_section.get("title", "MSR Console"),
            resource_refresh_seconds=int(web_section.get("resource_refresh_seconds", 3)),
        ),
        runtime=RuntimeConfig(
            max_parallel_tasks=_read_int(runtime_section, "max_parallel_tasks", 1, minimum=1),
            max_queued_tasks=_read_int(runtime_section, "max_queued_tasks", 2, minimum=0),
            recent_task_limit=_read_int(runtime_section, "recent_task_limit", 50, minimum=1),
            data_dir=data_dir,
        ),
        models=models,
    )


def _read_int(section: dict[str, Any], key: str, default: int, minimum: int) -> int:
    raw = section.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid integer for runtime.{key}: {raw}") from exc
    if value < minimum:
        raise ConfigurationError(f"runtime.{key} must be >= {minimum}, got {value}")
    return value


def _normalize_model_options(options: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(options).items():
        if key.endswith("_path") and isinstance(value, str):
            path = Path(value)
            normalized[key] = path if path.is_absolute() else root / path
            continue
        normalized[key] = value
    return normalized
