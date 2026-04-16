from __future__ import annotations

import importlib
import logging
from pathlib import Path
import re
from typing import Any, Sequence

from msr.backends.asr.base import ASRBackend
from msr.core.errors import BackendLoadError, TranscriptionError
from msr.core.runtime_env import format_runtime_context
from msr.core.types import TextSegment, TimedToken
from msr.services.audio_io import probe_duration


logger = logging.getLogger(__name__)

_LANGUAGE_MAP = {
    "zh": "Chinese",
    "cmn": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


class QwenASRBackend(ASRBackend):
    def __init__(self, model_id: str, options: dict | None = None):
        super().__init__(model_id=model_id, options=options)
        self._model = None
        self._engine = "vllm"

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        load_options = {**self.options, **(options or {})}
        engine = str(load_options.get("engine", "vllm")).strip().lower() or "vllm"
        if engine != "vllm":
            raise BackendLoadError(
                f"Failed to load Qwen3-ASR model from {local_path}: unsupported engine '{engine}', only 'vllm' is supported."
            )

        forced_aligner_path = _require_forced_aligner_path(load_options, local_path)

        try:
            importlib.import_module("qwen_asr")
        except ModuleNotFoundError as exc:
            raise _missing_qwen_dependency_error(local_path, exc) from exc

        try:
            importlib.import_module("vllm")
        except ModuleNotFoundError as exc:
            raise _missing_qwen_dependency_error(local_path, exc) from exc

        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            gpu_memory_utilization = float(load_options.get("gpu_memory_utilization", 0.7))
            max_inference_batch_size = int(load_options.get("max_inference_batch_size", 32))
            max_new_tokens = int(load_options.get("max_new_tokens", 1024))
            max_model_len = int(load_options.get("max_model_len", 16384))
            forced_aligner_dtype = _resolve_torch_dtype(
                torch,
                str(load_options.get("forced_aligner_dtype", "float16")),
            )
            logger.info(
                "Initializing Qwen3-ASR backend model_id=%s engine=%s device=%s path=%s forced_aligner=%s gpu_memory_utilization=%.2f max_model_len=%s max_inference_batch_size=%s",
                self.model_id,
                engine,
                device,
                local_path,
                forced_aligner_path,
                gpu_memory_utilization,
                max_model_len,
                max_inference_batch_size,
            )
            self._model = Qwen3ASRModel.LLM(
                model=str(local_path),
                gpu_memory_utilization=gpu_memory_utilization,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                max_model_len=max_model_len,
                forced_aligner=str(forced_aligner_path),
                forced_aligner_kwargs={
                    "dtype": forced_aligner_dtype,
                    "device_map": _normalize_device(device),
                },
            )
            self.options = load_options
            self._engine = engine
            self.loaded = True
        except ModuleNotFoundError as exc:
            raise _missing_qwen_dependency_error(local_path, exc) from exc
        except Exception as exc:
            raise _normalize_qwen_load_error(local_path, exc, load_options) from exc

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
        if not self._model:
            raise TranscriptionError("Qwen3-ASR backend is not loaded.")

        try:
            results = self._model.transcribe(
                audio=str(audio_path),
                language=_map_qwen_language(language),
                return_time_stamps=bool(timestamps),
            )
        except Exception as exc:
            raise TranscriptionError(f"Qwen3-ASR transcription failed: {exc}") from exc

        return _convert_qwen_results(results, [audio_path])[0]

    def transcribe_many(
        self,
        audio_paths: Sequence[Path],
        language: str | Sequence[str | None],
        timestamps: bool = True,
    ) -> list[list[TextSegment]]:
        if not self._model:
            raise TranscriptionError("Qwen3-ASR backend is not loaded.")
        if not audio_paths:
            return []
        if len(audio_paths) == 1:
            single_language = language[0] if isinstance(language, Sequence) and not isinstance(language, str) else language
            return [self.transcribe(audio_paths[0], language=str(single_language or "auto"), timestamps=timestamps)]

        languages = _expand_qwen_languages(audio_paths, language)

        try:
            results = self._model.transcribe(
                audio=[str(path) for path in audio_paths],
                language=languages,
                return_time_stamps=bool(timestamps),
            )
        except Exception as exc:
            raise TranscriptionError(f"Qwen3-ASR transcription failed: {exc}") from exc

        return _convert_qwen_results(results, audio_paths)


def _convert_qwen_results(results: Any, audio_paths: Sequence[Path]) -> list[list[TextSegment]]:
    if not isinstance(results, list):
        raise TranscriptionError("Qwen3-ASR returned an unexpected result payload.")
    if len(results) != len(audio_paths):
        raise TranscriptionError(
            f"Qwen3-ASR returned {len(results)} results for {len(audio_paths)} audio clips."
        )
    return [_result_to_segments(result, audio_path) for result, audio_path in zip(results, audio_paths)]


def _result_to_segments(result: Any, audio_path: Path) -> list[TextSegment]:
    duration = probe_duration(audio_path)
    raw_text = str(getattr(result, "text", "") or "").strip()
    raw_timestamps = getattr(result, "time_stamps", None)
    tokens = _convert_qwen_timestamps(raw_timestamps)

    if tokens:
        text = raw_text or "".join(token.text for token in tokens)
        return [
            TextSegment(
                start=tokens[0].start,
                end=tokens[-1].end,
                text=text,
                tokens=tokens,
            )
        ]

    if raw_text:
        return [TextSegment(start=0.0, end=duration, text=raw_text)]

    return [TextSegment(start=0.0, end=duration, text="[Empty transcription]")]


def _convert_qwen_timestamps(raw_timestamps: Any) -> tuple[TimedToken, ...]:
    if not raw_timestamps:
        return ()

    tokens: list[TimedToken] = []
    for item in raw_timestamps:
        text = str(getattr(item, "text", "") or "").strip()
        try:
            start = float(getattr(item, "start_time"))
            end = float(getattr(item, "end_time"))
        except (TypeError, ValueError, AttributeError):
            continue
        if end <= start or not text:
            continue
        tokens.append(TimedToken(text=text, start=start, end=end))
    return tuple(tokens)


def _expand_qwen_languages(
    audio_paths: Sequence[Path],
    language: str | Sequence[str | None],
) -> list[str | None]:
    if isinstance(language, str) or language is None:
        return [_map_qwen_language(language)] * len(audio_paths)

    languages = list(language)
    if len(languages) != len(audio_paths):
        raise TranscriptionError("Qwen3-ASR batch language count does not match audio clips.")
    return [_map_qwen_language(item) for item in languages]


def _map_qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in {"auto", "none"}:
        return None
    if lowered in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[lowered]
    return None


def _normalize_device(device: str) -> str:
    if device.startswith("cuda") and ":" not in device:
        return "cuda:0"
    return device


def _require_forced_aligner_path(load_options: dict[str, Any], local_path: Path) -> Path:
    raw_path = load_options.get("forced_aligner_path")
    if not raw_path:
        raise BackendLoadError(
            f"Failed to load Qwen3-ASR model from {local_path}: missing required option 'forced_aligner_path'."
        )
    forced_aligner_path = Path(raw_path)
    if not forced_aligner_path.exists():
        raise BackendLoadError(
            f"Failed to load Qwen3-ASR model from {local_path}: forced aligner path does not exist: {forced_aligner_path}"
        )
    return forced_aligner_path


def _resolve_torch_dtype(torch: Any, raw_dtype: str):
    normalized = str(raw_dtype).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise BackendLoadError(f"Unsupported forced_aligner_dtype: {raw_dtype}")


def _missing_qwen_dependency_error(local_path: Path, exc: ModuleNotFoundError) -> BackendLoadError:
    missing_name = exc.name or ""
    if missing_name == "qwen_asr" or "qwen_asr" in str(exc):
        return BackendLoadError(
            "Failed to load Qwen3-ASR model from "
            f"{local_path}: missing dependency 'qwen-asr'. "
            f"Current runtime: {format_runtime_context()}. "
            "请使用 `bash tools/runtime_env.sh run qwen` 启动服务，或先执行 "
            "`bash tools/runtime_env.sh setup qwen` 创建独立环境。"
        )
    if missing_name == "vllm" or "vllm" in str(exc):
        return BackendLoadError(
            "Failed to load Qwen3-ASR model from "
            f"{local_path}: missing dependency 'vllm'. "
            f"Current runtime: {format_runtime_context()}. "
            "请使用 `bash tools/runtime_env.sh run qwen` 启动服务，或先执行 "
            "`bash tools/runtime_env.sh setup qwen` 创建独立环境。"
        )
    return BackendLoadError(f"Failed to load Qwen3-ASR model from {local_path}: {exc}")


def _normalize_qwen_load_error(local_path: Path, exc: Exception, load_options: dict[str, Any]) -> BackendLoadError:
    message = str(exc)
    lowered = message.lower()
    if "available kv cache memory" in lowered or "estimated maximum model length" in lowered:
        configured_max_model_len = int(load_options.get("max_model_len", 16384))
        configured_gpu_memory_utilization = float(load_options.get("gpu_memory_utilization", 0.7))
        estimated = _extract_estimated_max_model_len(message)
        hint = (
            f"Failed to load Qwen3-ASR model from {local_path}: vLLM KV cache memory is insufficient for the current "
            f"startup settings. 当前配置 `max_model_len={configured_max_model_len}`、"
            f"`gpu_memory_utilization={configured_gpu_memory_utilization:.2f}`。"
        )
        if estimated is not None:
            hint += f" 该 GPU 当前估算可接受的 `max_model_len` 大约是 {estimated}。"
        hint += " 请优先减小 `max_model_len`，必要时再调低批量或调整 `gpu_memory_utilization`。"
        hint += f" 原始错误: {message}"
        return BackendLoadError(hint)
    return BackendLoadError(f"Failed to load Qwen3-ASR model from {local_path}: {exc}")


def _extract_estimated_max_model_len(message: str) -> int | None:
    matched = re.search(r"estimated maximum model length is (\d+)", message)
    if not matched:
        return None
    try:
        return int(matched.group(1))
    except ValueError:
        return None
