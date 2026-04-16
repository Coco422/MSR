from __future__ import annotations

from io import BytesIO
import builtins
import json
from pathlib import Path
import threading
import time
import types
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from msr.app.main import create_app
from msr.backends.asr.base import ASRBackend
from msr.backends.asr.faster_whisper_backend import FasterWhisperBackend
from msr.backends.asr.funasr_backend import FunASRBackend, _parse_funasr_result
from msr.backends.asr.qwen_asr_backend import QwenASRBackend
from msr.backends.diarization.three_d_speaker_backend import ThreeDSpeakerBackend
from msr.backends.diarization.pyannote_backend import (
    PyannoteBackend,
    _extract_pyannote_annotation,
    _resolve_pyannote_checkpoint,
)
from msr.core.errors import BackendLoadError
from msr.core.config import AppConfig, ModelConfig, RuntimeConfig, SecurityConfig, Settings, WebConfig, load_settings
from msr.core.types import SpeakerSegment, TextSegment, TimedToken
from msr.services.audio_io import probe_duration
from msr.services.model_manager import ASR_FACTORIES, DIARIZATION_FACTORIES
from msr.services.transcription_service import (
    _build_speaker_presentations,
    _build_speakers_info,
    _build_transcripts,
    _match_text_to_speakers,
    _offset_text_segment,
    _split_ranges_for_asr,
)


class FakeASRBackend:
    def __init__(self, model_id: str, options: dict | None = None):
        self.model_id = model_id
        self.options = options or {}
        self.loaded = False

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
        duration = max(probe_duration(audio_path), 0.1)
        delay = float(self.options.get("sleep_per_second", 0.0)) * duration
        if delay > 0:
            time.sleep(delay)
        return [TextSegment(start=0.0, end=duration, text="测试文本")]

    def transcribe_many(
        self,
        audio_paths: list[Path],
        language: str | list[str | None],
        timestamps: bool = True,
    ) -> list[list[TextSegment]]:
        if isinstance(language, str) or language is None:
            languages = [str(language or "auto")] * len(audio_paths)
        else:
            languages = [str(item or "auto") for item in language]
        return [
            self.transcribe(audio_path, language=item, timestamps=timestamps)
            for audio_path, item in zip(audio_paths, languages)
        ]


class FakeDiarizationBackend:
    def __init__(self, model_id: str, options: dict | None = None):
        self.model_id = model_id
        self.options = options or {}
        self.loaded = False

    def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def diarize(self, audio_path: Path, speaker_hints: dict | None = None) -> list[SpeakerSegment]:
        duration = max(probe_duration(audio_path), 1.0)
        return [SpeakerSegment(speaker_id="0", start=0.0, end=duration)]


def build_settings(
    tmp_path: Path,
    *,
    max_parallel_tasks: int = 1,
    max_queued_tasks: int = 2,
    recent_task_limit: int = 20,
    asr_options: dict | None = None,
    extra_models: list[ModelConfig] | None = None,
) -> Settings:
    asr_dir = tmp_path / "asr"
    diar_dir = tmp_path / "diar"
    asr_dir.mkdir()
    diar_dir.mkdir()

    models = [
        ModelConfig(
            id="funasr-paraformer-zh",
            kind="asr",
            backend="funasr",
            local_path=asr_dir,
            options=dict(asr_options or {}),
        ),
        ModelConfig(
            id="3dspeaker-default",
            kind="diarization",
            backend="3d_speaker",
            local_path=diar_dir,
        ),
    ]
    models.extend(extra_models or [])

    return Settings(
        project_root=Path(__file__).resolve().parents[1],
        app=AppConfig(
            name="MSR",
            service_name="Multi Speaker Recognization",
            version="0.1.0",
            host="127.0.0.1",
            port=8011,
            log_level="INFO",
            default_language="zh",
            temp_dir=tmp_path / "tmp",
            strict_offline=True,
        ),
        security=SecurityConfig(api_key="test-key", api_key_header="X-API-Key"),
        web=WebConfig(title="MSR Console", resource_refresh_seconds=3),
        runtime=RuntimeConfig(
            max_parallel_tasks=max_parallel_tasks,
            max_queued_tasks=max_queued_tasks,
            recent_task_limit=recent_task_limit,
            data_dir=tmp_path / "data",
        ),
        models=models,
    )


def build_client(tmp_path: Path, **kwargs) -> TestClient:
    ASR_FACTORIES["funasr"] = FakeASRBackend
    DIARIZATION_FACTORIES["3d_speaker"] = FakeDiarizationBackend
    app = create_app(settings=build_settings(tmp_path, **kwargs))
    return TestClient(app)


def make_audio_bytes(seconds: float = 1.0, speech: bool = True) -> bytes:
    sample_rate = 16000
    timeline = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    if speech:
        audio = 0.1 * np.sin(2 * np.pi * 220 * timeline)
    else:
        audio = np.zeros_like(timeline)
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def load_models(client: TestClient, headers: dict[str, str]) -> None:
    load_asr = client.post("/api/v1/models/asr/funasr-paraformer-zh/load", headers=headers)
    load_diar = client.post("/api/v1/models/diarization/3dspeaker-default/load", headers=headers)
    assert load_asr.status_code == 200
    assert load_diar.status_code == 200


def wait_for_counts(
    client: TestClient,
    headers: dict[str, str],
    *,
    active: int | None = None,
    queued: int | None = None,
    timeout: float = 3.0,
) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/api/v1/runtime/tasks", headers=headers)
        payload = response.json()
        counts = payload["counts"]
        if (active is None or counts["active"] == active) and (queued is None or counts["queued"] == queued):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for counts active={active} queued={queued}")


def transcribe_request(
    client: TestClient,
    headers: dict[str, str],
    *,
    seconds: float,
    filename: str = "sample.wav",
    speech: bool = True,
) -> tuple[int, dict]:
    response = client.post(
        "/transcribe/",
        headers=headers,
        files={"audio": (filename, make_audio_bytes(seconds=seconds, speech=speech), "audio/wav")},
    )
    return response.status_code, response.json()


def test_health_is_public(tmp_path: Path):
    with build_client(tmp_path) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_console_pages_are_public(tmp_path: Path):
    with build_client(tmp_path) as client:
        for path in ["/", "/models", "/runtime", "/transcribe"]:
            response = client.get(path)
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


def test_resolve_pyannote_checkpoint_prefers_config_yaml(tmp_path: Path):
    model_dir = tmp_path / "pyannote-model"
    model_dir.mkdir()
    config_path = model_dir / "config.yaml"

    assert _resolve_pyannote_checkpoint(model_dir) == config_path.resolve()
    assert _resolve_pyannote_checkpoint(config_path) == config_path.resolve()


def test_extract_pyannote_annotation_supports_diarize_output():
    annotation = object()

    class DiarizeOutput:
        speaker_diarization = annotation

    assert _extract_pyannote_annotation(annotation) is annotation
    assert _extract_pyannote_annotation(DiarizeOutput()) is annotation


def test_transcribe_requires_api_key(tmp_path: Path):
    with build_client(tmp_path) as client:
        response = client.post("/transcribe/")
        assert response.status_code == 401


def test_model_load_and_transcribe_flow(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(tmp_path) as client:
        load_models(client, headers)

        response = client.post(
            "/transcribe/",
            headers=headers,
            files={"audio": ("sample.wav", make_audio_bytes(), "audio/wav")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert "task_id" in payload
        assert isinstance(payload["transcripts"], list)
        assert payload["transcripts"][0]["speaker_id"] == "0"
        assert payload["processing_speed"].endswith("x")

        tasks = client.get("/api/v1/runtime/tasks", headers=headers)
        recent = tasks.json()["recent"]
        assert recent
        assert recent[0]["status"] == "completed"
        assert recent[0]["filename"] == "sample.wav"


def test_models_endpoint_lists_and_loads_qwen_models(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    qwen_model_dir = tmp_path / "qwen-asr"
    qwen_aligner_dir = tmp_path / "qwen-aligner"
    qwen_model_dir.mkdir()
    qwen_aligner_dir.mkdir()

    ASR_FACTORIES["funasr"] = FakeASRBackend
    ASR_FACTORIES["qwen_asr"] = FakeASRBackend
    DIARIZATION_FACTORIES["3d_speaker"] = FakeDiarizationBackend

    settings = build_settings(
        tmp_path,
        extra_models=[
            ModelConfig(
                id="qwen3-asr-0.6b",
                kind="asr",
                backend="qwen_asr",
                local_path=qwen_model_dir,
                options={
                    "engine": "vllm",
                    "forced_aligner_path": qwen_aligner_dir,
                    "forced_aligner_dtype": "float16",
                },
            ),
            ModelConfig(
                id="qwen3-asr-1.7b",
                kind="asr",
                backend="qwen_asr",
                local_path=qwen_model_dir,
                options={
                    "engine": "vllm",
                    "forced_aligner_path": qwen_aligner_dir,
                    "forced_aligner_dtype": "float16",
                },
            ),
        ],
    )

    with TestClient(create_app(settings=settings)) as client:
        models = client.get("/api/v1/models", headers=headers)
        assert models.status_code == 200
        payload = models.json()
        qwen_ids = [item["id"] for item in payload if item["backend"] == "qwen_asr"]
        assert qwen_ids == ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]

        load = client.post("/api/v1/models/asr/qwen3-asr-0.6b/load", headers=headers)
        assert load.status_code == 200
        assert load.json()["id"] == "qwen3-asr-0.6b"


def test_transcribe_without_loaded_models_returns_409_and_does_not_queue(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(tmp_path) as client:
        response = client.post(
            "/transcribe/",
            headers=headers,
            files={"audio": ("sample.wav", make_audio_bytes(), "audio/wav")},
        )
        assert response.status_code == 409

        tasks = client.get("/api/v1/runtime/tasks", headers=headers)
        assert tasks.json()["counts"] == {"active": 0, "queued": 0, "recent": 0}


def test_transcribe_silence_returns_400(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(tmp_path) as client:
        load_models(client, headers)

        response = client.post(
            "/transcribe/",
            headers=headers,
            files={"audio": ("silence.wav", make_audio_bytes(seconds=1.0, speech=False), "audio/wav")},
        )
        assert response.status_code == 400
        assert "No speech activity detected" in response.json()["detail"]


def test_async_transcription_endpoints_return_status_and_result(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(
        tmp_path,
        max_parallel_tasks=1,
        max_queued_tasks=1,
        asr_options={"sleep_per_second": 0.2},
    ) as client:
        load_models(client, headers)

        submit = client.post(
            "/api/v1/transcriptions/submit",
            headers=headers,
            files={"audio": ("async.wav", make_audio_bytes(seconds=1.0), "audio/wav")},
        )
        assert submit.status_code == 200
        task_id = submit.json()["task_id"]

        status = client.get(f"/api/v1/transcriptions/{task_id}", headers=headers)
        assert status.status_code == 200
        assert status.json()["task_id"] == task_id

        deadline = time.time() + 5.0
        result_response = None
        while time.time() < deadline:
            result_response = client.get(f"/api/v1/transcriptions/{task_id}/result", headers=headers)
            if result_response.status_code == 200:
                break
            assert result_response.status_code == 202
            time.sleep(0.05)

        assert result_response is not None
        assert result_response.status_code == 200
        assert result_response.json()["status"] == "completed"
        assert result_response.json()["task_id"] == task_id


def test_task_results_are_trimmed_with_recent_limit(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(
        tmp_path,
        max_parallel_tasks=1,
        max_queued_tasks=0,
        recent_task_limit=2,
    ) as client:
        load_models(client, headers)

        for index in range(3):
            response = client.post(
                "/api/v1/transcriptions/submit",
                headers=headers,
                files={"audio": (f"task-{index}.wav", make_audio_bytes(seconds=0.5), "audio/wav")},
            )
            assert response.status_code == 200
            task_id = response.json()["task_id"]

            deadline = time.time() + 5.0
            result = None
            while time.time() < deadline:
                result = client.get(f"/api/v1/transcriptions/{task_id}/result", headers=headers)
                if result.status_code == 200:
                    break
                time.sleep(0.05)
            assert result is not None
            assert result.status_code == 200

        recent_path = tmp_path / "data" / "recent_tasks.json"
        recent_payload = json.loads(recent_path.read_text(encoding="utf-8"))
        assert len(recent_payload) == 2

        results_dir = tmp_path / "data" / "task_results"
        assert len(list(results_dir.glob("*.json"))) == 2


def test_queue_full_returns_503(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(
        tmp_path,
        max_parallel_tasks=1,
        max_queued_tasks=1,
        asr_options={"sleep_per_second": 0.4},
    ) as client:
        load_models(client, headers)

        results: dict[str, tuple[int, dict]] = {}

        def run_in_thread(name: str, duration: float) -> None:
            results[name] = transcribe_request(client, headers, seconds=duration, filename=f"{name}.wav")

        first = threading.Thread(target=run_in_thread, args=("first", 2.5))
        second = threading.Thread(target=run_in_thread, args=("second", 2.5))
        first.start()
        wait_for_counts(client, headers, active=1, queued=0)

        second.start()
        wait_for_counts(client, headers, active=1, queued=1)

        status_code, payload = transcribe_request(client, headers, seconds=1.0, filename="third.wav")
        assert status_code == 503
        assert payload["detail"]["code"] == "queue_full"
        assert payload["detail"]["active_tasks"] == 1
        assert payload["detail"]["queued_tasks"] == 1

        first.join()
        second.join()
        assert results["first"][0] == 200
        assert results["second"][0] == 200


def test_shorter_task_can_finish_before_longer_task(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(
        tmp_path,
        max_parallel_tasks=2,
        max_queued_tasks=0,
        asr_options={"sleep_per_second": 0.3},
    ) as client:
        load_models(client, headers)

        completion_order: list[str] = []

        def run_and_record(name: str, duration: float) -> None:
            status_code, _ = transcribe_request(client, headers, seconds=duration, filename=f"{name}.wav")
            assert status_code == 200
            completion_order.append(name)

        long_thread = threading.Thread(target=run_and_record, args=("long", 3.0))
        short_thread = threading.Thread(target=run_and_record, args=("short", 1.0))
        long_thread.start()
        short_thread.start()
        long_thread.join()
        short_thread.join()

        assert completion_order[0] == "short"


def test_runtime_limits_can_be_updated_online_for_new_tasks(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(
        tmp_path,
        max_parallel_tasks=1,
        max_queued_tasks=0,
        asr_options={"sleep_per_second": 0.35},
    ) as client:
        load_models(client, headers)

        results: dict[str, tuple[int, dict]] = {}

        def run_in_thread(name: str, duration: float) -> None:
            results[name] = transcribe_request(client, headers, seconds=duration, filename=f"{name}.wav")

        first = threading.Thread(target=run_in_thread, args=("first", 2.0))
        first.start()
        wait_for_counts(client, headers, active=1, queued=0)

        update = client.post(
            "/api/v1/runtime/limits",
            headers=headers,
            json={"max_parallel_tasks": 1, "max_queued_tasks": 1, "recent_task_limit": 5},
        )
        assert update.status_code == 200

        second = threading.Thread(target=run_in_thread, args=("second", 1.0))
        second.start()
        wait_for_counts(client, headers, active=1, queued=1)

        first.join()
        second.join()
        assert results["first"][0] == 200
        assert results["second"][0] == 200

        override_path = tmp_path / "data" / "runtime_overrides.toml"
        assert override_path.exists()
        content = override_path.read_text(encoding="utf-8")
        assert "max_queued_tasks = 1" in content
        assert "recent_task_limit = 5" in content


def test_runtime_limits_endpoint_returns_latest_values(tmp_path: Path):
    headers = {"X-API-Key": "test-key"}
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/runtime/limits",
            headers=headers,
            json={"max_parallel_tasks": 3, "max_queued_tasks": 4, "recent_task_limit": 6},
        )
        assert response.status_code == 200
        assert response.json() == {
            "max_parallel_tasks": 3,
            "max_queued_tasks": 4,
            "recent_task_limit": 6,
        }

        current = client.get("/api/v1/runtime/limits", headers=headers)
        assert current.status_code == 200
        assert current.json()["max_parallel_tasks"] == 3


def test_load_settings_applies_runtime_override_file(tmp_path: Path):
    project_root = tmp_path / "project"
    config_dir = project_root / "config"
    data_dir = project_root / "runtime-state"
    models_dir = project_root / "models"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (models_dir / "asr").mkdir(parents=True)
    (models_dir / "diar").mkdir(parents=True)

    (config_dir / "app.toml").write_text(
        """
[app]
name = "MSR"
service_name = "Multi Speaker Recognization"
version = "0.1.0"
host = "127.0.0.1"
port = 8011
log_level = "INFO"
default_language = "zh"
temp_dir = "tmp"
strict_offline = true

[security]
api_key = "test-key"

[web]
title = "MSR Console"

[runtime]
max_parallel_tasks = 1
max_queued_tasks = 2
recent_task_limit = 10
data_dir = "runtime-state"
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "models.toml").write_text(
        """
[[models]]
id = "funasr-paraformer-zh"
kind = "asr"
backend = "funasr"
local_path = "models/asr"

[[models]]
id = "3dspeaker-default"
kind = "diarization"
backend = "3d_speaker"
local_path = "models/diar"
""".strip(),
        encoding="utf-8",
    )
    (data_dir / "runtime_overrides.toml").write_text(
        """
[runtime]
max_parallel_tasks = 4
max_queued_tasks = 7
recent_task_limit = 9
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(project_root=project_root)
    assert settings.runtime.max_parallel_tasks == 4
    assert settings.runtime.max_queued_tasks == 7
    assert settings.runtime.recent_task_limit == 9
    assert settings.runtime.data_dir == data_dir


def test_funasr_timestamp_parser_splits_segments_and_converts_milliseconds(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    sf.write(str(audio_path), np.zeros(32000, dtype=np.float32), 16000)

    raw = [
        {
            "text": "你 好 啊 我 来 了",
            "timestamp": [
                [100, 240],
                [240, 380],
                [380, 520],
                [1200, 1360],
                [1360, 1520],
                [1520, 1700],
            ],
        }
    ]

    segments = _parse_funasr_result(raw, audio_path)

    assert [(segment.text, round(segment.start, 2), round(segment.end, 2)) for segment in segments] == [
        ("你好啊", 0.1, 0.52),
        ("我来了", 1.2, 1.7),
    ]
    assert [token.text for token in segments[0].tokens] == ["你", "好", "啊"]
    assert [token.text for token in segments[1].tokens] == ["我", "来", "了"]


def test_match_text_to_speakers_splits_token_runs_on_speaker_boundaries():
    speaker_segments = [
        SpeakerSegment(speaker_id="1", start=12.03, end=13.90),
        SpeakerSegment(speaker_id="4", start=13.90, end=17.79),
    ]
    text_segments = [
        TextSegment(
            start=12.13,
            end=14.45,
            text="然后我还想吃汉堡包辣辣辣",
            tokens=(
                TimedToken(text="然", start=12.13, end=12.35),
                TimedToken(text="后", start=12.35, end=12.53),
                TimedToken(text="我", start=12.53, end=12.65),
                TimedToken(text="还", start=12.65, end=12.79),
                TimedToken(text="想", start=12.79, end=13.01),
                TimedToken(text="吃", start=13.01, end=13.21),
                TimedToken(text="汉", start=13.21, end=13.37),
                TimedToken(text="堡", start=13.37, end=13.57),
                TimedToken(text="包", start=13.57, end=13.81),
                TimedToken(text="辣", start=13.89, end=14.03),
                TimedToken(text="辣", start=14.03, end=14.21),
                TimedToken(text="辣", start=14.21, end=14.45),
            ),
        )
    ]

    speaker_texts = _match_text_to_speakers(text_segments, speaker_segments)

    assert [segment.text for segment in speaker_texts["1"]] == ["然后我还想吃汉堡包"]
    assert [segment.text for segment in speaker_texts["4"]] == ["辣辣辣"]


def test_split_ranges_for_asr_limits_chunk_length():
    assert _split_ranges_for_asr([(0.0, 45.0), (50.0, 55.0)], 20.0) == [
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 45.0),
        (50.0, 55.0),
    ]


def test_funasr_load_disables_update_check_by_default(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))

    backend = FunASRBackend("funasr-paraformer-zh")
    backend.load(tmp_path, "cuda")

    assert captured["model"] == str(tmp_path)
    assert captured["device"] == "cuda:0"
    assert captured["disable_update"] is True


def test_asr_backend_transcribe_many_defaults_to_sequential_calls(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)
    calls: list[tuple[str, str, bool]] = []

    class CountingBackend(ASRBackend):
        def load(self, local_path: Path, device: str, options: dict | None = None) -> None:
            self.loaded = True

        def unload(self) -> None:
            self.loaded = False

        def transcribe(self, audio_path: Path, language: str, timestamps: bool = True) -> list[TextSegment]:
            calls.append((audio_path.name, language, timestamps))
            return [TextSegment(start=0.0, end=0.5, text=language)]

    backend = CountingBackend("counting")
    results = backend.transcribe_many([audio_path, audio_path], language="zh", timestamps=True)

    assert len(results) == 2
    assert [item[0].text for item in results] == ["zh", "zh"]
    assert calls == [("sample.wav", "zh", True), ("sample.wav", "zh", True)]


def test_offset_text_segment_preserves_and_offsets_tokens():
    segment = TextSegment(
        start=0.0,
        end=0.5,
        text="你好",
        tokens=(
            TimedToken(text="你", start=0.0, end=0.1),
            TimedToken(text="好", start=0.1, end=0.3),
        ),
    )

    adjusted = _offset_text_segment(segment, types.SimpleNamespace(start=5.0, end=6.0))

    assert adjusted is not None
    assert adjusted.start == 5.0
    assert adjusted.end == 5.3
    assert [(token.text, token.start, token.end) for token in adjusted.tokens] == [
        ("你", 5.0, 5.1),
        ("好", 5.1, 5.3),
    ]


def test_faster_whisper_load_reports_runtime_hint_when_dependency_is_missing(monkeypatch, tmp_path: Path):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = FasterWhisperBackend("faster-whisper-large-v3")
    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path, "cuda")

    message = str(exc_info.value)
    assert "missing dependency 'faster_whisper'" in message
    assert "tools/runtime_env.sh run pyannote" in message
    assert "python=" in message


def test_pyannote_load_reports_runtime_hint_when_dependency_is_missing(monkeypatch, tmp_path: Path):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyannote.audio":
            raise ModuleNotFoundError("No module named 'pyannote.audio'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        "msr.backends.diarization.pyannote_backend._ensure_torchaudio_pyannote_compat",
        lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda value: value))
    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = PyannoteBackend("pyannote-community-1")
    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path, "cuda")

    message = str(exc_info.value)
    assert "missing dependency 'pyannote.audio'" in message
    assert "tools/runtime_env.sh run pyannote" in message
    assert "python=" in message


def test_qwen_load_reports_runtime_hint_when_qwen_dependency_is_missing(monkeypatch, tmp_path: Path):
    def fake_import_module(name: str, package: str | None = None):
        if name == "qwen_asr":
            raise ModuleNotFoundError("No module named 'qwen_asr'")
        return ModuleType(name)

    monkeypatch.setattr("msr.backends.asr.qwen_asr_backend.importlib.import_module", fake_import_module)

    aligner_dir = tmp_path / "aligner"
    aligner_dir.mkdir()
    backend = QwenASRBackend(
        "qwen3-asr-0.6b",
        options={"engine": "vllm", "forced_aligner_path": aligner_dir, "forced_aligner_dtype": "float16"},
    )
    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path, "cuda")

    message = str(exc_info.value)
    assert "missing dependency 'qwen-asr'" in message
    assert "tools/runtime_env.sh run qwen" in message
    assert "python=" in message


def test_qwen_load_reports_runtime_hint_when_vllm_dependency_is_missing(monkeypatch, tmp_path: Path):
    def fake_import_module(name: str, package: str | None = None):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'")
        return ModuleType(name)

    monkeypatch.setattr("msr.backends.asr.qwen_asr_backend.importlib.import_module", fake_import_module)
    monkeypatch.setitem(sys.modules, "qwen_asr", ModuleType("qwen_asr"))

    aligner_dir = tmp_path / "aligner"
    aligner_dir.mkdir()
    backend = QwenASRBackend(
        "qwen3-asr-0.6b",
        options={"engine": "vllm", "forced_aligner_path": aligner_dir, "forced_aligner_dtype": "float16"},
    )
    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path, "cuda")

    message = str(exc_info.value)
    assert "missing dependency 'vllm'" in message
    assert "tools/runtime_env.sh run qwen" in message
    assert "python=" in message


def test_3d_speaker_load_reports_runtime_hint_when_speakerlab_is_missing(monkeypatch, tmp_path: Path):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "speakerlab.bin.infer_diarization":
            raise ModuleNotFoundError("No module named 'speakerlab'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = ThreeDSpeakerBackend("3dspeaker-default")
    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path, "cuda")

    message = str(exc_info.value)
    assert "missing dependency 'speakerlab'" in message
    assert "tools/runtime_env.sh run default" in message
    assert "tools/runtime_env.sh setup qwen" in message
    assert "python=" in message


def test_qwen_load_fails_when_forced_aligner_path_is_missing(tmp_path: Path):
    backend = QwenASRBackend(
        "qwen3-asr-0.6b",
        options={
            "engine": "vllm",
            "forced_aligner_path": tmp_path / "missing-aligner",
            "forced_aligner_dtype": "float16",
        },
    )

    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(tmp_path / "missing-model", "cuda")

    assert "forced aligner path does not exist" in str(exc_info.value)


def test_qwen_backend_maps_timestamps_and_languages(monkeypatch, tmp_path: Path):
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    sf.write(str(audio_a), np.zeros(16000, dtype=np.float32), 16000)
    sf.write(str(audio_b), np.zeros(16000, dtype=np.float32), 16000)

    model_dir = tmp_path / "qwen"
    aligner_dir = tmp_path / "aligner"
    model_dir.mkdir()
    aligner_dir.mkdir()

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )

    class FakeTimestamp:
        def __init__(self, text: str, start_time: float, end_time: float):
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class FakeResult:
        def __init__(self, text: str, language: str, stamps: list[FakeTimestamp]):
            self.text = text
            self.language = language
            self.time_stamps = stamps

    class FakeQwenModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[dict] = []

        def transcribe(self, **kwargs):
            self.calls.append(kwargs)
            audio = kwargs["audio"]
            if isinstance(audio, list):
                return [
                    FakeResult("你好", "Chinese", [FakeTimestamp("你", 0.0, 0.2), FakeTimestamp("好", 0.2, 0.4)]),
                    FakeResult("hello", "English", [FakeTimestamp("hel", 0.0, 0.3), FakeTimestamp("lo", 0.3, 0.6)]),
                ]
            return [FakeResult("你好", "Chinese", [FakeTimestamp("你", 0.0, 0.2), FakeTimestamp("好", 0.2, 0.4)])]

    class FakeQwen3ASRModel:
        instances: list[FakeQwenModel] = []

        @classmethod
        def LLM(cls, **kwargs):
            instance = FakeQwenModel(**kwargs)
            cls.instances.append(instance)
            return instance

    fake_qwen_module = ModuleType("qwen_asr")
    fake_qwen_module.Qwen3ASRModel = FakeQwen3ASRModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_module)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))

    backend = QwenASRBackend(
        "qwen3-asr-0.6b",
        options={
            "engine": "vllm",
            "gpu_memory_utilization": 0.65,
            "max_inference_batch_size": 32,
            "max_new_tokens": 1024,
            "max_model_len": 16384,
            "forced_aligner_path": aligner_dir,
            "forced_aligner_dtype": "float16",
        },
    )
    backend.load(model_dir, "cuda")
    results = backend.transcribe_many([audio_a, audio_b], language=["zh", "en"], timestamps=True)

    assert len(results) == 2
    assert results[0][0].text == "你好"
    assert [token.text for token in results[0][0].tokens] == ["你", "好"]
    assert [(token.start, token.end) for token in results[1][0].tokens] == [(0.0, 0.3), (0.3, 0.6)]

    model_instance = FakeQwen3ASRModel.instances[0]
    assert model_instance.kwargs["model"] == str(model_dir)
    assert model_instance.kwargs["forced_aligner"] == str(aligner_dir)
    assert model_instance.kwargs["forced_aligner_kwargs"]["device_map"] == "cuda:0"
    assert model_instance.kwargs["forced_aligner_kwargs"]["dtype"] == "float16"
    assert model_instance.kwargs["max_model_len"] == 16384
    assert model_instance.calls[0]["language"] == ["Chinese", "English"]


def test_qwen_load_rewrites_kv_cache_error_with_actionable_hint(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "qwen"
    aligner_dir = tmp_path / "aligner"
    model_dir.mkdir()
    aligner_dir.mkdir()

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **kwargs):
            raise RuntimeError(
                "To serve at least one request with the models's max seq len (65536), "
                "(7.0 GiB KV cache is needed, which is larger than the available KV cache memory (2.39 GiB). "
                "Based on the available memory, the estimated maximum model length is 22336."
            )

    fake_qwen_module = ModuleType("qwen_asr")
    fake_qwen_module.Qwen3ASRModel = FakeQwen3ASRModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_module)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))

    backend = QwenASRBackend(
        "qwen3-asr-0.6b",
        options={
            "engine": "vllm",
            "gpu_memory_utilization": 0.65,
            "max_inference_batch_size": 16,
            "max_new_tokens": 1024,
            "max_model_len": 16384,
            "forced_aligner_path": aligner_dir,
            "forced_aligner_dtype": "float16",
        },
    )

    with pytest.raises(BackendLoadError) as exc_info:
        backend.load(model_dir, "cuda")

    message = str(exc_info.value)
    assert "KV cache memory is insufficient" in message
    assert "max_model_len=16384" in message
    assert "22336" in message


def test_load_settings_resolves_model_option_paths(tmp_path: Path):
    project_root = tmp_path / "project"
    config_dir = project_root / "config"
    models_dir = project_root / "models"
    config_dir.mkdir(parents=True)
    (models_dir / "qwen" / "qwen3-asr-0.6b").mkdir(parents=True)
    (models_dir / "qwen" / "qwen3-forced-aligner-0.6b").mkdir(parents=True)

    (config_dir / "app.toml").write_text(
        """
[app]
name = "MSR"
service_name = "Multi Speaker Recognization"
version = "0.1.0"
host = "127.0.0.1"
port = 8011
log_level = "INFO"
default_language = "zh"
temp_dir = "tmp"
strict_offline = true

[security]
api_key = "test-key"
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "models.toml").write_text(
        """
[[models]]
id = "qwen3-asr-0.6b"
kind = "asr"
backend = "qwen_asr"
local_path = "models/qwen/qwen3-asr-0.6b"

[models.options]
engine = "vllm"
forced_aligner_path = "models/qwen/qwen3-forced-aligner-0.6b"
forced_aligner_dtype = "float16"
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(project_root=project_root)

    assert settings.models[0].local_path == project_root / "models" / "qwen" / "qwen3-asr-0.6b"
    assert settings.models[0].options["forced_aligner_path"] == (
        project_root / "models" / "qwen" / "qwen3-forced-aligner-0.6b"
    )


def test_speaker_outputs_filter_invalid_speakers_and_relabel_sequentially():
    speaker_segments = [
        SpeakerSegment(speaker_id="4", start=0.0, end=8.0),
        SpeakerSegment(speaker_id="1", start=8.0, end=14.0),
        SpeakerSegment(speaker_id="0", start=18.0, end=21.0),
        SpeakerSegment(speaker_id="3", start=21.0, end=26.0),
        SpeakerSegment(speaker_id="9", start=30.0, end=31.0),
    ]
    speaker_texts = {
        "4": [TextSegment(start=0.1, end=3.0, text="今天天气不错啊")],
        "1": [TextSegment(start=8.2, end=11.0, text="我明天中午吃麻辣烫")],
        "0": [TextSegment(start=18.2, end=21.0, text="明天中午吃麦当劳")],
        "3": [TextSegment(start=21.1, end=24.0, text="会不会吃了太胖了")],
    }

    presentations = _build_speaker_presentations(speaker_texts)
    transcripts = _build_transcripts(speaker_texts, presentations)
    speakers_info = _build_speakers_info(speaker_segments, speaker_texts, presentations)

    assert [item["speaker_id"] for item in speakers_info] == ["0", "1", "2", "3"]
    assert [item["speaker_label"] for item in speakers_info] == [
        "说话人 A",
        "说话人 B",
        "说话人 C",
        "说话人 D",
    ]
    assert len(transcripts) == 4
    assert [item["speaker_id"] for item in transcripts] == ["0", "1", "2", "3"]
    assert transcripts[0]["speaker_label"] == "说话人 A"
    assert all(item["speaker_id"] != "9" for item in speakers_info)
