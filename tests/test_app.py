from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from msr.app.main import create_app
from msr.core.config import AppConfig, ModelConfig, SecurityConfig, Settings, WebConfig
from msr.core.types import SpeakerSegment, TextSegment
from msr.services.model_manager import ASR_FACTORIES, DIARIZATION_FACTORIES


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
        return [TextSegment(start=0.0, end=1.0, text="测试文本")]


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
        return [SpeakerSegment(speaker_id="0", start=0.0, end=1.0)]


def build_settings(tmp_path: Path) -> Settings:
    asr_dir = tmp_path / "asr"
    diar_dir = tmp_path / "diar"
    asr_dir.mkdir()
    diar_dir.mkdir()

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
        models=[
            ModelConfig(
                id="funasr-paraformer-zh",
                kind="asr",
                backend="funasr",
                local_path=asr_dir,
            ),
            ModelConfig(
                id="3dspeaker-default",
                kind="diarization",
                backend="3d_speaker",
                local_path=diar_dir,
            ),
        ],
    )


def build_client(tmp_path: Path) -> TestClient:
    ASR_FACTORIES["funasr"] = FakeASRBackend
    DIARIZATION_FACTORIES["3d_speaker"] = FakeDiarizationBackend
    app = create_app(settings=build_settings(tmp_path))
    return TestClient(app)


def make_audio_bytes() -> bytes:
    sample_rate = 16000
    seconds = 1.0
    timeline = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220 * timeline)
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def test_health_is_public(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_transcribe_requires_api_key(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.post("/transcribe/")
    assert response.status_code == 401


def test_model_load_and_transcribe_flow(tmp_path: Path):
    client = build_client(tmp_path)
    headers = {"X-API-Key": "test-key"}

    load_asr = client.post("/api/v1/models/asr/funasr-paraformer-zh/load", headers=headers)
    load_diar = client.post("/api/v1/models/diarization/3dspeaker-default/load", headers=headers)
    assert load_asr.status_code == 200
    assert load_diar.status_code == 200

    audio_bytes = make_audio_bytes()
    response = client.post(
        "/transcribe/",
        headers=headers,
        files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert "task_id" in payload
    assert isinstance(payload["transcripts"], list)
    assert payload["transcripts"][0]["speaker_id"] == "0"
