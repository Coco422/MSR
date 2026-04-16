#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MSR_DOCKER_IMAGE:-msr-gpu-runtime:latest}"
API_KEY="${MSR_API_KEY:-change-me}"
ASR_MODEL="${MSR_ASR_MODEL:-faster-whisper-large-v3}"
DIAR_MODEL="${MSR_DIAR_MODEL:-3dspeaker-default}"
AUDIO_PATH="${1:-$ROOT_DIR/samples/smoke/smoke_zh_3spk.mp3}"

if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "[offline-check] 音频文件不存在: $AUDIO_PATH" >&2
  exit 1
fi

TEMP_SCRIPT="$(mktemp)"
cleanup() {
  rm -f "$TEMP_SCRIPT"
}
trap cleanup EXIT

cat >"$TEMP_SCRIPT" <<'PY'
import json
import os
import time
import uuid
from pathlib import Path
from urllib import request

base = "http://127.0.0.1:8011"
api_key = os.environ["MSR_API_KEY"]
asr_model = os.environ["MSR_ASR_MODEL"]
diar_model = os.environ["MSR_DIAR_MODEL"]
audio_path = Path(os.environ["MSR_AUDIO_PATH"])


def call(method, path, data=None, content_type=None, timeout=300):
    req = request.Request(base + path, data=data, method=method)
    req.add_header("X-API-Key", api_key)
    if content_type:
        req.add_header("Content-Type", content_type)
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


for _ in range(90):
    try:
        status, _ = call("GET", "/health", timeout=10)
        if status == 200:
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("service did not become ready")

for model_path in (
    f"/api/v1/models/asr/{asr_model}/load",
    f"/api/v1/models/diarization/{diar_model}/load",
):
    status, body = call("POST", model_path, b"{}", "application/json", timeout=300)
    print(model_path, status)
    if status != 200:
        print(body)
        raise SystemExit(1)

boundary = "----MSR" + uuid.uuid4().hex
file_bytes = audio_path.read_bytes()
parts = [
    f"--{boundary}\r\n".encode(),
    f'Content-Disposition: form-data; name="audio"; filename="{audio_path.name}"\r\n'.encode(),
    b"Content-Type: application/octet-stream\r\n\r\n",
    file_bytes,
    b"\r\n",
    f"--{boundary}--\r\n".encode(),
]
payload = b"".join(parts)
status, response = call(
    "POST",
    "/transcribe/",
    payload,
    f"multipart/form-data; boundary={boundary}",
    timeout=600,
)
print("transcribe", status)
parsed = json.loads(response)
print(
    json.dumps(
        {
            "task_id": parsed.get("task_id"),
            "status": parsed.get("status"),
            "segments": len(parsed.get("transcripts", [])),
            "speakers": parsed.get("total_speakers"),
            "processing_time": parsed.get("processing_time"),
        },
        ensure_ascii=False,
    )
)
PY

AUDIO_DIR="$(cd "$(dirname "$AUDIO_PATH")" && pwd)"
AUDIO_BASENAME="$(basename "$AUDIO_PATH")"

docker run --rm --gpus all --network none \
  -e MSR_API_KEY="$API_KEY" \
  -e MSR_ASR_MODEL="$ASR_MODEL" \
  -e MSR_DIAR_MODEL="$DIAR_MODEL" \
  -e MSR_AUDIO_PATH="/mnt/audio/$AUDIO_BASENAME" \
  -v "$ROOT_DIR/config:/app/config:ro" \
  -v "$ROOT_DIR/models:/app/models:ro" \
  -v "$ROOT_DIR/data:/app/data" \
  -v "$AUDIO_DIR:/mnt/audio:ro" \
  -v "$TEMP_SCRIPT:/tmp/msr_offline_check.py:ro" \
  "$IMAGE" \
  bash -lc 'python -m msr.app.main >/tmp/msr-offline.log 2>&1 & pid=$!; python /tmp/msr_offline_check.py; code=$?; kill $pid; wait $pid 2>/dev/null; tail -n 80 /tmp/msr-offline.log; exit $code'
