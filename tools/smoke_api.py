from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSR API smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--diar-model", required=True)
    parser.add_argument("--audio", required=True, help="Absolute path to a sample audio file")
    return parser.parse_args()


def call(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    print(f"{method} {path} -> {response.status_code}")
    try:
        payload = response.json()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)
        payload = None
    response.raise_for_status()
    return payload


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120.0) as client:
        call(client, "GET", "/health")
        call(client, "GET", "/api/v1/auth/check")
        call(client, "GET", "/api/v1/models")
        call(client, "POST", f"/api/v1/models/asr/{args.asr_model}/load")
        call(client, "POST", f"/api/v1/models/diarization/{args.diar_model}/load")
        call(client, "GET", "/api/v1/runtime/active")

        with audio_path.open("rb") as handle:
            files = {
                "audio": (audio_path.name, handle, "audio/wav"),
            }
            call(client, "POST", "/transcribe/", files=files)

        call(client, "POST", f"/api/v1/models/diarization/{args.diar_model}/unload")
        call(client, "POST", f"/api/v1/models/asr/{args.asr_model}/unload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
