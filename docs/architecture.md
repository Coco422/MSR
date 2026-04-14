# MSR Architecture

## Summary

MSR is a single-process FastAPI service designed for offline multi-speaker transcription. The project favors explicit runtime state and minimal moving parts over automation-heavy orchestration.

## Runtime model

MSR keeps exactly two runtime model slots:

- one active ASR backend
- one active diarization backend

Models are registered statically in `config/models.toml`, but they are not loaded during process startup. All loading and unloading happens through management APIs.

## Main request flow

1. Client uploads audio to `POST /transcribe/`
2. Service validates API key
3. Service verifies that ASR and diarization backends are loaded
4. Audio is normalized to mono 16 kHz
5. WebRTC VAD produces coarse speech ranges
6. Diarization backend analyzes the full audio
7. Each VAD clip is sent through the active ASR backend
8. ASR segments are matched to the diarization timeline by overlap
9. Response is shaped to preserve the old demo contract

## Module boundaries

- `core/`: pure app wiring, config, security, errors
- `services/`: orchestration and runtime state
- `backends/`: third-party model adapters only
- `api/`: HTTP contracts only
- `web/`: static management console only

## Offline contract

MSR enforces offline execution by policy:

- model config must point to local directories
- no runtime model download logic is allowed
- no token-based fetch is allowed in service code
- startup sets strict offline environment variables

## Extension policy

v1 only treats ASR as a formal pluggable boundary. Diarization remains internally switchable, but we intentionally avoid a larger plugin system until the current runtime proves stable.
