# MSR Roadmap

## Phase 0: Design Freeze

Goal:

- Freeze naming, module boundaries, config format, API compatibility goals, and README structure.

Outputs:

- `docs/architecture.md`
- `config/models.toml` template
- API surface for public and admin routes

Acceptance:

- No runtime-critical behavior depends on undocumented assumptions.

## Phase 1: Minimum Service Skeleton

Goal:

- Build the service shell and management plane without auto-loading any models.

Outputs:

- FastAPI entrypoint
- single API key security
- model registry and runtime state manager
- static management page

Acceptance:

- Service starts cleanly
- `/health` responds without auth
- protected APIs require `X-API-Key`
- no models are loaded at startup

## Phase 2: Default Path

Goal:

- Make `faster-whisper + 3D-Speaker + WebRTC VAD` work as the default offline pipeline.

Outputs:

- explicit model load/unload
- `/transcribe/` compatibility response
- resource monitoring
- bounded parallel execution and bounded queueing for synchronous requests
- runtime task observation for active, queued and recent jobs

Acceptance:

- service can run with `--network none`
- startup does not trigger downloads
- loaded models can process a sample file end-to-end
- overload returns a readable `queue_full` error instead of unbounded pile-up

## Phase 3: Alternate Backends

Goal:

- Enable `faster-whisper`, `pyannote`, and experimental `Qwen3-ASR` as switchable alternatives.

Outputs:

- alternate ASR adapter
- alternate diarization adapter
- config-driven backend switching
- local in-process `vLLM` profile for `Qwen3-ASR`
- `ForcedAligner` integration as an internal dependency of the Qwen ASR backend
- preliminary design for cross-audio speaker identity registry

Acceptance:

- switching does not require a service restart
- failure messages stay readable
- default path remains unaffected
- Qwen path can load strictly from local model directories and reuse the existing `/transcribe/` and async result contracts
- roadmap covers how speaker identity can persist beyond a single audio file

## Phase 4: Hardening

Goal:

- Add operational checks and deployment polish.

Outputs:

- `tools/doctor.py`
- stable Docker GPU runtime
- smoke test instructions
- fully updated README
- admin console that reflects runtime limits, queue state and recent task summaries
- benchmark notes for `Qwen3-ASR 0.6B` on `RTX 3060 12GB`, with `1.7B` tracked as experimental

Acceptance:

- offline deployment can be repeated from docs
- common errors are diagnosable
- GPU image passes load, transcribe, unload smoke flow
- runtime controls can be audited from the management plane
- Qwen experimental chain has a documented go/no-go conclusion instead of remaining implicit
