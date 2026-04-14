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

- Make `FunASR + 3D-Speaker + WebRTC VAD` work as the default offline pipeline.

Outputs:

- explicit model load/unload
- `/transcribe/` compatibility response
- resource monitoring

Acceptance:

- service can run with `--network none`
- startup does not trigger downloads
- loaded models can process a sample file end-to-end

## Phase 3: Alternate Backends

Goal:

- Enable `faster-whisper` and `pyannote` as switchable alternatives.

Outputs:

- alternate ASR adapter
- alternate diarization adapter
- config-driven backend switching

Acceptance:

- switching does not require a service restart
- failure messages stay readable
- default path remains unaffected

## Phase 4: Hardening

Goal:

- Add operational checks and deployment polish.

Outputs:

- `tools/doctor.py`
- stable Docker GPU runtime
- smoke test instructions
- fully updated README

Acceptance:

- offline deployment can be repeated from docs
- common errors are diagnosable
- GPU image passes load, transcribe, unload smoke flow
