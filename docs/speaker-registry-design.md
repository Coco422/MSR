# Speaker Registry Design

## Summary

Speaker registry is the next admin-only capability after the current transcription pipeline. Its job is to persist speaker voice features across audio files so the service can recognize that diarization speaker `A` in audio 1 and diarization speaker `C` in audio 2 are the same real person, for example `Alice`.

This document focuses on a file-based v1 design that works without an external database.

## Goals

- Reuse known speaker identities across audio files
- Keep the current `/transcribe/` and async task contracts backward compatible
- Default to local file persistence under `data/`
- Support unknown identities first, then allow admin rename and merge
- Keep the matching path offline-first and explainable

## Non-goals

- No external database in v1
- No realtime streaming identity resolution in v1
- No end-user self-service identity management
- No automatic global re-clustering of all historical audio in v1

## Default assumptions

- Default service chain is `faster-whisper + 3D-Speaker + WebRTC VAD`
- Registry matching is an internal post-diarization step, not a replacement for diarization
- Unknown unmatched speakers are auto-created as local identities
- Admins can later rename, merge, disable or archive identities

## Core entities

### Registry

- `registry_id`
- `version`
- `created_at`
- `updated_at`
- `embedding_backend`
- `distance_metric`
- `match_threshold`

### Identity

- `identity_id`
- `display_name`
- `status`
- `primary_embedding_id`
- `embedding_ids`
- `sample_ids`
- `notes`
- `created_at`
- `updated_at`

Suggested `status` values:

- `unknown`
- `named`
- `merged`
- `archived`

### Embedding

- `embedding_id`
- `identity_id`
- `backend`
- `dimension`
- `vector_path`
- `source_task_id`
- `source_speaker_id`
- `duration_seconds`
- `quality_score`
- `created_at`

### Sample

- `sample_id`
- `identity_id`
- `audio_path`
- `start`
- `end`
- `duration_seconds`
- `source_task_id`
- `source_filename`
- `created_at`

## Storage layout

Recommended v1 layout:

```text
data/
  speaker_registry/
    registry.json
    identities/
      identity-0001.json
      identity-0002.json
    embeddings/
      emb-0001.npy
      emb-0002.npy
    samples/
      identity-0001/
        sample-0001.wav
      identity-0002/
        sample-0003.wav
    events.jsonl
```

Storage rules:

- metadata uses JSON or JSONL for easy inspection
- vectors use `.npy`
- reference audio clips are optional but recommended for admin review
- writes should be atomic with temp file + rename

## Matching flow

### 1. Collect speaker-level speech

After diarization and transcript post-processing:

- group effective speech regions by final speaker
- discard regions that are too short or too noisy
- merge nearby speech spans into a registry candidate clip

### 2. Build registry embedding

For each effective speaker candidate:

- extract one or more reference clips
- compute embedding with a dedicated embedding backend
- score clip quality
- pick the best embedding as primary

### 3. Match against registry

- compare candidate embedding against active identities
- if top match exceeds threshold and margin is stable, resolve to known identity
- otherwise create a new `unknown-*` identity

### 4. Persist decision

- write sample metadata
- write embedding metadata
- append an event to `events.jsonl`
- if unmatched, create unknown identity immediately

## Integration with current pipeline

Registry resolution should sit after current speaker/text alignment:

1. upload
2. normalize
3. VAD
4. diarization
5. ASR
6. speaker token-level reassignment
7. registry candidate extraction
8. embedding + match
9. replace `speaker_label` with known identity when matched
10. persist registry side effects

Backward compatibility rule:

- `speaker_id` remains the session-local diarization speaker id
- registry should add optional identity fields later instead of replacing current core fields immediately

## Proposed internal modules

- `src/msr/services/speaker_registry.py`
  responsibility: identity CRUD, match, persist, merge
- `src/msr/services/speaker_embedding.py`
  responsibility: embedding extraction from waveform or clip path
- `src/msr/core/speaker_registry_types.py`
  responsibility: typed models for identity, embedding, sample, event

## Admin APIs for later implementation

Suggested admin routes:

- `GET /api/v1/speaker-registry/identities`
- `GET /api/v1/speaker-registry/identities/{identity_id}`
- `POST /api/v1/speaker-registry/identities/{identity_id}/rename`
- `POST /api/v1/speaker-registry/identities/{identity_id}/merge`
- `POST /api/v1/speaker-registry/identities/{identity_id}/archive`
- `GET /api/v1/speaker-registry/events`

Suggested task-side behavior:

- keep `/transcribe/` response compatible
- add optional registry info only when feature flag is enabled

## Rollout phases

### Phase 1

- file-based store
- unknown auto-create
- manual rename
- manual merge
- basic candidate extraction

### Phase 2

- configurable thresholds
- quality filters
- identity audit trail
- admin review UI

### Phase 3

- periodic re-embed or rebuild
- batch backfill for historical tasks
- optional cross-project registry separation

## Open questions

- embedding backend should be `speakerlab`-based or a dedicated speaker encoder
- whether one registry is global or scoped by tenant/project
- whether unmatched speakers should always persist or only persist above a duration threshold
- when to expose registry fields in public API without breaking upstream assumptions
