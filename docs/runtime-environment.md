# Runtime environment

The repository has completed its migration to the canonical modern runtime.
Legacy runtime code, compatibility-only wrappers, and legacy import paths have been removed.

## Supported architecture

Only the following runtime path is supported:

`src.app.cli` → `src.worker` → `JobRunner` → `PipelineOrchestrator` → `TranscriptionStep` → `AnalysisStep`

Canonical modules to use in new development:

- `src.app.cli`
- `src.worker`
- `src.config.models`
- `src.infrastructure.llm.adapter`
- `src.infrastructure.transcription.whisperx_adapter`
- canonical `src.core.*` and `src.infrastructure.*` runtime/storage modules

## Operational notes

- `models.yaml` is the canonical operator-facing source of LLM backend and generation configuration.
- `ACTIVE_MODEL_PROFILE` only selects the active `models.yaml` profile.
- Legacy LLM env vars such as `TEMPERATURE`, `NUM_CTX`, `NUM_PREDICT`, `TOP_P`, and `REPEAT_PENALTY` are unsupported and ignored by the canonical runtime; move those values into `models.yaml` profile params instead.
- Ollama runtime context configuration is sourced from `models.yaml` via `profiles.<name>.params.num_ctx`; `context_size` is not supported for Ollama profiles.
- `DATABASE_URL` is required for the active worker runtime.
- WhisperX runtime settings are sourced from environment variables in `src.infrastructure.transcription.whisperx_runtime`.
  - `TRANSCRIPTION_MODEL_NAME` defaults to `large-v3`.
  - `TRANSCRIPTION_DEVICE` defaults to `cuda`.
  - `ALIGNMENT_LANGUAGE_CODE` defaults to `ru`.
  - `ALIGNMENT_MODEL_NAME` defaults to `facebook/wav2vec2-base-960h`.
- WhisperX, pyannote, torch, and related ML dependencies remain a compatibility-sensitive stack at the package level, even though legacy runtime code is gone.
- Qdrant is not part of the canonical runtime contract.

## Migration status

Migration is complete:

- legacy runtime and manual workflow paths are removed
- compatibility-only config/adaptor/storage shims are removed
- only canonical imports and entrypoints are supported
- new feature work must target the canonical runtime only
