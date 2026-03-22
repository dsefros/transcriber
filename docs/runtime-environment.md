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

- `models.yaml` is the single supported source of model runtime configuration.
- `DATABASE_URL` is required for the active worker runtime.
- WhisperX, pyannote, torch, and related ML dependencies remain a compatibility-sensitive stack at the package level, even though legacy runtime code is gone.
- Qdrant is not part of the canonical runtime contract.

## Migration status

Migration is complete:

- legacy runtime and manual workflow paths are removed
- compatibility-only config/adaptor/storage shims are removed
- only canonical imports and entrypoints are supported
- new feature work must target the canonical runtime only
