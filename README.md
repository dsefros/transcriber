# Transcriber

This repository now supports **one runtime architecture only**: the canonical modern runtime.
The migration is complete, and all legacy runtime paths, compatibility shims, and migration-era import aliases have been removed.

## Supported runtime path

Canonical execution flow:

`src.app.cli` → `src.worker` → `JobRunner` → `PipelineOrchestrator` → `TranscriptionStep` → `AnalysisStep`

Supported foundations:

- CLI entrypoint: `src.app.cli`
- Worker composition root: `src.worker`
- Canonical model config loading: `src.config.models`
- Canonical transcription adapter: `src.infrastructure.transcription.whisperx_adapter`
- Canonical LLM adapter: `src.infrastructure.llm.adapter`
- Canonical storage/runtime infrastructure under `src.core` and `src.infrastructure`

## Breaking change: legacy runtime removed

The following are no longer supported:

- `src.legacy.*` imports of any kind
- legacy/manual pipeline workflow entrypoints
- compatibility-only storage/config re-export modules
- compatibility-only adapter wrappers
- migration-era documentation that treated legacy paths as valid runtime options

New development must target the canonical runtime only.
Do not add new compatibility wrappers or reintroduce legacy import paths.

## Running the canonical CLI

```bash
python -m src.app.cli /path/to/audio.wav
python -m src.app.cli /path/to/job.json --json
```

## Runtime expectations

- `DATABASE_URL` must be set for the worker runtime.
- `models.yaml` must define the active LLM profile used by `src.config.models`.
- WhisperX runtime dependencies must be installed for audio transcription runs.

## Validation commands

```bash
pytest
python -m src.app.runtime_doctor --help
python -m src.app.cli --help
python -m src.app.runtime_doctor
```
