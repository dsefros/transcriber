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
- `models.yaml` is the canonical operator-facing source of LLM backend and generation configuration.
- `ACTIVE_MODEL_PROFILE` only selects which `models.yaml` profile is active at runtime.
- Legacy LLM env vars such as `TEMPERATURE`, `NUM_CTX`, `NUM_PREDICT`, `TOP_P`, and `REPEAT_PENALTY` are unsupported and ignored by the canonical runtime.
- Ollama profiles in `models.yaml` must use `params.num_ctx` for the runtime context window; `context_size` is not a supported Ollama config key.
- WhisperX runtime dependencies must be installed for audio transcription runs.
- Transcription runtime knobs stay env-driven and default to the existing behavior used by `src.infrastructure.transcription.whisperx_runtime`:
  - `TRANSCRIPTION_MODEL_NAME=large-v3`
  - `TRANSCRIPTION_DEVICE=cuda`
  - `ALIGNMENT_LANGUAGE_CODE=ru`
  - `ALIGNMENT_MODEL_NAME=facebook/wav2vec2-base-960h`

## Validation commands

```bash
pytest
python -m src.app.runtime_doctor --help
python -m src.app.cli --help
python -m src.app.runtime_doctor
```
