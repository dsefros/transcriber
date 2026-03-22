# Canonical Runtime Environment Contract

This document defines the supported baseline for the canonical execution path:

`CLI -> Worker -> JobRunner -> PipelineOrchestrator -> TranscriptionStep -> AnalysisStep`

It intentionally documents a **narrow, reproducible baseline** rather than every historical or accidental environment that might partially work.

## 1. Short inventory of the current dependency/runtime story

Grounded repository observations before this PR:

- Dependencies were declared in a single top-level `requirements.txt`, even though the codebase already behaved as if it had multiple layers: lightweight CLI/config imports, runtime DB wiring, audio ML runtime, and test tooling.
- The previous README referenced `requirements-ml.txt`, but that file did not exist, so bootstrap instructions were partially implicit and not reproducible.
- Several active-runtime imports were not declared at all in dependency files even though the canonical path uses them directly or transitively: `SQLAlchemy`, `PyYAML`, `python-dotenv`, `psutil`, and `whisperx`.
- Test tooling was implicit: `pytest` was required for the repository workflow but was not declared in a dedicated development dependency set.
- The active transcription path is GPU-oriented by implementation today: `WhisperXTranscriptionRuntime.transcribe()` defaults to `device="cuda"`, and the pinned torch wheels target CUDA 12.1.
- The code already shows visible compatibility-risk zones in the ML stack because `torch`, `torchaudio`, `pyannote.audio`, and `whisperx` must cooperate, but only part of that stack was previously pinned.
- The canonical runtime also depends on environment state that was only partly encoded in docs:
  - `DATABASE_URL` is required before `Worker` startup.
  - `HF_TOKEN` is practically required for pyannote diarization downloads/auth flows.
  - `models.yaml` must exist and the selected `ACTIVE_MODEL_PROFILE` must resolve successfully.
  - For `llama_cpp` profiles, the configured local GGUF path must exist.
  - For `ollama` profiles, the Python package and local Ollama service must both be available.

## 2. Supported baseline contract

This PR documents the following supported baseline for reproducible local execution of the canonical runtime.

### Python

- **Supported:** Python `3.10` and `3.11`.
- **Recommended baseline:** Python `3.10.x` for the most conservative match with the CUDA/ML stack.
- Anything outside that range is treated as **outside the documented support contract** and should be considered best-effort only.

### Compute / GPU

- **Supported baseline:** NVIDIA GPU runtime with CUDA `12.1` userspace via `torch==2.3.0+cu121`.
- **CPU-only execution:** may sometimes work for debugging, but it is **not the supported canonical baseline** for WhisperX execution in this repository.
- The doctor tool reports CPU-only environments as warnings rather than pretending full support.

### Canonical runtime dependency groups

#### A. Runtime core (`requirements/runtime.txt`)
Required for the canonical CLI/Worker/config/database path:

- `PyYAML`
- `SQLAlchemy`
- `python-dotenv`
- `psutil`
- `ollama`
- `llama-cpp-python` (optional backend-specific dependency; needed only for `backend=llama_cpp` profiles)

#### B. ML runtime (`requirements/ml.txt`)
Required for audio transcription/diarization:

- `torch==2.3.0+cu121`
- `torchaudio==2.3.0+cu121`
- `whisperx==3.3.1`
- `pyannote.audio==3.4.0`
- `openai-whisper`
- `pydub`
- `transformers`
- `huggingface-hub`
- `numpy`
- `tqdm`

#### C. Development/test (`requirements/dev.txt`)
Required only for contributor workflows such as automated testing:

- everything from runtime + ML groups
- `pytest`

### Services/config required by the canonical path

- `DATABASE_URL` pointing at PostgreSQL.
- `models.yaml` present and valid.
- `ACTIVE_MODEL_PROFILE` optional only if `default_model` in `models.yaml` is valid.
- `HF_TOKEN` recommended for supported diarization flows with pyannote/Hugging Face access.
- Ollama daemon available locally when using an `ollama` profile.
- Local GGUF file present when using a `llama_cpp` profile.

## 3. Known mismatch / instability areas

These are intentionally documented instead of hidden:

- `whisperx`, `pyannote.audio`, and `torch` remain a compatibility-sensitive stack; changes to one often require coordinated upgrades of the others.
- The repository still contains legacy/Qdrant-era modules, but Qdrant is **not** part of the canonical runtime contract.
- `HF_TOKEN` is not hard-failed by CLI preflight today, but real diarization flows can still fail or require downloads without it.
- The supported matrix is intentionally narrower than “anything importable”; unsupported combinations may still partially run, but they are not considered contractually supported.

## 4. Bootstrap workflows

### Runtime-only bootstrap

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/runtime.txt
pip install -r requirements/ml.txt
cp .env.example .env
# then fill at least DATABASE_URL and, for diarization, HF_TOKEN
python -m src.app.runtime_doctor
```

### Development/test bootstrap

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env
python -m pytest -m unit
```

## 5. Runtime doctor usage

The repository now includes a lightweight environment report for operators and contributors. It automatically loads a local `.env` file before evaluating the environment so local bootstrap behavior matches the canonical CLI.

### Basic usage

```bash
python -m src.app.runtime_doctor
```

If `.env` exists in the working directory, the doctor loads it first without overwriting already-exported shell variables.

### JSON output

```bash
python -m src.app.runtime_doctor --json
```

### Optional database connectivity check

```bash
python -m src.app.runtime_doctor --check-db-connection
```

### Strict CI-style mode

```bash
python -m src.app.runtime_doctor --strict
```

The doctor reports:

- Python version
- active model profile/backend from `models.yaml`
- key package versions
- presence of required env vars
- whether GPU support appears present and whether CUDA matches the documented baseline
- whether DB prerequisites are configured, and optionally whether a live DB connection succeeds
- missing local GGUF path for `llama_cpp` profiles

## 6. Canonical validation commands

### Runtime contract validation

```bash
python -m src.app.runtime_doctor
```

If `.env` exists in the working directory, the doctor loads it first without overwriting already-exported shell variables.

### Canonical CLI preflight / runtime invocation

```bash
PYTHONPATH=. python -m src.app.cli path/to/audio.wav
```

### Test validation

```bash
python -m pytest -m unit
python -m pytest -m integration
```

## 7. Unsupported / postponed areas

Deliberately **not** done in this PR:

- full removal of the remaining compatibility-only modules in `src.legacy`
- making CPU execution a first-class supported transcription target
- auto-repair or hidden fallback logic for incompatible ML stacks
- a containerized/relocked dependency solver for every platform

The remaining legacy surface is intentionally narrow:

- `src/legacy/v1/pipeline/main.py` for manual compatibility workflows
- `src/legacy/v1/storage/*` for old storage import paths
- thin config/adaptor shims that point back to canonical modules

Historical snapshots and dead one-off legacy helpers are not part of the supported
runtime contract.

This PR is about making the current canonical runtime **inspectable, reproducible, and honest**.
