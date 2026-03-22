"""Lightweight preflight checks for the canonical CLI/runtime path.

This module is intentionally dependency-light so operators get fast, actionable
errors before worker construction triggers database connections or ML runtime
initialization.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from src.config.models import ModelConfigError, ModelProfile, get_active_model_profile, load_models_config


class PreflightError(RuntimeError):
    """Raised when the canonical runtime cannot start safely."""


_ML_INSTALL_HINT = "pip install -r requirements-ml.txt"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _validate_backend_dependencies(profile: ModelProfile) -> None:
    if profile.backend == "ollama":
        if not _module_available("ollama"):
            raise PreflightError(
                "Ollama backend selected but Python package 'ollama' is not installed. "
                "Install the runtime dependencies with: pip install -r requirements.txt"
            )
        return

    if profile.backend == "llama_cpp":
        if not _module_available("llama_cpp"):
            raise PreflightError(
                "llama_cpp backend selected but llama-cpp-python is not installed. "
                "Install the runtime dependencies with: pip install -r requirements.txt"
            )
        if profile.path and not Path(profile.path).exists():
            raise PreflightError(
                f"llama_cpp backend selected but model file does not exist: {profile.path}"
            )
        return

    raise PreflightError(f"Unsupported model backend configured: {profile.backend}")


def _validate_transcription_dependencies() -> None:
    missing = [name for name in ("torch", "whisperx", "pydub") if not _module_available(name)]
    if missing:
        missing_csv = ", ".join(missing)
        raise PreflightError(
            "WhisperX transcription runtime requires missing dependencies: "
            f"{missing_csv}. Install with: {_ML_INSTALL_HINT}"
        )


def run_preflight(source_path: Path, *, source_type: str, models_config_path: str = "models.yaml") -> dict[str, str]:
    """Validate the canonical runtime inputs before expensive startup work."""
    resolved_source = Path(source_path)
    if not resolved_source.exists():
        raise PreflightError(f"Input source path does not exist: {resolved_source}")

    if source_type == "audio":
        _validate_transcription_dependencies()

    try:
        models_config = load_models_config(models_config_path)
        profile = get_active_model_profile(models_config)
    except ModelConfigError as exc:
        raise PreflightError(f"Model configuration preflight failed: {exc}") from exc

    _validate_backend_dependencies(profile)

    if not os.getenv("DATABASE_URL"):
        raise PreflightError(
            "DATABASE_URL is required for the active runtime worker. "
            "Set DATABASE_URL before running the CLI."
        )

    return {
        "source_path": str(resolved_source),
        "source_type": source_type,
        "model_profile": profile.key,
        "model_backend": profile.backend,
    }
