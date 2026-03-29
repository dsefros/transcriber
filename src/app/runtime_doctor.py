from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config.env import load_env_file_if_present
from src.config.models import ModelConfigError, get_active_model_profile, load_models_config
from src.infrastructure.transcription.whisperx_runtime import get_transcription_settings

SUPPORTED_PYTHON = ((3, 10), (3, 11))
SUPPORTED_TORCH = "2.3.0+cu121"
SUPPORTED_PYANNOTE = "3.4.0"
SUPPORTED_WHISPERX = "3.3.1"
SUPPORTED_CUDA = "12.1"
DEFAULT_MODELS_PATH = "models.yaml"
DEPRECATED_LLM_ENV_VARS = (
    "TEMPERATURE",
    "NUM_CTX",
    "NUM_PREDICT",
    "TOP_P",
    "REPEAT_PENALTY",
)

RUNTIME_PACKAGES = {
    "PyYAML": True,
    "SQLAlchemy": True,
    "python-dotenv": True,
    "psutil": True,
    "psycopg2-binary": True,
    "ollama": False,
    "llama-cpp-python": False,
}

ML_PACKAGES = {
    # Optional for the default lightweight server image.
    # Required only for full local audio/transcription execution paths.
    "torch": False,
    "torchaudio": False,
    "whisperx": False,
    "pyannote.audio": False,
    "pydub": False,
    "transformers": False,
    "huggingface-hub": False,
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class RuntimeDoctorError(RuntimeError):
    pass


def _package_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _python_status() -> CheckResult:
    current = sys.version_info[:3]
    major_minor = current[:2]
    supported_labels = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON)
    if major_minor in SUPPORTED_PYTHON:
        return CheckResult(
            "python",
            "ok",
            f"Python {current[0]}.{current[1]}.{current[2]} is within the supported baseline ({supported_labels}).",
        )
    return CheckResult(
        "python",
        "warn",
        f"Python {current[0]}.{current[1]}.{current[2]} is outside the documented supported baseline ({supported_labels}).",
    )


def _models_status(models_path: str) -> tuple[CheckResult, dict[str, Any]]:
    path = Path(models_path)
    if not path.exists():
        return CheckResult("models.yaml", "fail", f"Config file is missing: {path}"), {}

    try:
        config = load_models_config(str(path))
        profile = get_active_model_profile(config)
    except ModelConfigError as exc:
        return CheckResult("models.yaml", "fail", f"Model config is invalid: {exc}"), {}

    analysis_prompt = config.get_default_analysis_prompt()
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / analysis_prompt

    details = {
        "path": str(path.resolve()),
        "active_profile": profile.key,
        "backend": profile.backend,
        "analysis_prompt": analysis_prompt,
    }

    if not prompt_path.exists():
        details["analysis_prompt_warning"] = (
            f"analysis prompt '{analysis_prompt}' not found under src/prompts/"
        )

    if profile.backend == "llama_cpp" and profile.path:
        model_path = Path(profile.path)
        if model_path.exists():
            detail = (
                f"Loaded active profile '{profile.key}' (backend={profile.backend}); local GGUF path exists: {model_path}."
            )
            status = "ok"
        else:
            detail = (
                f"Loaded active profile '{profile.key}' (backend={profile.backend}); configured GGUF path is missing: {model_path}."
            )
            status = "warn"
        details["model_path"] = str(model_path)
        if "analysis_prompt_warning" in details and status == "ok":
            status = "warn"
        if "analysis_prompt_warning" in details:
            detail = f"{detail} WARNING: {details['analysis_prompt_warning']}"
        return CheckResult("models.yaml", status, detail), details

    detail = f"Loaded active profile '{profile.key}' (backend={profile.backend})."
    if "analysis_prompt_warning" in details:
        detail = f"{detail} WARNING: {details['analysis_prompt_warning']}"
        return CheckResult("models.yaml", "warn", detail), details
    return CheckResult("models.yaml", "ok", detail), details


def _env_var_status(name: str, *, required: bool, extra_hint: str | None = None) -> CheckResult:
    value = os.getenv(name)
    if value:
        return CheckResult(name, "ok", f"{name} is set.")
    if required:
        suffix = f" {extra_hint}" if extra_hint else ""
        return CheckResult(name, "fail", f"{name} is not set.{suffix}")
    suffix = f" {extra_hint}" if extra_hint else ""
    return CheckResult(name, "warn", f"{name} is not set.{suffix}")


def _deprecated_llm_env_var_checks(active_profile: str | None) -> list[CheckResult]:
    profile_label = active_profile or "the active profile"
    checks: list[CheckResult] = []
    for name in DEPRECATED_LLM_ENV_VARS:
        value = os.getenv(name)
        if not value:
            continue
        checks.append(
            CheckResult(
                name,
                "warn",
                f"{name} is set to '{value}' but is ignored by the canonical runtime; move this value into models.yaml under the '{profile_label}' profile params.",
            )
        )
    return checks


def _package_status(distribution_name: str, *, required: bool, expected: str | None = None) -> CheckResult:
    version = _package_version(distribution_name)
    if version is None:
        if required:
            return CheckResult(distribution_name, "fail", f"Package is missing (required).")
        return CheckResult(distribution_name, "warn", f"Package is missing (optional or backend-specific).")

    if expected and version != expected:
        status = "warn" if required else "info"
        return CheckResult(
            distribution_name,
            status,
            f"Installed version is {version}; documented baseline is {expected}.",
        )

    return CheckResult(distribution_name, "ok", f"Installed version is {version}.")


def _collect_package_checks(active_backend: str | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for package_name, required in RUNTIME_PACKAGES.items():
        backend_required = required
        if package_name == "ollama":
            backend_required = active_backend == "ollama"
        elif package_name == "llama-cpp-python":
            backend_required = active_backend == "llama_cpp"
        results.append(_package_status(package_name, required=backend_required))

    for package_name, required in ML_PACKAGES.items():
        expected = None
        if package_name == "torch":
            expected = SUPPORTED_TORCH
        elif package_name == "pyannote.audio":
            expected = SUPPORTED_PYANNOTE
        elif package_name == "whisperx":
            expected = SUPPORTED_WHISPERX
        results.append(_package_status(package_name, required=required, expected=expected))

    return results


def _gpu_status() -> CheckResult:
    torch_version = _package_version("torch")
    if torch_version is None:
        return CheckResult("gpu", "warn", "torch is not installed; GPU/audio transcription stack validation is skipped for lightweight runtime images.")

    import importlib

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return CheckResult(
            "gpu",
            "warn",
            f"torch is installed but could not be imported cleanly: {exc}",
        )

    cuda_available = bool(torch.cuda.is_available())
    runtime_cuda = getattr(torch.version, "cuda", None)
    if cuda_available and runtime_cuda == SUPPORTED_CUDA:
        return CheckResult(
            "gpu",
            "ok",
            f"CUDA is available through torch and reports CUDA {runtime_cuda}, matching the supported baseline.",
        )
    if cuda_available:
        return CheckResult(
            "gpu",
            "warn",
            f"CUDA is available through torch but reports CUDA {runtime_cuda}; documented baseline is {SUPPORTED_CUDA}.",
        )
    return CheckResult(
        "gpu",
        "warn",
        "CUDA is not available. The canonical WhisperX runtime is documented as GPU-backed; CPU-only execution is best-effort and unsupported.",
    )


def _database_status(*, check_connection: bool) -> CheckResult:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return CheckResult("database", "fail", "DATABASE_URL is not set.")

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url
    except Exception as exc:  # pragma: no cover - defensive import path
        return CheckResult("database", "fail", f"SQLAlchemy is unavailable: {exc}")

    try:
        parsed = make_url(database_url)
    except Exception as exc:
        return CheckResult("database", "fail", f"DATABASE_URL could not be parsed: {exc}")

    if not check_connection:
        return CheckResult(
            "database",
            "ok",
            f"DATABASE_URL is configured for dialect '{parsed.get_backend_name()}'; connection not attempted.",
        )

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        return CheckResult("database", "warn", f"DATABASE_URL is configured but connection check failed: {exc}")

    return CheckResult("database", "ok", "DATABASE_URL is configured and a lightweight connection check succeeded.")


def collect_runtime_report(*, models_path: str = DEFAULT_MODELS_PATH, check_db_connection: bool = False) -> dict[str, Any]:
    load_env_file_if_present()

    report: dict[str, Any] = {
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "supported_baseline": {
            "python": [f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON],
            "gpu": "NVIDIA GPU with CUDA 12.1 userspace via torch 2.3.0+cu121",
            "database": "PostgreSQL via DATABASE_URL",
            "llm_backends": ["ollama", "llama_cpp"],
            "transcription": get_transcription_settings(),
        },
    }

    model_check, model_details = _models_status(models_path)
    active_backend = model_details.get("backend")
    active_profile = model_details.get("active_profile")

    checks = [
        _python_status(),
        model_check,
        _env_var_status("DATABASE_URL", required=True, extra_hint="Required before Worker startup."),
        _env_var_status("HF_TOKEN", required=False, extra_hint="Needed for pyannote diarization downloads/auth."),
        _env_var_status("TRANSCRIPTION_MODEL_NAME", required=False, extra_hint="Defaults to large-v3 for WhisperX transcription."),
        _env_var_status("TRANSCRIPTION_DEVICE", required=False, extra_hint="Defaults to cuda for WhisperX transcription."),
        _env_var_status("ALIGNMENT_LANGUAGE_CODE", required=False, extra_hint="Defaults to ru for WhisperX alignment loading."),
        _env_var_status("ALIGNMENT_MODEL_NAME", required=False, extra_hint="Defaults to facebook/wav2vec2-base-960h for WhisperX alignment loading."),
        *_deprecated_llm_env_var_checks(active_profile),
        _database_status(check_connection=check_db_connection),
        _gpu_status(),
        *_collect_package_checks(active_backend),
    ]

    report["models"] = model_details
    report["checks"] = [check.__dict__ for check in checks]
    report["summary"] = {
        "fail": sum(1 for check in checks if check.status == "fail"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "ok": sum(1 for check in checks if check.status == "ok"),
    }
    return report


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        "Runtime doctor report",
        f"Platform: Python {report['platform']['python']} on {report['platform']['system']} {report['platform']['machine']}",
        f"Supported Python: {', '.join(report['supported_baseline']['python'])}",
        f"Supported GPU baseline: {report['supported_baseline']['gpu']}",
        f"Supported database: {report['supported_baseline']['database']}",
    ]

    model_info = report.get("models") or {}
    if model_info:
        lines.append(
            f"Active model profile: {model_info.get('active_profile')} (backend={model_info.get('backend')})"
        )
        lines.append(
            f"Analysis prompt: {model_info.get('analysis_prompt')}"
        )

    lines.append("")
    for check in report["checks"]:
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "info": "INFO"}.get(check["status"], check["status"].upper())
        lines.append(f"[{icon}] {check['name']}: {check['detail']}")

    summary = report["summary"]
    lines.append("")
    lines.append(
        f"Summary: {summary['ok']} ok, {summary['warn']} warnings, {summary['fail']} failures"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Canonical runtime environment doctor")
    parser.add_argument("--models-config", default=DEFAULT_MODELS_PATH)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--check-db-connection", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings as well as failures.")
    args = parser.parse_args(argv)

    report = collect_runtime_report(
        models_path=args.models_config,
        check_db_connection=args.check_db_connection,
    )

    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_render_text(report))

    failures = report["summary"]["fail"]
    warnings = report["summary"]["warn"]
    if failures:
        return 1
    if args.strict and warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
