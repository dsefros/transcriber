"""Compatibility wrappers around the canonical active-runtime model config loader.

Active runtime code should import from ``src.config.models`` directly.
This module remains only as a migration-safe shim for older infrastructure imports.
"""
import os
from typing import Any, Dict

from src.config.models import ModelConfigError, load_models_config as load_canonical_models_config


def load_models_config(path: str) -> Dict[str, Any]:
    """Return a dict view of the canonical config for compatibility callers."""
    config = load_canonical_models_config(config_path=path)
    return {
        "default_model": config.default_model,
        "profiles": {
            key: profile.to_backend_config()
            for key, profile in config.profiles.items()
        },
    }


def get_active_model_profile(models_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the active profile using canonical env/default selection semantics."""
    active_profile = os.getenv("ACTIVE_MODEL_PROFILE") or models_config.get("default_model")
    profiles = models_config.get("profiles", {})

    if active_profile not in profiles:
        raise ModelConfigError(
            f"Model profile '{active_profile}' not found in models.yaml"
        )

    return profiles[active_profile]
