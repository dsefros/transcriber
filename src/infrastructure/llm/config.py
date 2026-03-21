"""Legacy-only dict compatibility shim around the canonical model config loader.

Active runtime code should import from ``src.config.models`` directly.
This module remains only for older/manual callers that still expect a plain-dict
view of the canonical configuration during the migration.
"""

import os
from typing import Any, Dict, Union

from src.config.models import (
    ModelConfigError,
    ModelsConfig,
    load_models_config as load_canonical_models_config,
)

CompatModelsConfig = Dict[str, Any]
CompatProfile = Dict[str, Any]


def _to_compat_profile(profile) -> CompatProfile:
    """Convert a canonical model profile into the legacy dict shape."""
    return profile.to_backend_config()


def _to_compat_models_config(config: ModelsConfig) -> CompatModelsConfig:
    """Convert canonical typed config into the legacy dict container."""
    return {
        "default_model": config.default_model,
        "profiles": {
            key: _to_compat_profile(profile) for key, profile in config.profiles.items()
        },
    }


def load_models_config(path: str) -> CompatModelsConfig:
    """Return a dict-shaped compatibility view of the canonical config loader."""
    return _to_compat_models_config(load_canonical_models_config(config_path=path))


def get_active_model_profile(
    models_config: Union[CompatModelsConfig, ModelsConfig],
) -> CompatProfile:
    """Return the active profile using canonical env/default selection semantics."""
    if isinstance(models_config, ModelsConfig):
        active_key = os.getenv("ACTIVE_MODEL_PROFILE", models_config.default_model)
        profile = models_config.profiles.get(active_key)
        if profile is None:
            raise ModelConfigError(
                f"Model profile '{active_key}' not found in models.yaml"
            )
        return _to_compat_profile(profile)

    active_profile = os.getenv("ACTIVE_MODEL_PROFILE") or models_config.get(
        "default_model"
    )
    profiles = models_config.get("profiles", {})

    if active_profile not in profiles:
        raise ModelConfigError(
            f"Model profile '{active_profile}' not found in models.yaml"
        )

    return profiles[active_profile]
