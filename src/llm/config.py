import os
import yaml
from typing import Dict, Any


class ModelConfigError(Exception):
    pass


def load_models_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_model_profile(models_config: Dict[str, Any]) -> Dict[str, Any]:
    active_profile = os.getenv("ACTIVE_MODEL_PROFILE")

    if not active_profile:
        active_profile = models_config.get("default_model")

    profiles = models_config.get("profiles", {})

    if active_profile not in profiles:
        raise ModelConfigError(
            f"Model profile '{active_profile}' not found in models.yaml"
        )

    profile = profiles[active_profile]
    profile["profile_name"] = active_profile
    return profile
