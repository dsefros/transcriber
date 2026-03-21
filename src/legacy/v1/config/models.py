"""Legacy-only compatibility shim for the retired v1 model-config import path.

The active runtime uses :mod:`src.config.models` directly.
This module remains only so the legacy v1 pipeline and manual workflows can keep
importing the old path without reviving a separate configuration implementation.
"""

from dotenv import load_dotenv

from src.config.models import (
    ModelConfigError,
    ModelProfile,
    ModelsConfig,
    get_active_model_profile,
    get_models_config,
    load_models_config,
)

load_dotenv()

__all__ = [
    "ModelConfigError",
    "ModelProfile",
    "ModelsConfig",
    "load_models_config",
    "get_models_config",
    "get_active_model_profile",
]
