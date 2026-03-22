"""Canonical model configuration loader for the active runtime."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ModelConfigError(Exception):
    """Ошибка конфигурации модели."""


class ModelProfile:
    """Профиль модели с валидацией."""

    def __init__(self, key: str, config: Dict[str, Any]):
        self.key = key
        self.backend = config.get("backend")
        self.name = config.get("name")
        self.path = config.get("path")
        self.description = config.get("description", "")
        self.params = config.get("params", {})

        if not self.backend:
            raise ModelConfigError(f"Профиль '{key}': отсутствует поле 'backend'")

        if self.backend == "ollama" and not self.name:
            raise ModelConfigError(
                f"Профиль '{key}': для backend='ollama' требуется поле 'name'"
            )
        if self.backend == "llama_cpp" and not self.path:
            raise ModelConfigError(
                f"Профиль '{key}': для backend='llama_cpp' требуется поле 'path'"
            )

    def to_backend_config(self) -> Dict[str, Any]:
        """Return the normalized dict shape expected by active backends."""
        config: Dict[str, Any] = {
            "profile_name": self.key,
            "backend": self.backend,
            "description": self.description,
            "params": self.params,
        }

        if self.name is not None:
            config["name"] = self.name
        if self.path is not None:
            config["path"] = self.path

        return config

    def __repr__(self):
        return f"<ModelProfile key={self.key} backend={self.backend}>"


class ModelsConfig:
    """Container for model profiles loaded from models.yaml."""

    def __init__(self, config_path: str = "models.yaml"):
        self.config_path = Path(config_path)
        self.profiles: Dict[str, ModelProfile] = {}
        self.default_model: str = ""
        self._load()

    def _load(self):
        if not self.config_path.exists():
            raise ModelConfigError(f"Файл конфигурации не найден: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        self.default_model = raw_config.get("default_model", "")
        profiles_raw = raw_config.get("profiles", {})

        if not profiles_raw:
            raise ModelConfigError("В конфигурации отсутствуют профили моделей")

        for key, cfg in profiles_raw.items():
            try:
                self.profiles[key] = ModelProfile(key, cfg)
            except ModelConfigError as e:
                print(f"⚠️ Пропущен некорректный профиль '{key}': {e}")

    def get_active_profile(self) -> ModelProfile:
        """Return the active profile selected by env override or default_model."""
        active_key = os.getenv("ACTIVE_MODEL_PROFILE", self.default_model)

        if not active_key:
            raise ModelConfigError(
                "Не указана активная модель. Установите ACTIVE_MODEL_PROFILE в .env "
                "или задайте default_model в models.yaml"
            )

        profile = self.profiles.get(active_key)
        if not profile:
            available = ", ".join(self.profiles.keys())
            raise ModelConfigError(
                f"Профиль '{active_key}' не найден. Доступные профили: {available}"
            )

        return profile

    def list_profiles(self) -> Dict[str, str]:
        """Return {profile_key: description} for all configured profiles."""
        return {key: profile.description for key, profile in self.profiles.items()}



def load_models_config(config_path: str = "models.yaml") -> ModelsConfig:
    """Canonical active-runtime loader for model configuration."""
    return ModelsConfig(config_path=config_path)


def get_active_model_profile(
    models_config: ModelsConfig | None = None,
    config_path: str = "models.yaml",
) -> ModelProfile:
    """Return the active typed profile from the canonical active-runtime config."""
    if models_config is None:
        models_config = load_models_config(config_path=config_path)
    return models_config.get_active_profile()
