from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from src.config.models import get_active_model_profile, load_models_config
from src.infrastructure.llm.types import LLMMetadata


class LLMDependencyError(RuntimeError):
    """Raised when the selected LLM backend dependencies are unavailable."""


class LLMAdapter:
    """
    Infrastructure-level LLM adapter.

    Responsibilities:
    - load model config from the canonical active-runtime source
    - select backend
    - lazy backend initialization
    - expose unified metadata
    """

    def __init__(self, models_config_path: str):
        print("🧠 LLMAdapter created (lazy init, no model loaded)")

        self.models_config = load_models_config(models_config_path)
        self.profile = get_active_model_profile(self.models_config)
        self._backend: Optional[Any] = None

    def _load_backend_class(self):
        backend_type = self.profile.backend
        try:
            if backend_type == "ollama":
                module = importlib.import_module("src.infrastructure.llm.backends.ollama")
                return module.OllamaBackend
            if backend_type == "llama_cpp":
                module = importlib.import_module("src.infrastructure.llm.backends.llama_cpp")
                return module.LlamaCppBackend
        except ModuleNotFoundError as exc:
            package_name = exc.name or backend_type
            raise LLMDependencyError(
                f"LLM backend '{backend_type}' requires missing dependency '{package_name}'. "
                "Install the runtime dependencies with: pip install -r requirements.txt"
            ) from exc

        raise ValueError(f"Unsupported backend: {backend_type}")

    def _init_backend(self):
        if self._backend is not None:
            return

        backend_class = self._load_backend_class()
        backend_config = self.profile.to_backend_config()
        self._backend = backend_class(backend_config)

    def generate(self, prompt: str) -> str:
        self._init_backend()
        params: Dict[str, Any] = self.profile.params
        return self._backend.generate(prompt, params)

    @property
    def meta(self) -> LLMMetadata:
        self._init_backend()
        return self._backend.meta
