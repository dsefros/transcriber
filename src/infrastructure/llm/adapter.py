from typing import Dict, Any, Optional

from src.infrastructure.llm.config import (
    load_models_config,
    get_active_model_profile,
)

from src.infrastructure.llm.backends.ollama import OllamaBackend
from src.infrastructure.llm.backends.llama_cpp import LlamaCppBackend
from src.infrastructure.llm.types import LLMMetadata


class LLMAdapter:
    """
    Infrastructure-level LLM adapter.

    Responsibilities:
    - load model config
    - select backend
    - lazy backend initialization
    - expose unified metadata
    """

    def __init__(self, models_config_path: str):
        print("🧠 LLMAdapter created (lazy init, no model loaded)")

        self.models_config = load_models_config(models_config_path)
        self.profile = get_active_model_profile(self.models_config)

        self._backend: Optional[Any] = None  # lazy-loaded

    def _init_backend(self):
        if self._backend is not None:
            return

        backend_type = self.profile.get("backend")

        if backend_type == "ollama":
            self._backend = OllamaBackend(self.profile)
        elif backend_type == "llama_cpp":
            self._backend = LlamaCppBackend(self.profile)
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

    def generate(self, prompt: str) -> str:
        """
        Run inference via selected backend.
        """
        self._init_backend()

        params: Dict[str, Any] = self.profile.get("params", {})
        return self._backend.generate(prompt, params)

    @property
    def meta(self) -> LLMMetadata:
        """
        Unified backend metadata.
        """
        self._init_backend()
        return self._backend.meta
