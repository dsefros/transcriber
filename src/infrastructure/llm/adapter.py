from typing import Any, Dict, Optional

from src.config.models import get_active_model_profile, load_models_config
from src.infrastructure.llm.backends.ollama import OllamaBackend
from src.infrastructure.llm.backends.llama_cpp import LlamaCppBackend
from src.infrastructure.llm.types import LLMMetadata


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

        self._backend: Optional[Any] = None  # lazy-loaded

    def _init_backend(self):
        if self._backend is not None:
            return

        backend_type = self.profile.backend
        backend_config = self.profile.to_backend_config()

        if backend_type == "ollama":
            self._backend = OllamaBackend(backend_config)
        elif backend_type == "llama_cpp":
            self._backend = LlamaCppBackend(backend_config)
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

    def generate(self, prompt: str) -> str:
        """
        Run inference via selected backend.
        """
        self._init_backend()

        params: Dict[str, Any] = self.profile.params
        return self._backend.generate(prompt, params)

    @property
    def meta(self) -> LLMMetadata:
        """
        Unified backend metadata.
        """
        self._init_backend()
        return self._backend.meta
