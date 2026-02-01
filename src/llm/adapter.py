from typing import Dict, Any, Optional

from src.llm.config import load_models_config, get_active_model_profile
from src.llm.backends.ollama import OllamaBackend
from src.llm.backends.llama_cpp import LlamaCppBackend


class LLMAdapter:
    """
    Infrastructure-level LLM adapter.

    IMPORTANT:
    - Does NOT load model on init
    - Loads backend lazily on first generate()
    """

    def __init__(self, models_config_path: str):
        print("🧠 LLMAdapter created (lazy init, no model loaded)")

        self.models_config = load_models_config(models_config_path)
        self.profile = get_active_model_profile(self.models_config)

        self._backend: Optional[Any] = None  # lazy-loaded

    def _init_backend(self):
        """Initialize LLM backend lazily."""
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
        # 🔑 Lazy initialization happens HERE
        self._init_backend()

        params: Dict[str, Any] = self.profile.get("params", {})
        return self._backend.generate(prompt, params)
