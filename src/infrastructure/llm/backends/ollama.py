from typing import Dict, Any

import ollama

from src.infrastructure.llm.backends.base import LLMBackend
from src.infrastructure.llm.types import LLMMetadata


class OllamaBackend(LLMBackend):
    """
    Backend for Ollama.

    Characteristics:
    - chat-based
    - supports system prompts
    - external daemon (stateless from our side)
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile

        # --- metadata fields ---
        self.profile_name: str = profile.get("profile_name", "unknown")

        self.model_name: str = profile.get("name")
        if not self.model_name:
            raise ValueError("Ollama backend requires 'name' in model profile")

        params = profile.get("params", {})
        self.context_size = params.get("context_size")  # may be None

        # --- client ---
        self.client = ollama.Client()

        # --- default generation params ---
        self.default_generation_params: Dict[str, Any] = {
            "temperature": params.get("temperature", 0.1),
            "top_p": params.get("top_p", 0.9),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
            "num_predict": params.get("num_predict", 2048),
        }

    def generate(self, prompt: str, params: Dict[str, Any] | None = None) -> str:
        """
        Run synchronous chat-based inference.
        """
        if params is None:
            params = {}

        generation_params = {
            **self.default_generation_params,
            **params,
        }

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options=generation_params,
            stream=False,
        )

        return response["message"]["content"].strip()

    @property
    def meta(self) -> LLMMetadata:
        """
        Canonical backend metadata.
        """
        return {
            "backend": "ollama",
            "model": self.model_name,
            "profile": self.profile_name,
            "context_size": self.context_size,
            "supports_chat": True,
            "supports_system_prompt": True,
        }
