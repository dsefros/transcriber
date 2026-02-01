from typing import Dict, Any

from llama_cpp import Llama

from src.infrastructure.llm.backends.base import LLMBackend
from src.infrastructure.llm.types import LLMMetadata


class LlamaCppBackend(LLMBackend):
    """
    Backend for llama.cpp.

    Characteristics:
    - completion-based (not chat)
    - no native system prompt support
    - local model, loaded into process
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile

        # --- metadata fields ---
        self.profile_name: str = profile.get("profile_name", "unknown")

        params = profile.get("params", {})
        self.context_size: int = params.get("n_ctx", 4096)

        # model identification
        self.model_path: str = profile.get("path")
        if not self.model_path:
            raise ValueError("llama_cpp backend requires 'path' in model profile")

        self.model_name: str = profile.get("model_id", self.model_path)

        # --- model init ---
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_gpu_layers=params.get("n_gpu_layers", 0),
            n_batch=params.get("n_batch", 512),
            verbose=params.get("verbose", False),
        )

        # --- default generation params ---
        self.default_generation_params: Dict[str, Any] = {
            "temperature": params.get("temperature", 0.1),
            "top_p": params.get("top_p", 0.9),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
            "max_tokens": params.get("max_tokens", 2048),
        }

    def generate(self, prompt: str, params: Dict[str, Any] | None = None) -> str:
        """
        Run synchronous completion inference.
        """
        if params is None:
            params = {}

        generation_params = {
            **self.default_generation_params,
            **params,
        }

        result = self.llm(
            prompt,
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"],
            repeat_penalty=generation_params["repeat_penalty"],
            max_tokens=generation_params["max_tokens"],
            echo=False,
        )

        return result["choices"][0]["text"].strip()

    @property
    def meta(self) -> LLMMetadata:
        """
        Canonical backend metadata.
        """
        return {
            "backend": "llama_cpp",
            "model": self.model_name,
            "profile": self.profile_name,
            "context_size": self.context_size,
            "supports_chat": False,
            "supports_system_prompt": False,
        }
