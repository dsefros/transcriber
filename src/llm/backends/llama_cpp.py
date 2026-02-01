from typing import Dict, Any

from llama_cpp import Llama

from src.llm.backends.base import LLMBackend


class LlamaCppBackend(LLMBackend):
    """
    Backend для llama.cpp.
    Загружает модель один раз и выполняет синхронный инференс.
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile

        model_path = profile.get("path")
        if not model_path:
            raise ValueError("llama_cpp backend requires 'path' in profile")

        params = profile.get("params", {})

        # --- Инициализация модели (ОДИН РАЗ) ---
        self.llm = Llama(
            model_path=model_path,
            n_ctx=params.get("n_ctx", 4096),
            n_gpu_layers=params.get("n_gpu_layers", 0),
            n_batch=params.get("n_batch", 512),
            verbose=params.get("verbose", False),
        )

        # --- Параметры генерации по умолчанию ---
        self.default_generation_params: Dict[str, Any] = {
            "temperature": params.get("temperature", 0.1),
            "top_p": params.get("top_p", 0.9),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
            "max_tokens": params.get("max_tokens", 2048),
        }

    def generate(self, prompt: str, params: Dict[str, Any] | None = None) -> str:
        """
        Выполнить синхронный инференс.

        :param prompt: готовый prompt
        :param params: параметры генерации (переопределяют defaults)
        :return: сгенерированный текст
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

        # llama_cpp всегда возвращает choices
        return result["choices"][0]["text"].strip()
