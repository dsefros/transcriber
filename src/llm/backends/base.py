from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMBackend(ABC):
    """
    Базовый контракт для всех LLM backend'ов.
    """

    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Выполнить синхронный инференс.

        :param prompt: итоговый prompt
        :param params: параметры генерации (temperature, max_tokens, etc)
        :return: сгенерированный текст
        """
        raise NotImplementedError
