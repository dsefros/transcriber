from abc import ABC, abstractmethod
from typing import Dict, Any

from src.infrastructure.llm.types import LLMMetadata


class LLMBackend(ABC):
    """
    Base contract for any LLM backend implementation.
    """

    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Run synchronous inference.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def meta(self) -> LLMMetadata:
        """
        Canonical metadata for this backend instance.
        Must be backend-agnostic and stable.
        """
        raise NotImplementedError
