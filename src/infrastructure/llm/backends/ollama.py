from typing import Dict, Any
from src.infrastructure.llm.backends.base import LLMBackend



class OllamaBackend(LLMBackend):
    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.model_name = profile.get("name")

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError("OllamaBackend not implemented yet")
