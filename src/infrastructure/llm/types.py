from typing import TypedDict, Literal, Optional


class LLMMetadata(TypedDict):
    """
    Canonical metadata for any LLM backend.
    Stable contract between infrastructure and core.
    """

    backend: Literal["ollama", "llama_cpp"]
    model: str                  # model id / name
    profile: str                # profile key from models.yaml

    context_size: Optional[int]

    supports_chat: bool
    supports_system_prompt: bool
