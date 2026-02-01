# src/pipeline_v2/services.py

from dataclasses import dataclass
from src.infrastructure.llm.adapter import LLMAdapter
from src.core.transcription.port import TranscriptionPort


@dataclass
class Services:
    def __init__(self, llm_adapter, transcription: TranscriptionPort):
        self.llm = llm_adapter
        self.transcription = transcription
