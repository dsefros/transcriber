# src/pipeline_v2/services.py

from dataclasses import dataclass
from src.llm.adapter import LLMAdapter


@dataclass
class Services:
    """
    Infrastructure services container.
    Lives on Worker level, injected into PipelineContext.
    """
    llm: LLMAdapter
