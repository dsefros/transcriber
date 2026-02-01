from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class AnalysisResult:
    """
    Stable contract for analysis output.
    """

    prompt_id: str
    generated_at: datetime

    summary_raw: str
    segment_count: int

    model_backend: str
    model_profile: str
