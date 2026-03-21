from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


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

    summary: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    transcript_preview: str = ""
    normalization: Dict[str, int] = field(default_factory=dict)
    parse_error: str | None = None
