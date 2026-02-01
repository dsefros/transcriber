from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


@dataclass
class StepResult:
    status: Literal["completed", "failed"]
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Step:
    """
    Stateless pipeline step.
    MUST NOT store or check execution state.
    """

    name: str

    def run(self, ctx) -> StepResult:
        raise NotImplementedError
