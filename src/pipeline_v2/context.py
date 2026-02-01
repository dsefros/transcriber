from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID
from typing import Dict, Any
import hashlib

from src.pipeline_v2.services import Services


@dataclass
class PipelineContext:
    """
    Stateless pipeline context.
    Used ONLY to pass data between steps.
    """

    job_id: UUID
    source_type: str           # audio | json
    source_path: Path
    services: Services         # 🔑 DI container

    source_hash: str = field(init=False)

    # Data-plane artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.source_type == "audio":
            with open(self.source_path, "rb") as f:
                self.source_hash = hashlib.sha256(f.read()).hexdigest()
        else:
            self.source_hash = hashlib.sha256(
                str(self.source_path).encode()
            ).hexdigest()
