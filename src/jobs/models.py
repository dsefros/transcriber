from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class Job:
    id: UUID = field(default_factory=uuid4)
    source_type: str = ""
    source_path: str = ""
    status: JobStatus = JobStatus.PENDING

    current_step: str | None = None
    error: str | None = None
    attempt: int = 0            # ← ВОТ ЭТО ПОЛЕ

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

from dataclasses import dataclass
from enum import Enum
from uuid import UUID
from datetime import datetime
from typing import Optional, Dict, Any


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobStep:
    id: UUID
    job_id: UUID
    step_name: str

    status: StepStatus
    attempt: int

    artifacts: Optional[Dict[str, Any]]
    error: Optional[str]

    created_at: datetime
    updated_at: datetime
