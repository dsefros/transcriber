from typing import Dict
from src.jobs.models import Job


class InMemoryJobRepository:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def save(self, job: Job):
        self._jobs[str(job.id)] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job: Job):
        self._jobs[str(job.id)] = job
