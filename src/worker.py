from src.core.jobs.runner import JobRunner
from src.core.jobs.models import Job
from src.core.jobs.postgres_repository import PostgresJobRepository

from src.core.pipeline.services import Services

from src.infrastructure.transcription.legacy_adapter import LegacyTranscriptionAdapter
from src.infrastructure.llm.adapter import LLMAdapter


class Worker:
    def __init__(self):
        # --- infrastructure ---
        self.repo = PostgresJobRepository()

        # Transcription backend (legacy adapter for now)
        self.transcription = LegacyTranscriptionAdapter()

        # LLM initialized once per worker (lazy model load inside)
        self.llm_adapter = LLMAdapter(models_config_path="models.yaml")

        # --- application services ---
        self.services = Services(
            llm_adapter=self.llm_adapter,
            transcription=self.transcription,
        )

        # --- job runner ---
        self.runner = JobRunner(self.repo, self.services)

    def submit(self, job: Job) -> Job:
        self.repo.save(job)
        self.runner.run(job)
        return job
