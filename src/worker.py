from src.core.jobs.models import Job
from src.core.jobs.runner import JobRunner
from src.core.pipeline.services import Services


class Worker:
    def __init__(self):
        # Import infrastructure at runtime so CLI help and lightweight tests can
        # import the canonical path without pulling database or ML dependencies.
        from src.core.jobs.postgres_repository import PostgresJobRepository
        from src.infrastructure.llm.adapter import LLMAdapter
        from src.infrastructure.transcription.whisperx_adapter import WhisperXTranscriptionAdapter

        self.repo = PostgresJobRepository()
        self.transcription = WhisperXTranscriptionAdapter()
        self.llm_adapter = LLMAdapter(models_config_path="models.yaml")
        self.services = Services(
            llm_adapter=self.llm_adapter,
            transcription=self.transcription,
        )
        self.runner = JobRunner(self.repo, self.services)

    def submit(self, job: Job) -> Job:
        self.repo.save(job)
        self.runner.run(job)
        return job

    def close(self) -> None:
        self.llm_adapter.close()
