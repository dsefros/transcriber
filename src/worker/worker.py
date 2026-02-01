from src.jobs.runner import JobRunner
from src.jobs.postgres_repository import PostgresJobRepository
from src.jobs.models import Job

from src.llm.adapter import LLMAdapter
from src.pipeline_v2.services import Services


class Worker:
    def __init__(self):
        # --- infrastructure ---
        self.repo = PostgresJobRepository()

        # 🔑 LLM INITIALIZED ONCE PER WORKER
        llm_adapter = LLMAdapter(models_config_path="models.yaml")

        self.services = Services(
            llm=llm_adapter
        )

        # --- runner ---
        self.runner = JobRunner(self.repo, self.services)

    def submit(self, job: Job) -> Job:
        self.repo.save(job)
        self.runner.run(job)
        return job
