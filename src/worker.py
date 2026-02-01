from src.core.jobs.runner import JobRunner
from src.core.jobs.repository import InMemoryJobRepository
from src.core.jobs.models import Job
from src.core.pipeline.services import Services

from src.infrastructure.llm.adapter import LLMAdapter





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
