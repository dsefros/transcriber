from src.core.jobs.runner import JobRunner
from src.core.jobs.repository import InMemoryJobRepository
from src.core.jobs.models import Job
from src.core.pipeline.services import Services



class Worker:
    def __init__(self):
        self.repo = InMemoryJobRepository()
        self.runner = JobRunner(self.repo)

    def submit(self, job: Job):
        self.repo.save(job)
        self.runner.run(job)
        return job
