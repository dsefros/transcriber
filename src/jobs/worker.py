from src.jobs.runner import JobRunner
from src.jobs.repository import InMemoryJobRepository
from src.jobs.models import Job


class Worker:
    def __init__(self):
        self.repo = InMemoryJobRepository()
        self.runner = JobRunner(self.repo)

    def submit(self, job: Job):
        self.repo.save(job)
        self.runner.run(job)
        return job
