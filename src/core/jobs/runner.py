from datetime import datetime
from pathlib import Path
import traceback

from src.core.jobs.models import Job, JobStatus
from src.core.pipeline.context import PipelineContext
from src.core.pipeline.orchestrator import PipelineOrchestrator
from src.core.pipeline.services import Services



class JobRunner:
    def __init__(self, repo, services: Services):
        self.repo = repo
        self.services = services

    def run(self, job: Job):
        try:
            # --- job start ---
            job.status = JobStatus.RUNNING
            job.current_step = None
            job.error = None
            job.updated_at = datetime.utcnow()
            self.repo.update(job)

            # --- pipeline context ---
            ctx = PipelineContext(
                job_id=job.id,
                source_type=job.source_type,
                source_path=Path(job.source_path),
                services=self.services,      # 🔑 inject services
            )

            # --- PIPELINE ---
            orchestrator = PipelineOrchestrator()
            orchestrator.run(job, ctx)

            # --- job success ---
            job.status = JobStatus.COMPLETED
            job.updated_at = datetime.utcnow()
            self.repo.update(job)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.updated_at = datetime.utcnow()
            self.repo.update(job)

            traceback.print_exc()
