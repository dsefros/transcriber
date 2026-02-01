from uuid import uuid4
from datetime import datetime
import json

from sqlalchemy import text

from src.core.jobs.models import JobStep, StepStatus
from src.infrastructure.storage.postgres import init_db



class JobStepRepository:
    """
    Control-plane repository for pipeline steps.
    НЕ путать с JobRepository.
    """

    def __init__(self):
        self.engine = init_db()

    def get(self, job_id, step_name):
        with self.engine.connect() as conn:
            row = conn.execute(
                text("""
                SELECT *
                FROM job_steps
                WHERE job_id = :job_id
                  AND step_name = :step_name
                """),
                {"job_id": job_id, "step_name": step_name},
            ).mappings().first()

        if not row:
            return None

        return JobStep(
            id=row["id"],
            job_id=row["job_id"],
            step_name=row["step_name"],
            status=StepStatus(row["status"]),
            attempt=row["attempt"],
            artifacts=row["artifacts"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_if_not_exists(self, job_id, step_name):
        step = self.get(job_id, step_name)
        if step:
            return step

        step_id = uuid4()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO job_steps (
                    id, job_id, step_name, status, attempt
                )
                VALUES (
                    :id, :job_id, :step_name, :status, 0
                )
                """),
                {
                    "id": step_id,
                    "job_id": job_id,
                    "step_name": step_name,
                    "status": StepStatus.PENDING.value,
                },
            )

        return self.get(job_id, step_name)

    def mark_running(self, step):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE job_steps
                SET status = :status,
                    attempt = attempt + 1,
                    updated_at = :updated_at
                WHERE id = :id
                """),
                {
                    "id": step.id,
                    "status": StepStatus.RUNNING.value,
                    "updated_at": datetime.utcnow(),
                },
            )

    import json

    def mark_completed(self, step, artifacts):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE job_steps
                SET status = :status,
                    artifacts = CAST(:artifacts AS JSONB),
                    error = NULL,
                    updated_at = :updated_at
                WHERE id = :id
                """),
                {
                    "id": step.id,
                    "status": StepStatus.COMPLETED.value,
                    "artifacts": json.dumps(artifacts),
                    "updated_at": datetime.utcnow(),
                },
            )


    def mark_failed(self, step, error):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE job_steps
                SET status = :status,
                    error = :error,
                    updated_at = :updated_at
                WHERE id = :id
                """),
                {
                    "id": step.id,
                    "status": StepStatus.FAILED.value,
                    "error": error,
                    "updated_at": datetime.utcnow(),
                },
            )
