from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from src.core.jobs.models import Job, JobStatus
from src.infrastructure.storage.postgres import init_db



class PostgresJobRepository:
    def __init__(self):
        self.engine = init_db()

    def save(self, job: Job):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO jobs (
                    id, source_type, source_path,
                    status, current_step, error,
                    attempt, created_at, updated_at
                )
                VALUES (
                    :id, :source_type, :source_path,
                    :status, :current_step, :error,
                    :attempt, :created_at, :updated_at
                )
                """),
                {
                    "id": job.id,
                    "source_type": job.source_type,
                    "source_path": job.source_path,
                    "status": job.status.value,
                    "current_step": job.current_step,
                    "error": job.error,
                    "attempt": job.attempt,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                }
            )

    def update(self, job: Job):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE jobs
                SET status = :status,
                    current_step = :current_step,
                    error = :error,
                    attempt = :attempt,
                    updated_at = :updated_at
                WHERE id = :id
                """),
                {
                    "id": job.id,
                    "status": job.status.value,
                    "current_step": job.current_step,
                    "error": job.error,
                    "attempt": job.attempt,
                    "updated_at": datetime.utcnow(),
                }

            )

    def get(self, job_id: UUID) -> Job | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM jobs WHERE id = :id"),
                {"id": job_id}
            ).mappings().first()

        if not row:
            return None

        return Job(
            id=row["id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            status=JobStatus(row["status"]),
            current_step=row["current_step"],
            error=row["error"],
            attempt=row["attempt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
