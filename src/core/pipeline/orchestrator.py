import logging
import time

from src.core.jobs.job_step_repository import JobStepRepository
from src.core.jobs.models import StepStatus


class PipelineOrchestrator:
    """
    Orchestrates execution of pipeline steps.
    Control-plane state is stored ONLY in job_steps table.
    """

    def __init__(self):
        self.repo = JobStepRepository()
        self.logger = logging.getLogger("pipeline")

        from src.core.pipeline.steps.analysis import AnalysisStep
        from src.core.pipeline.steps.transcription import TranscriptionStep

        self.steps = [
            TranscriptionStep(),
            AnalysisStep(),
        ]

    def run(self, job, ctx):
        """Execute pipeline for a given job and context."""

        for step in self.steps:
            step_name = step.name
            job.current_step = step_name
            ctx.services  # explicit dependency touch for readability

            self.logger.info(
                "step_started",
                extra={
                    "extra": {
                        "event": "step_started",
                        "job_id": str(job.id),
                        "step": step_name,
                        "component": "pipeline",
                    }
                },
            )

            step_state = self.repo.create_if_not_exists(job_id=job.id, step_name=step_name)

            if step_state.status == StepStatus.COMPLETED:
                if step_state.artifacts:
                    ctx.artifacts[step_name] = step_state.artifacts
                continue

            if step_state.status == StepStatus.RUNNING:
                raise RuntimeError(f"Step '{step_name}' already RUNNING for job {job.id}")

            self.repo.mark_running(step_state)
            started_at = time.monotonic()

            try:
                result = step.run(ctx)
            except Exception as exc:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                self.repo.mark_failed(step_state, str(exc))
                self.logger.error(
                    "step_exception",
                    extra={
                        "extra": {
                            "event": "step_exception",
                            "job_id": str(job.id),
                            "step": step_name,
                            "component": "pipeline",
                            "error": str(exc),
                            "duration_ms": duration_ms,
                        }
                    },
                )
                raise

            duration_ms = int((time.monotonic() - started_at) * 1000)

            if result.status == "completed":
                self.repo.mark_completed(step_state, artifacts=result.artifacts)
                ctx.artifacts[step_name] = result.artifacts or {}
                self.logger.info(
                    "step_completed",
                    extra={
                        "extra": {
                            "event": "step_completed",
                            "job_id": str(job.id),
                            "step": step_name,
                            "component": "pipeline",
                            "duration_ms": duration_ms,
                        }
                    },
                )
                continue

            if result.status == "skipped":
                self.repo.mark_completed(step_state, artifacts=result.artifacts)
                ctx.artifacts[step_name] = result.artifacts or {}
                self.logger.info(
                    "step_skipped",
                    extra={
                        "extra": {
                            "event": "step_skipped",
                            "job_id": str(job.id),
                            "step": step_name,
                            "component": "pipeline",
                            "duration_ms": duration_ms,
                        }
                    },
                )
                continue

            self.repo.mark_failed(step_state, error=result.error or "unknown error")
            self.logger.error(
                "step_failed",
                extra={
                    "extra": {
                        "event": "step_failed",
                        "job_id": str(job.id),
                        "step": step_name,
                        "component": "pipeline",
                        "error": result.error,
                    }
                },
            )
            raise RuntimeError(result.error or "unknown error")
