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

        # порядок выполнения pipeline
        from src.core.pipeline.steps.transcription import TranscriptionStep
        from src.core.pipeline.steps.analysis import AnalysisStep

        self.steps = [
            TranscriptionStep(),
            AnalysisStep(),
        ]

    def run(self, job, ctx):
        """
        Execute pipeline for a given job and context.
        """

        for step in self.steps:
            step_name = step.name

            # --- фиксируем текущий шаг job ---
            job.current_step = step_name
            ctx.services  # just to make intent explicit

            # --- лог: шаг стартовал ---
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

            step_state = self.repo.create_if_not_exists(
                job_id=job.id,
                step_name=step_name,
            )

            # если шаг уже завершён — просто прокидываем artifacts
            if step_state.status == StepStatus.COMPLETED:
                if step_state.artifacts:
                    ctx.artifacts[step_name] = step_state.artifacts
                continue

            # RUNNING — инвариантная ошибка
            if step_state.status == StepStatus.RUNNING:
                raise RuntimeError(
                    f"Step '{step_name}' already RUNNING for job {job.id}"
                )

            self.repo.mark_running(step_state)

            started_at = time.monotonic()

            try:
                result = step.run(ctx)
                duration_ms = int((time.monotonic() - started_at) * 1000)

                if result.status == "completed":
                    self.repo.mark_completed(
                        step_state,
                        artifacts=result.artifacts,
                    )

                    ctx.artifacts[step_name] = result.artifacts or {}

                    # --- лог: шаг завершён ---
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

                else:
                    self.repo.mark_failed(
                        step_state,
                        error=result.error or "unknown error",
                    )

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
                    return

            except Exception as e:
                duration_ms = int((time.monotonic() - started_at) * 1000)

                self.repo.mark_failed(step_state, str(e))
