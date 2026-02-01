from src.jobs.job_step_repository import JobStepRepository
from src.jobs.models import StepStatus

from src.pipeline_v2.steps.transcription import TranscriptionStep
from src.pipeline_v2.steps.analysis import AnalysisStep


class PipelineOrchestrator:
    """
    Orchestrates execution of pipeline steps.
    Control-plane state is stored ONLY in job_steps table.
    """

    def __init__(self):
        self.repo = JobStepRepository()

        # порядок выполнения pipeline
        self.steps = [
            TranscriptionStep(),
            AnalysisStep(),
        ]

    def run(self, job, ctx):
        """
        Execute pipeline for a given job and context.
        """

        for step in self.steps:
            # 1. Получаем или создаём состояние шага
            step_state = self.repo.create_if_not_exists(
                job_id=job.id,
                step_name=step.name,
            )

            # 2. Если шаг уже завершён — просто прокидываем artifacts дальше
            if step_state.status == StepStatus.COMPLETED:
                if step_state.artifacts:
                    ctx.artifacts[step.name] = step_state.artifacts
                continue

            # 3. RUNNING — это инвариантная ошибка
            if step_state.status == StepStatus.RUNNING:
                raise RuntimeError(
                    f"Step '{step.name}' already RUNNING for job {job.id}"
                )

            # 4. Переводим в RUNNING (attempt++)
            self.repo.mark_running(step_state)

            try:
                # 5. Выполняем шаг
                result = step.run(ctx)

                # 6. Обрабатываем результат
                if result.status == "completed":
                    self.repo.mark_completed(
                        step_state,
                        artifacts=result.artifacts,
                    )

                    # 🔑 КЛЮЧЕВОЕ МЕСТО
                    # передаём artifacts следующему шагу
                    ctx.artifacts[step.name] = result.artifacts or {}

                else:
                    self.repo.mark_failed(
                        step_state,
                        error=result.error or "unknown error",
                    )
                    return  # pipeline останавливается

            except Exception as e:
                # 7. Любое исключение = FAILED
                self.repo.mark_failed(step_state, str(e))
                raise
