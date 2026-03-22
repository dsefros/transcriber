import logging

from src.core.pipeline.steps.base import Step, StepResult
from src.core.pipeline.context import PipelineContext
from src.core.transcription.contracts import write_transcription_segments


class TranscriptionStep(Step):
    name = "transcription"
    """
    Шаг пайплайна: транскрипция аудио.

    Core:
    - не привязан к legacy naming
    - не знает, WhisperX или сервис
    - работает ТОЛЬКО через TranscriptionPort
    - пишет каноничный artifact ``output/<job_id>_segments.json``
    """

    def __init__(self):
        self.logger = logging.getLogger("transcription")

    def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.source_type == "json":
            return StepResult(status="skipped")

        if ctx.source_type != "audio":
            return StepResult(
                status="failed",
                error=f"TranscriptionStep supports only audio source, got: {ctx.source_type}",
            )

        segments = ctx.services.transcription.transcribe(str(ctx.source_path))

        try:
            artifacts = write_transcription_segments(segments=segments, job_id=ctx.job_id)
        except ValueError as exc:
            return StepResult(
                status="failed",
                error=f"Invalid transcription artifact: {exc}",
            )

        if artifacts.get("contract_warnings"):
            self.logger.warning(
                "transcription_artifact_normalized",
                extra={
                    "extra": {
                        "event": "transcription_artifact_normalized",
                        "job_id": str(ctx.job_id),
                        "segments_path": artifacts["segments_path"],
                        "warning_count": len(artifacts["contract_warnings"]),
                        "warnings": artifacts["contract_warnings"],
                    }
                },
            )

        return StepResult(status="completed", artifacts=artifacts)
