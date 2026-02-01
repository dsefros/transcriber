from src.core.pipeline.steps.base import Step, StepResult
from src.core.pipeline.context import PipelineContext
import json
from pathlib import Path


class TranscriptionStep(Step):
    name = "transcription"
    """
    Шаг пайплайна: транскрипция аудио.

    Core:
    - не знает, legacy это или нет
    - не знает, WhisperX или сервис
    - работает ТОЛЬКО через TranscriptionPort
    """

    def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.source_type != "audio":
            return StepResult(
                status="failed",
                error=f"TranscriptionStep supports only audio source, got: {ctx.source_type}",
            )

        # 🔑 Вызов через порт
        segments = ctx.services.transcription.transcribe(
            str(ctx.source_path)
        )

        if not segments:
            return StepResult(
                status="failed",
                error="Empty transcription result",
            )

        # 📦 Сохраняем сегменты в файл (канонично для pipeline)
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        segments_path = output_dir / f"{ctx.job_id}_segments.json"

        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        return StepResult(
            status="completed",
            artifacts={
                "segments_path": str(segments_path),
                "segment_count": len(segments),
            },
        )
