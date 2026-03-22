import json
import logging
import time
from pathlib import Path
from datetime import datetime

from src.core.pipeline.steps.base import Step, StepResult
from src.prompts.registry import PromptRegistry
from src.contracts.analysis_result import AnalysisResult
from src.core.transcription.contracts import load_transcription_segments


class AnalysisStep(Step):
    name = "analysis"

    PROMPT_PATH = "analysis/v1.yaml"

    def __init__(self):
        self.prompt_registry = PromptRegistry()
        self.logger = logging.getLogger("analysis")

    def run(self, ctx) -> StepResult:
        job_id = str(ctx.job_id)
        t0 = time.monotonic()

        transcription = ctx.artifacts.get("transcription")
        if not transcription:
            return StepResult(
                status="failed",
                error="Missing transcription artifacts",
            )

        segments_path = transcription.get("segments_path")
        if not segments_path:
            return StepResult(
                status="failed",
                error="Missing segments_path in transcription artifacts",
            )

        try:
            segments = load_transcription_segments(segments_path)
        except FileNotFoundError as exc:
            return StepResult(
                status="failed",
                error=str(exc),
            )
        except ValueError as exc:
            return StepResult(
                status="failed",
                error=f"Invalid transcription artifact: {exc}",
            )

        transcript = "\n".join(segment["text"] for segment in segments).strip()
        if not transcript:
            return StepResult(
                status="failed",
                error="Empty transcription text",
            )

        prompt = self.prompt_registry.render(
            self.PROMPT_PATH,
            transcript=transcript,
        )

        try:
            llm_response = ctx.services.llm.generate(prompt)
        except Exception as e:
            return StepResult(
                status="failed",
                error=f"LLM inference failed: {e}",
            )

        meta = ctx.services.llm.meta

        result = AnalysisResult(
            prompt_id="analysis.v1",
            generated_at=datetime.utcnow(),
            summary_raw=llm_response,
            segment_count=len(segments),
            model_backend=meta["backend"],
            model_profile=meta["profile"],
        )

        output_path = Path(segments_path).with_name(
            Path(segments_path).stem.replace("_segments", "_analysis") + ".json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                result.__dict__,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        total_ms = int((time.monotonic() - t0) * 1000)

        self.logger.info(
            "analysis_completed",
            extra={
                "extra": {
                    "event": "analysis_completed",
                    "job_id": job_id,
                    "analysis_path": str(output_path),
                    "total_ms": total_ms,
                }
            },
        )

        return StepResult(
            status="completed",
            artifacts={
                "analysis_path": str(output_path),
                "prompt_id": result.prompt_id,
                "segment_count": result.segment_count,
            },
        )
