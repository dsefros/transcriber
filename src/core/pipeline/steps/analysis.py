import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.contracts.analysis_result import AnalysisResult
from src.core.pipeline.steps.base import Step, StepResult
from src.core.transcription.contracts import build_segments_artifact_path, load_transcription_segments
from src.prompts.registry import PromptRegistry


def _prompt_id_from_path(prompt_path: str) -> str:
    if prompt_path.endswith(".yaml"):
        prompt_path = prompt_path[:-5]
    return prompt_path.replace("/", ".")


def _resolve_prompt_path(ctx) -> str:
    models_config = getattr(ctx.services.llm, "models_config", None)
    if models_config and hasattr(models_config, "get_default_analysis_prompt"):
        return models_config.get_default_analysis_prompt()
    return "analysis/v1.yaml"


def _resolve_segments_path(ctx) -> Path:
    if ctx.source_type == "json":
        return Path(ctx.source_path)

    transcription = ctx.artifacts.get("transcription")
    if not transcription:
        raise ValueError("Missing transcription artifacts")

    segments_path = transcription.get("segments_path")
    if not segments_path:
        raise ValueError("Missing segments_path in transcription artifacts")

    return Path(segments_path)


def _build_analysis_path(ctx, segments_path: Path) -> Path:
    if ctx.source_type == "json":
        return build_segments_artifact_path(ctx.job_id).with_name(f"{ctx.job_id}_analysis.json")

    return segments_path.with_name(
        segments_path.stem.replace("_segments", "_analysis") + ".json"
    )


class AnalysisStep(Step):
    name = "analysis"

    def __init__(self):
        self.prompt_registry = PromptRegistry()
        self.logger = logging.getLogger("analysis")

    def run(self, ctx) -> StepResult:
        job_id = str(ctx.job_id)
        t0 = time.monotonic()

        try:
            segments_path = _resolve_segments_path(ctx)
            segments = load_transcription_segments(segments_path)
        except FileNotFoundError as exc:
            return StepResult(status="failed", error=str(exc))
        except ValueError as exc:
            return StepResult(
                status="failed",
                error=f"Invalid transcription artifact: {exc}",
            )

        transcript = "\n".join(segment["text"] for segment in segments).strip()
        if not transcript:
            return StepResult(status="failed", error="Empty transcription text")

        prompt_path = _resolve_prompt_path(ctx)
        prompt_id = _prompt_id_from_path(prompt_path)
        prompt = self.prompt_registry.render(prompt_path, transcript=transcript)

        try:
            llm_response = ctx.services.llm.generate(prompt)
        except Exception as exc:
            return StepResult(
                status="failed",
                error=f"LLM inference failed: {exc}",
            )

        meta = ctx.services.llm.meta
        result = AnalysisResult(
            prompt_id=prompt_id,
            generated_at=datetime.utcnow(),
            summary_raw=llm_response,
            segment_count=len(segments),
            model_backend=meta["backend"],
            model_profile=meta["profile"],
        )

        output_path = _build_analysis_path(ctx, segments_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                result.__dict__,
                file_obj,
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
                "prompt_path": prompt_path,
                "segment_count": result.segment_count,
            },
        )
