import json
from pathlib import Path
from datetime import datetime

from src.core.pipeline.steps.base import Step, StepResult
from src.prompts.registry import PromptRegistry
from src.contracts.analysis_result import AnalysisResult


class AnalysisStep(Step):
    name = "analysis"

    PROMPT_PATH = "analysis/v1.yaml"

    def __init__(self):
        self.prompt_registry = PromptRegistry()

    def run(self, ctx) -> StepResult:
        # 1. Get transcription artifacts
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

        segments_path = Path(segments_path)
        if not segments_path.exists():
            return StepResult(
                status="failed",
                error=f"Segments file not found: {segments_path}",
            )

        # 2. Load segments
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        transcript = "\n".join(seg.get("text", "") for seg in segments).strip()
        if not transcript:
            return StepResult(
                status="failed",
                error="Empty transcription text",
            )

        # 3. Render prompt
        prompt = self.prompt_registry.render(
            self.PROMPT_PATH,
            transcript=transcript,
        )

        # 4. LLM inference
        try:
            llm_response = ctx.services.llm.generate(prompt)
        except Exception as e:
            return StepResult(
                status="failed",
                error=f"LLM inference failed: {e}",
            )

        # 5. Read unified metadata
        meta = ctx.services.llm.meta

        # 6. Build result contract
        result = AnalysisResult(
            prompt_id="analysis.v1",
            generated_at=datetime.utcnow(),
            summary_raw=llm_response,
            segment_count=len(segments),
            model_backend=meta["backend"],
            model_profile=meta["profile"],
        )

        # 7. Persist
        output_path = segments_path.with_name(
            segments_path.stem.replace("_segments", "_analysis") + ".json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, indent=2, default=str)

        return StepResult(
            status="completed",
            artifacts={
                "analysis_path": str(output_path),
                "prompt_id": result.prompt_id,
                "segment_count": result.segment_count,
            },
        )
