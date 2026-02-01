from pathlib import Path
import json

from src.pipeline_v2.steps.base import Step, StepResult
from src.legacy.pipeline.main import transcribe_and_diarize


class TranscriptionStep(Step):
    name = "transcription"

    def run(self, ctx):
        segments = transcribe_and_diarize(
            audio_path=str(ctx.source_path)
        )

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)

        segments_path = out_dir / f"{ctx.source_hash}_segments.json"
        segments_path.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return StepResult(
            status="completed",
            artifacts={
                "segments_path": str(segments_path),
                "segment_count": len(segments),
            },
        )
