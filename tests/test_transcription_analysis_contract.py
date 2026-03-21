import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from uuid import uuid4

sys.modules.setdefault("llama_cpp", types.SimpleNamespace(Llama=object))

from src.core.pipeline.context import PipelineContext
from src.core.pipeline.steps.analysis import AnalysisStep
from src.core.pipeline.steps.transcription import TranscriptionStep
from src.core.transcription.contracts import (
    build_segments_artifact_path,
    load_transcription_segments,
    normalize_transcription_segments,
    validate_transcription_segments,
)
from src.prompts.registry import PromptRegistry


class FakeTranscriptionService:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path: str):
        return list(self._segments)


class FakeLLM:
    def __init__(self):
        self.meta = {"backend": "fake", "profile": "test"}
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Summary output"


class FakeServices:
    def __init__(self, llm, transcription):
        self.llm = llm
        self.transcription = transcription


class TranscriptionAnalysisContractTests(unittest.TestCase):
    def test_validate_transcription_segments_accepts_active_shape(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": " Hello ", "start": 0, "end": 1.25},
            {"speaker": "SPEAKER_01", "text": "World", "start": 1.25, "end": 2},
        ]

        validated = validate_transcription_segments(segments)

        self.assertEqual(2, len(validated))
        self.assertEqual("Hello", validated[0]["text"])
        self.assertEqual(2.0, validated[1]["end"])

    def test_normalize_transcription_segments_clamps_inverted_timing_and_warns(self):
        normalized, warnings = normalize_transcription_segments(
            [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Closing note",
                    "start": 2699.08,
                    "end": 2698.43,
                }
            ]
        )

        self.assertEqual(2699.08, normalized[0]["start"])
        self.assertEqual(2699.08, normalized[0]["end"])
        self.assertEqual(1, len(warnings))
        self.assertIn("inverted timing", warnings[0])

    def test_validate_transcription_segments_rejects_missing_required_keys(self):
        with self.assertRaisesRegex(ValueError, "missing required keys: end"):
            validate_transcription_segments(
                [{"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0}]
            )

    def test_transcription_step_writes_canonical_artifact_and_warning_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            audio_path = workdir / "sample.wav"
            audio_path.write_bytes(b"audio")

            ctx = PipelineContext(
                job_id=uuid4(),
                source_type="audio",
                source_path=audio_path,
                services=FakeServices(
                    llm=FakeLLM(),
                    transcription=FakeTranscriptionService(
                        [
                            {
                                "speaker": "SPEAKER_00",
                                "text": "Hello",
                                "start": 2699.08,
                                "end": 2698.43,
                            }
                        ]
                    ),
                ),
            )

            current_dir = Path.cwd()
            try:
                os.chdir(workdir)
                result = TranscriptionStep().run(ctx)
            finally:
                os.chdir(current_dir)

            self.assertEqual("completed", result.status)
            self.assertEqual(1, result.artifacts["segment_count"])
            self.assertEqual(1, result.artifacts["normalized_segment_count"])
            self.assertEqual(1, len(result.artifacts["contract_warnings"]))
            expected_path = build_segments_artifact_path(ctx.job_id)
            self.assertEqual(str(expected_path), result.artifacts["segments_path"])
            self.assertEqual(
                [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Hello",
                        "start": 2699.08,
                        "end": 2699.08,
                    }
                ],
                json.loads((workdir / expected_path).read_text(encoding="utf-8")),
            )

    def test_analysis_step_consumes_transcription_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            segments_path = workdir / "job_segments.json"
            segments_path.write_text(
                json.dumps(
                    [
                        {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
                        {"speaker": "SPEAKER_01", "text": "World", "start": 1.0, "end": 2.0},
                    ]
                ),
                encoding="utf-8",
            )
            audio_path = workdir / "sample.wav"
            audio_path.write_bytes(b"audio")
            llm = FakeLLM()
            ctx = PipelineContext(
                job_id=uuid4(),
                source_type="audio",
                source_path=audio_path,
                services=FakeServices(llm=llm, transcription=FakeTranscriptionService([])),
                artifacts={"transcription": {"segments_path": str(segments_path), "segment_count": 2}},
            )
            step = AnalysisStep()
            step.prompt_registry = PromptRegistry(base_dir=Path.cwd() / "src/prompts")

            result = step.run(ctx)

            self.assertEqual("completed", result.status)
            self.assertTrue(Path(result.artifacts["analysis_path"]).exists())
            self.assertIn("Hello\nWorld", llm.prompts[0])

    def test_transcription_to_analysis_boundary_smoke_tolerates_inverted_timing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            audio_path = workdir / "sample.wav"
            audio_path.write_bytes(b"audio")
            llm = FakeLLM()
            ctx = PipelineContext(
                job_id=uuid4(),
                source_type="audio",
                source_path=audio_path,
                services=FakeServices(
                    llm=llm,
                    transcription=FakeTranscriptionService(
                        [
                            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
                            {
                                "speaker": "SPEAKER_01",
                                "text": "there",
                                "start": 2699.08,
                                "end": 2698.43,
                            },
                        ]
                    ),
                ),
            )
            step = AnalysisStep()
            step.prompt_registry = PromptRegistry(base_dir=Path.cwd() / "src/prompts")

            current_dir = Path.cwd()
            try:
                os.chdir(workdir)
                transcription_result = TranscriptionStep().run(ctx)
                ctx.artifacts["transcription"] = transcription_result.artifacts
                analysis_result = step.run(ctx)
            finally:
                os.chdir(current_dir)

            self.assertEqual("completed", transcription_result.status)
            self.assertEqual("completed", analysis_result.status)
            analysis_path = workdir / analysis_result.artifacts["analysis_path"]
            segments_path = workdir / transcription_result.artifacts["segments_path"]

            self.assertEqual(
                2,
                json.loads(analysis_path.read_text(encoding="utf-8"))["segment_count"],
            )
            self.assertEqual(
                [
                    {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
                    {
                        "speaker": "SPEAKER_01",
                        "text": "there",
                        "start": 2699.08,
                        "end": 2699.08,
                    },
                ],
                load_transcription_segments(segments_path),
            )
            self.assertEqual(1, transcription_result.artifacts["normalized_segment_count"])

    def test_validate_transcription_segments_still_rejects_unusable_artifact(self):
        with self.assertRaisesRegex(ValueError, "speaker must be non-empty"):
            validate_transcription_segments(
                [{"speaker": " ", "text": "Hello", "start": 0.0, "end": 1.0}]
            )


if __name__ == "__main__":
    unittest.main()
