from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.pipeline.steps.transcription import TranscriptionStep

pytestmark = pytest.mark.unit


class TranscriptionService:
    def __init__(self, segments=None, error=None):
        self.segments = segments
        self.error = error
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        if self.error:
            raise self.error
        return self.segments


def _ctx(tmp_path, *, source_type="audio", segments=None):
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"audio")
    transcription = TranscriptionService(segments=[{"speaker": " Speaker ", "text": " Hello ", "start": 2, "end": 1}] if segments is None else segments)
    services = SimpleNamespace(transcription=transcription)
    return SimpleNamespace(job_id=uuid4(), source_type=source_type, source_path=source, services=services), transcription


def test_transcription_step_rejects_non_audio(tmp_path):
    ctx, _ = _ctx(tmp_path, source_type="json")

    result = TranscriptionStep().run(ctx)

    assert result.status == "failed"
    assert "supports only audio" in result.error


def test_transcription_step_writes_normalized_artifact_and_warns(tmp_path, monkeypatch):
    ctx, transcription = _ctx(tmp_path)
    monkeypatch.chdir(tmp_path)
    warnings = []
    step = TranscriptionStep()
    step.logger = SimpleNamespace(warning=lambda *a, **k: warnings.append(k["extra"]["extra"]))

    result = step.run(ctx)

    assert transcription.calls == [str(ctx.source_path)]
    assert result.status == "completed"
    assert result.artifacts["segment_count"] == 1
    assert result.artifacts["normalized_segment_count"] == 1
    assert warnings[0]["warning_count"] == 1
    stored = json.loads((tmp_path / "output" / f"{ctx.job_id}_segments.json").read_text(encoding="utf-8"))
    assert stored == [{"speaker": "Speaker", "text": "Hello", "start": 2.0, "end": 2.0}]


@pytest.mark.parametrize(
    "segments,error_match",
    [
        ([], "non-empty list"),
        ([{"speaker": "s", "text": "", "start": 0, "end": 1}], "text must be non-empty"),
        ([{"speaker": "s", "text": "x", "start": "a", "end": 1}], "start/end must be numeric"),
    ],
)
def test_transcription_step_validation_failures_return_failed_result(tmp_path, segments, error_match):
    ctx, _ = _ctx(tmp_path, segments=segments)
    
    result = TranscriptionStep().run(ctx)

    assert result.status == "failed"
    assert error_match in result.error
