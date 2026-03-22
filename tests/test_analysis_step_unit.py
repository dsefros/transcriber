from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.pipeline.steps.analysis import AnalysisStep

pytestmark = pytest.mark.unit


class FakePromptRegistry:
    def __init__(self):
        self.calls = []

    def render(self, prompt_id, **variables):
        self.calls.append((prompt_id, variables))
        return f"PROMPT::{variables['transcript']}"


class FakeLLM:
    def __init__(self, result="summary", error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.meta = {"backend": "ollama", "profile": "primary"}

    def generate(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.result


def _ctx(tmp_path, *, artifacts=None, llm=None):
    services = SimpleNamespace(llm=llm or FakeLLM())
    return SimpleNamespace(job_id=uuid4(), artifacts=artifacts or {}, services=services)


def test_analysis_step_requires_transcription_artifacts(tmp_path):
    result = AnalysisStep().run(_ctx(tmp_path))

    assert result.status == "failed"
    assert result.error == "Missing transcription artifacts"


def test_analysis_step_handles_missing_segments_path(tmp_path):
    result = AnalysisStep().run(_ctx(tmp_path, artifacts={"transcription": {"segment_count": 1}}))

    assert result.status == "failed"
    assert result.error == "Missing segments_path in transcription artifacts"


def test_analysis_step_loads_segments_renders_prompt_and_writes_artifact(tmp_path, monkeypatch):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([
        {"speaker": "A", "text": "Hello", "start": 0, "end": 1},
        {"speaker": "B", "text": "World", "start": 1, "end": 2},
    ]), encoding="utf-8")
    llm = FakeLLM(result="final summary")
    ctx = _ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}, llm=llm)
    prompt_registry = FakePromptRegistry()
    step = AnalysisStep()
    step.prompt_registry = prompt_registry

    result = step.run(ctx)

    assert prompt_registry.calls == [("analysis/v1.yaml", {"transcript": "Hello\nWorld"})]
    assert llm.calls == ["PROMPT::Hello\nWorld"]
    assert result.status == "completed"
    analysis_path = result.artifacts["analysis_path"]
    payload = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    assert payload["summary_raw"] == "final summary"
    assert payload["model_backend"] == "ollama"
    assert payload["model_profile"] == "primary"
    assert result.artifacts["prompt_id"] == "analysis.v1"
    assert result.artifacts["segment_count"] == 2


def test_analysis_step_returns_failed_result_for_missing_segments_file(tmp_path):
    segments_path = tmp_path / "missing.json"

    result = AnalysisStep().run(_ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}))

    assert result.status == "failed"
    assert "Segments file not found" in result.error


def test_analysis_step_converts_llm_exception_to_failed_result(tmp_path):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([{"speaker": "A", "text": "Hello", "start": 0, "end": 1}]), encoding="utf-8")
    llm = FakeLLM(error=RuntimeError("offline"))
    ctx = _ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}, llm=llm)
    step = AnalysisStep()
    step.prompt_registry = FakePromptRegistry()

    result = step.run(ctx)

    assert result.status == "failed"
    assert result.error == "LLM inference failed: offline"
