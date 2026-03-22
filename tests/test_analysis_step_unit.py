from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.pipeline.steps.analysis import AnalysisStep, _prompt_id_from_path, _resolve_prompt_path

pytestmark = pytest.mark.unit


class FakePromptRegistry:
    def __init__(self):
        self.calls = []

    def render(self, prompt_id, **variables):
        self.calls.append((prompt_id, variables))
        return f"PROMPT::{variables['transcript']}"


class FakeModelsConfig:
    def __init__(self, prompt_path="analysis/v1.yaml"):
        self.prompt_path = prompt_path

    def get_default_analysis_prompt(self):
        return self.prompt_path


class FakeLLM:
    def __init__(self, result="summary", error=None, prompt_path="analysis/v1.yaml"):
        self.result = result
        self.error = error
        self.calls = []
        self.meta = {"backend": "ollama", "profile": "primary"}
        self.models_config = FakeModelsConfig(prompt_path=prompt_path)

    def generate(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.result


class LegacyFakeLLM:
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


def _ctx(tmp_path, *, source_type="audio", source_path=None, artifacts=None, llm=None):
    services = SimpleNamespace(llm=llm or FakeLLM())
    if source_path is None:
        source_path = tmp_path / ("segments.json" if source_type == "json" else "meeting.wav")
        if source_type == "json":
            source_path.write_text('[]', encoding='utf-8')
        else:
            source_path.write_bytes(b'audio')
    return SimpleNamespace(
        job_id=uuid4(),
        source_type=source_type,
        source_path=source_path,
        artifacts=artifacts or {},
        services=services,
    )


def test_analysis_step_requires_transcription_artifacts_for_audio(tmp_path):
    result = AnalysisStep().run(_ctx(tmp_path))

    assert result.status == "failed"
    assert result.error == "Invalid transcription artifact: Missing transcription artifacts"


def test_analysis_step_handles_missing_segments_path_for_audio(tmp_path):
    result = AnalysisStep().run(_ctx(tmp_path, artifacts={"transcription": {"segment_count": 1}}))

    assert result.status == "failed"
    assert result.error == "Invalid transcription artifact: Missing segments_path in transcription artifacts"


def test_analysis_step_loads_segments_renders_prompt_and_writes_artifact(tmp_path):
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
    assert payload["prompt_id"] == "analysis.v1"
    assert result.artifacts["prompt_id"] == "analysis.v1"
    assert result.artifacts["prompt_path"] == "analysis/v1.yaml"
    assert result.artifacts["segment_count"] == 2


def test_analysis_step_supports_json_source_without_transcription_artifact(tmp_path, monkeypatch):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([
        {"speaker": "A", "text": "Hello", "start": 0, "end": 1},
        {"speaker": "B", "text": "World", "start": 1, "end": 2},
    ]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM(result="json summary")
    ctx = _ctx(tmp_path, source_type="json", source_path=segments_path, llm=llm)
    step = AnalysisStep()
    step.prompt_registry = FakePromptRegistry()

    result = step.run(ctx)

    assert result.status == "completed"
    analysis_path = Path(result.artifacts["analysis_path"])
    assert analysis_path.resolve() == (tmp_path / "output" / f"{ctx.job_id}_analysis.json").resolve()
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["summary_raw"] == "json summary"
    assert result.artifacts["segment_count"] == 2


def test_analysis_step_returns_failed_result_for_missing_segments_file(tmp_path):
    segments_path = tmp_path / "missing.json"

    result = AnalysisStep().run(_ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}))

    assert result.status == "failed"
    assert "Segments file not found" in result.error


def test_analysis_step_returns_failed_result_for_invalid_json_segments_input(tmp_path):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([{"speaker": "A", "text": "", "start": 0, "end": 1}]), encoding="utf-8")

    result = AnalysisStep().run(_ctx(tmp_path, source_type="json", source_path=segments_path))

    assert result.status == "failed"
    assert result.error == "Invalid transcription artifact: Segment 0 text must be non-empty"


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


def test_analysis_step_uses_prompt_path_from_models_config(tmp_path):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([{"speaker": "A", "text": "Hello", "start": 0, "end": 1}]), encoding="utf-8")
    llm = FakeLLM(prompt_path="analysis/v2.yaml")
    ctx = _ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}, llm=llm)
    prompt_registry = FakePromptRegistry()
    step = AnalysisStep()
    step.prompt_registry = prompt_registry

    result = step.run(ctx)

    assert result.status == "completed"
    assert prompt_registry.calls == [("analysis/v2.yaml", {"transcript": "Hello"})]
    payload = json.loads(Path(result.artifacts["analysis_path"]).read_text(encoding="utf-8"))
    assert payload["prompt_id"] == "analysis.v2"
    assert result.artifacts["prompt_id"] == "analysis.v2"
    assert result.artifacts["prompt_path"] == "analysis/v2.yaml"


@pytest.mark.parametrize(
    ("prompt_path", "expected_prompt_id"),
    [
        ("analysis/v1.yaml", "analysis.v1"),
        ("analysis/v2.yaml", "analysis.v2"),
        ("foo/bar/baz.yaml", "foo.bar.baz"),
    ],
)
def test_prompt_id_from_path(prompt_path, expected_prompt_id):
    assert _prompt_id_from_path(prompt_path) == expected_prompt_id


def test_analysis_step_falls_back_to_legacy_prompt_when_llm_has_no_models_config(tmp_path):
    segments_path = tmp_path / "segments.json"
    segments_path.write_text(json.dumps([{"speaker": "A", "text": "Hello", "start": 0, "end": 1}]), encoding="utf-8")
    llm = LegacyFakeLLM()
    ctx = _ctx(tmp_path, artifacts={"transcription": {"segments_path": str(segments_path)}}, llm=llm)
    prompt_registry = FakePromptRegistry()
    step = AnalysisStep()
    step.prompt_registry = prompt_registry

    result = step.run(ctx)

    assert result.status == "completed"
    assert prompt_registry.calls == [("analysis/v1.yaml", {"transcript": "Hello"})]
    payload = json.loads(Path(result.artifacts["analysis_path"]).read_text(encoding="utf-8"))
    assert payload["prompt_id"] == "analysis.v1"
    assert result.artifacts["prompt_id"] == "analysis.v1"
    assert result.artifacts["prompt_path"] == "analysis/v1.yaml"


def test_resolve_prompt_path_uses_models_config_when_available(tmp_path):
    assert _resolve_prompt_path(_ctx(tmp_path, llm=FakeLLM(prompt_path="analysis/v2.yaml"))) == "analysis/v2.yaml"


def test_resolve_prompt_path_falls_back_when_models_config_is_missing(tmp_path):
    assert _resolve_prompt_path(_ctx(tmp_path, llm=LegacyFakeLLM())) == "analysis/v1.yaml"
