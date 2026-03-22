from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.jobs.models import Job, StepStatus
from src.core.pipeline.steps.base import StepResult
from src.core.pipeline.orchestrator import PipelineOrchestrator

pytestmark = pytest.mark.unit


@dataclass
class StepState:
    id: str
    job_id: object
    step_name: str
    status: StepStatus
    attempt: int = 0
    artifacts: dict | None = None
    error: str | None = None


class FakeRepo:
    def __init__(self, states):
        self.states = states
        self.created = []
        self.running = []
        self.completed = []
        self.failed = []

    def create_if_not_exists(self, job_id, step_name):
        self.created.append(step_name)
        return self.states[step_name]

    def mark_running(self, step_state):
        self.running.append(step_state.step_name)

    def mark_completed(self, step_state, artifacts):
        self.completed.append((step_state.step_name, artifacts))

    def mark_failed(self, step_state, error):
        self.failed.append((step_state.step_name, error))


class FakeStep:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = []

    def run(self, ctx):
        self.calls.append(ctx)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def job_and_ctx():
    job = Job(source_type="audio", source_path="input.wav")
    ctx = SimpleNamespace(artifacts={}, services=object())
    return job, ctx


def test_orchestrator_runs_steps_in_order_and_rehydrates_completed(job_and_ctx):
    job, ctx = job_and_ctx
    transcription_state = StepState("1", job.id, "transcription", StepStatus.COMPLETED, artifacts={"segments_path": "existing.json"})
    analysis_state = StepState("2", job.id, "analysis", StepStatus.PENDING)
    repo = FakeRepo({"transcription": transcription_state, "analysis": analysis_state})
    analysis = FakeStep("analysis", StepResult(status="completed", artifacts={"analysis_path": "out.json"}))

    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.repo = repo
    orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    orchestrator.steps = [FakeStep("transcription"), analysis]

    orchestrator.run(job, ctx)

    assert repo.created == ["transcription", "analysis"]
    assert repo.running == ["analysis"]
    assert repo.completed == [("analysis", {"analysis_path": "out.json"})]
    assert ctx.artifacts["transcription"] == {"segments_path": "existing.json"}
    assert ctx.artifacts["analysis"] == {"analysis_path": "out.json"}
    assert analysis.calls == [ctx]
    assert job.current_step == "analysis"


def test_orchestrator_raises_for_running_step(job_and_ctx):
    job, ctx = job_and_ctx
    repo = FakeRepo({"transcription": StepState("1", job.id, "transcription", StepStatus.RUNNING)})
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.repo = repo
    orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    orchestrator.steps = [FakeStep("transcription")]

    with pytest.raises(RuntimeError, match="already RUNNING"):
        orchestrator.run(job, ctx)


def test_orchestrator_marks_failed_and_returns_early_for_failed_result(job_and_ctx):
    job, ctx = job_and_ctx
    repo = FakeRepo({
        "transcription": StepState("1", job.id, "transcription", StepStatus.PENDING),
        "analysis": StepState("2", job.id, "analysis", StepStatus.PENDING),
    })
    transcription = FakeStep("transcription", StepResult(status="failed", error="bad audio"))
    analysis = FakeStep("analysis", StepResult(status="completed", artifacts={"ignored": True}))
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.repo = repo
    orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    orchestrator.steps = [transcription, analysis]

    orchestrator.run(job, ctx)

    assert repo.running == ["transcription"]
    assert repo.failed == [("transcription", "bad audio")]
    assert analysis.calls == []
    assert job.current_step == "transcription"


def test_orchestrator_marks_failed_and_stops_on_exception(job_and_ctx):
    job, ctx = job_and_ctx
    repo = FakeRepo({"transcription": StepState("1", job.id, "transcription", StepStatus.PENDING)})
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.repo = repo
    orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    orchestrator.steps = [FakeStep("transcription", error=ValueError("boom"))]

    orchestrator.run(job, ctx)

    assert repo.failed == [("transcription", "boom")]
