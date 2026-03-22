from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.jobs.models import Job, JobStatus, StepStatus
from src.core.jobs.runner import JobRunner
from src.core.pipeline.steps.base import StepResult

pytestmark = pytest.mark.unit


class Repo:
    def __init__(self):
        self.updated = []

    def update(self, job):
        self.updated.append((job.status, job.error, job.current_step))


class FakeStepState:
    def __init__(self, step_name, status=StepStatus.PENDING, artifacts=None):
        self.id = step_name
        self.job_id = None
        self.step_name = step_name
        self.status = status
        self.attempt = 0
        self.artifacts = artifacts
        self.error = None


class FakeStepRepo:
    def __init__(self, states):
        self.states = states
        self.failed = []
        self.completed = []

    def create_if_not_exists(self, job_id, step_name):
        return self.states[step_name]

    def mark_running(self, step_state):
        step_state.status = StepStatus.RUNNING

    def mark_completed(self, step_state, artifacts):
        self.completed.append((step_state.step_name, artifacts))
        step_state.status = StepStatus.COMPLETED
        step_state.artifacts = artifacts

    def mark_failed(self, step_state, error):
        self.failed.append((step_state.step_name, error))
        step_state.status = StepStatus.FAILED
        step_state.error = error


class FakeStep:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def _job(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    return Job(source_type="audio", source_path=str(source))


def _runner(monkeypatch, fake_repo, steps):
    from src.core.pipeline.orchestrator import PipelineOrchestrator

    def orchestrator_factory():
        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator.repo = fake_repo
        orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
        orchestrator.steps = steps
        return orchestrator

    monkeypatch.setattr("src.core.jobs.runner.PipelineOrchestrator", orchestrator_factory)
    return JobRunner(Repo(), SimpleNamespace())


def test_failed_transcription_step_result_fails_job(monkeypatch, tmp_path):
    fake_repo = FakeStepRepo({
        "transcription": FakeStepState("transcription"),
        "analysis": FakeStepState("analysis"),
    })
    runner = _runner(monkeypatch, fake_repo, [
        FakeStep("transcription", StepResult(status="failed", error="bad audio")),
        FakeStep("analysis", StepResult(status="completed", artifacts={"analysis_path": "unused"})),
    ])
    job = _job(tmp_path)

    runner.run(job)

    assert fake_repo.failed == [("transcription", "bad audio")]
    assert job.status is JobStatus.FAILED
    assert job.error == "bad audio"


def test_failed_analysis_step_result_fails_job(monkeypatch, tmp_path):
    fake_repo = FakeStepRepo({
        "transcription": FakeStepState("transcription", artifacts={"segments_path": "existing.json"}),
        "analysis": FakeStepState("analysis"),
    })
    runner = _runner(monkeypatch, fake_repo, [
        FakeStep("transcription", StepResult(status="completed", artifacts={"segments_path": "existing.json"})),
        FakeStep("analysis", StepResult(status="failed", error="llm failed")),
    ])
    job = _job(tmp_path)

    runner.run(job)

    assert fake_repo.failed == [("analysis", "llm failed")]
    assert job.status is JobStatus.FAILED
    assert job.error == "llm failed"


def test_raised_transcription_exception_fails_job(monkeypatch, tmp_path):
    fake_repo = FakeStepRepo({"transcription": FakeStepState("transcription")})
    runner = _runner(monkeypatch, fake_repo, [FakeStep("transcription", error=RuntimeError("explode"))])
    job = _job(tmp_path)

    runner.run(job)

    assert fake_repo.failed == [("transcription", "explode")]
    assert job.status is JobStatus.FAILED
    assert job.error == "explode"


def test_raised_analysis_exception_fails_job(monkeypatch, tmp_path):
    fake_repo = FakeStepRepo({
        "transcription": FakeStepState("transcription"),
        "analysis": FakeStepState("analysis"),
    })
    runner = _runner(monkeypatch, fake_repo, [
        FakeStep("transcription", StepResult(status="completed", artifacts={"segments_path": "existing.json"})),
        FakeStep("analysis", error=RuntimeError("analysis blew up")),
    ])
    job = _job(tmp_path)

    runner.run(job)

    assert fake_repo.failed == [("analysis", "analysis blew up")]
    assert job.status is JobStatus.FAILED
    assert job.error == "analysis blew up"
