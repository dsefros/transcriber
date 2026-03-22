from __future__ import annotations

from pathlib import Path

import pytest

from src.core.jobs.models import Job, JobStatus
from src.core.jobs.runner import JobRunner

pytestmark = pytest.mark.unit


class Repo:
    def __init__(self):
        self.updated = []

    def update(self, job):
        self.updated.append((job.status, job.current_step, job.error))


class StubServices:
    pass


class FakeOrchestrator:
    def __init__(self, *, side_effect=None):
        self.calls = []
        self.side_effect = side_effect

    def run(self, job, ctx):
        self.calls.append((job, ctx))
        if self.side_effect:
            raise self.side_effect


def test_job_defaults_are_canonical():
    job = Job()

    assert job.status is JobStatus.PENDING
    assert job.current_step is None
    assert job.error is None
    assert job.attempt == 0


def test_runner_sets_running_then_completed_and_clears_error(monkeypatch, tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    repo = Repo()
    services = StubServices()
    orchestrator = FakeOrchestrator()
    monkeypatch.setattr("src.core.jobs.runner.PipelineOrchestrator", lambda: orchestrator)

    runner = JobRunner(repo, services)
    job = Job(source_type="audio", source_path=str(source), error="old", current_step="analysis")

    runner.run(job)

    assert repo.updated[0][0] is JobStatus.RUNNING
    assert repo.updated[0][1] is None
    assert repo.updated[0][2] is None
    assert repo.updated[-1][0] is JobStatus.COMPLETED
    ctx = orchestrator.calls[0][1]
    assert ctx.job_id == job.id
    assert ctx.source_type == "audio"
    assert ctx.source_path == Path(source)
    assert ctx.services is services


def test_runner_marks_failed_and_stores_error_on_exception(monkeypatch, tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    repo = Repo()
    services = StubServices()
    monkeypatch.setattr("src.core.jobs.runner.PipelineOrchestrator", lambda: FakeOrchestrator(side_effect=RuntimeError("explode")))

    runner = JobRunner(repo, services)
    job = Job(source_type="audio", source_path=str(source))

    runner.run(job)

    assert job.status is JobStatus.FAILED
    assert job.error == "explode"
    assert repo.updated[-1] == (JobStatus.FAILED, None, "explode")
