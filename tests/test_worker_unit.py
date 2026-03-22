from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.core.jobs.models import Job
from src.worker import Worker

pytestmark = pytest.mark.unit


class FakeServices:
    def __init__(self, *, llm_adapter, transcription):
        self.llm = llm_adapter
        self.transcription = transcription


class FakeRunner:
    def __init__(self, repo, services):
        self.repo = repo
        self.services = services
        self.run = Mock()


def test_worker_constructs_runtime_graph(monkeypatch):
    fake_repo = object()
    fake_transcription = object()
    fake_llm = Mock()
    runner_instances: list[FakeRunner] = []

    monkeypatch.setitem(__import__("sys").modules, "src.core.jobs.postgres_repository", type("M", (), {"PostgresJobRepository": lambda: fake_repo}))
    monkeypatch.setitem(__import__("sys").modules, "src.infrastructure.transcription.whisperx_adapter", type("M", (), {"WhisperXTranscriptionAdapter": lambda: fake_transcription}))
    monkeypatch.setitem(__import__("sys").modules, "src.infrastructure.llm.adapter", type("M", (), {"LLMAdapter": lambda models_config_path: fake_llm}))
    monkeypatch.setattr("src.worker.Services", FakeServices)

    def fake_runner(repo, services):
        runner = FakeRunner(repo, services)
        runner_instances.append(runner)
        return runner

    monkeypatch.setattr("src.worker.JobRunner", fake_runner)

    worker = Worker()

    assert worker.repo is fake_repo
    assert worker.transcription is fake_transcription
    assert worker.llm_adapter is fake_llm
    assert worker.services.llm is fake_llm
    assert worker.services.transcription is fake_transcription
    assert worker.runner is runner_instances[0]


def test_worker_submit_saves_before_runner_and_returns_same_job():
    worker = Worker.__new__(Worker)
    call_order: list[str] = []
    worker.repo = Mock()
    worker.runner = Mock()
    worker.repo.save.side_effect = lambda job: call_order.append("save")
    worker.runner.run.side_effect = lambda job: call_order.append("run")
    job = Job(source_type="audio", source_path="a.wav")

    returned = Worker.submit(worker, job)

    assert returned is job
    assert call_order == ["save", "run"]


def test_worker_close_is_safe_without_backend():
    worker = Worker.__new__(Worker)
    worker.llm_adapter = Mock()

    Worker.close(worker)

    worker.llm_adapter.close.assert_called_once_with()
