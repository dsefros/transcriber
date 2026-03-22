from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from src.core.jobs.job_step_repository import JobStepRepository
from src.core.jobs.models import JobStep, StepStatus

pytestmark = pytest.mark.unit


class FakeBegin:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self.engine

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self):
        self.rows = {}

    def begin(self):
        return FakeBegin(self)

    def connect(self):
        return FakeBegin(self)

    def execute(self, statement, params):
        sql = str(statement)
        if "INSERT INTO job_steps" in sql:
            self.rows[(params["job_id"], params["step_name"])] = {
                "id": params["id"],
                "job_id": params["job_id"],
                "step_name": params["step_name"],
                "status": params["status"],
                "attempt": 0,
                "artifacts": None,
                "error": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            return _Result(None)
        if "SELECT *\n                FROM job_steps" in sql:
            return _Result(self.rows.get((params["job_id"], params["step_name"])))
        if "attempt = attempt + 1" in sql:
            row = self._row_by_id(params["id"])
            row["status"] = params["status"]
            row["attempt"] += 1
            row["updated_at"] = params["updated_at"]
            return _Result(None)
        if "artifacts = CAST(:artifacts AS JSONB)" in sql:
            row = self._row_by_id(params["id"])
            import json
            row["status"] = params["status"]
            row["artifacts"] = json.loads(params["artifacts"])
            row["error"] = None
            row["updated_at"] = params["updated_at"]
            return _Result(None)
        if "SET status = :status,\n                    error = :error" in sql:
            row = self._row_by_id(params["id"])
            row["status"] = params["status"]
            row["error"] = params["error"]
            row["updated_at"] = params["updated_at"]
            return _Result(None)
        raise AssertionError(sql)

    def _row_by_id(self, row_id):
        return next(row for row in self.rows.values() if row["id"] == row_id)


class _Result:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


@pytest.fixture
def repo(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr("src.core.jobs.job_step_repository.init_db", lambda: engine)
    return JobStepRepository(), engine


def test_create_if_not_exists_creates_then_reuses_existing_row(repo):
    repository, engine = repo
    job_id = uuid4()

    created = repository.create_if_not_exists(job_id, "transcription")
    reused = repository.create_if_not_exists(job_id, "transcription")

    assert created.step_name == "transcription"
    assert reused.id == created.id
    assert len(engine.rows) == 1


def test_mark_running_completed_and_failed_update_state(repo):
    repository, _engine = repo
    job_id = uuid4()
    step = repository.create_if_not_exists(job_id, "analysis")

    repository.mark_running(step)
    running = repository.get(job_id, "analysis")
    repository.mark_completed(running, {})
    completed = repository.get(job_id, "analysis")
    repository.mark_failed(completed, "oops")
    failed = repository.get(job_id, "analysis")

    assert running.status is StepStatus.RUNNING
    assert running.attempt == 1
    assert completed.status is StepStatus.COMPLETED
    assert completed.artifacts == {}
    assert failed.status is StepStatus.FAILED
    assert failed.error == "oops"


def test_mark_completed_rejects_non_json_serializable_artifacts(repo):
    repository, _engine = repo
    job_id = uuid4()
    step = repository.create_if_not_exists(job_id, "analysis")

    with pytest.raises(TypeError):
        repository.mark_completed(step, {"bad": object()})
