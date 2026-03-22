from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from src.core.jobs.models import Job, JobStatus
from src.core.jobs.postgres_repository import PostgresJobRepository

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
        if "INSERT INTO jobs" in sql:
            self.rows[params["id"]] = dict(params, updated_at=params["updated_at"])
            return _Result(None)
        if "UPDATE jobs" in sql:
            row = self.rows[params["id"]]
            row.update(params)
            return _Result(None)
        if "SELECT * FROM jobs" in sql:
            return _Result(self.rows.get(params["id"]))
        raise AssertionError(sql)


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
    monkeypatch.setattr("src.core.jobs.postgres_repository.init_db", lambda: engine)
    return PostgresJobRepository(), engine


def test_save_update_and_get_round_trip_job(repo):
    repository, _engine = repo
    job = Job(source_type="audio", source_path="meeting.wav")

    repository.save(job)
    job.status = JobStatus.RUNNING
    job.current_step = "transcription"
    job.error = "warn"
    job.attempt = 2
    repository.update(job)
    loaded = repository.get(job.id)

    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.status is JobStatus.RUNNING
    assert loaded.current_step == "transcription"
    assert loaded.error == "warn"
    assert loaded.attempt == 2
