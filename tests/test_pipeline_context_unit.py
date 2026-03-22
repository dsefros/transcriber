from __future__ import annotations

import hashlib

import pytest

from src.core.pipeline.context import PipelineContext

pytestmark = pytest.mark.unit


class StubServices:
    pass


def test_pipeline_context_stores_core_fields_and_initializes_artifacts(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")

    ctx = PipelineContext(job_id="job-1", source_type="json", source_path=source, services=StubServices())

    assert ctx.job_id == "job-1"
    assert ctx.source_type == "json"
    assert ctx.source_path == source
    assert ctx.artifacts == {}
    assert ctx.source_hash == hashlib.sha256(str(source).encode()).hexdigest()


def test_pipeline_context_hashes_audio_content_deterministically(tmp_path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    first.write_bytes(b"same")
    second.write_bytes(b"different")

    ctx1 = PipelineContext(job_id="job-1", source_type="audio", source_path=first, services=StubServices())
    ctx2 = PipelineContext(job_id="job-2", source_type="audio", source_path=first, services=StubServices())
    ctx3 = PipelineContext(job_id="job-3", source_type="audio", source_path=second, services=StubServices())

    assert ctx1.source_hash == ctx2.source_hash
    assert ctx1.source_hash != ctx3.source_hash


def test_pipeline_context_missing_audio_file_raises(tmp_path):
    missing = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        PipelineContext(job_id="job-1", source_type="audio", source_path=missing, services=StubServices())
