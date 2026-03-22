from __future__ import annotations

import json
from uuid import uuid4

import pytest

from src.core.transcription.contracts import load_transcription_segments, write_transcription_segments

pytestmark = pytest.mark.unit


SEGMENTS = [{"speaker": "A", "text": "Hello", "start": 0, "end": 1}]


def test_write_and_load_transcription_segments_round_trip(tmp_path):
    artifact = write_transcription_segments(SEGMENTS, job_id=uuid4(), output_dir=tmp_path)

    assert artifact["segment_count"] == 1
    loaded = load_transcription_segments(artifact["segments_path"])
    assert loaded == SEGMENTS


def test_load_transcription_segments_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Segments file not found"):
        load_transcription_segments(tmp_path / "missing.json")


def test_load_transcription_segments_malformed_json_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_transcription_segments(path)
