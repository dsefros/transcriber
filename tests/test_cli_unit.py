from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.app import cli
from src.app.preflight import PreflightError
from src.core.jobs.models import Job

pytestmark = pytest.mark.unit


def test_detect_source_type_uses_json_extension_only():
    assert cli._detect_source_type(Path("segments.json")) == "json"
    assert cli._detect_source_type(Path("segments.JSON")) == "json"
    assert cli._detect_source_type(Path("meeting.wav")) == "audio"
    assert cli._detect_source_type(Path("archive.jsonl")) == "audio"


def test_parser_accepts_required_source_without_json_flag():
    parser = cli.build_parser()

    args = parser.parse_args(["meeting.wav"])

    assert args.source == Path("meeting.wav")
    assert not hasattr(args, "json")


def test_main_loads_env_before_preflight_and_submits_audio_job(monkeypatch, capsys):
    calls: list[str] = []
    fake_worker = Mock()
    fake_result = Job(source_type="audio", source_path="meeting.wav")
    fake_result.status = "completed"
    fake_worker.submit.return_value = fake_result

    monkeypatch.setattr(cli, "build_parser", lambda: _parser_for(Path("meeting.wav")))
    monkeypatch.setattr(cli, "load_env_file_if_present", lambda: calls.append("env"))
    monkeypatch.setattr(cli, "setup_logging", lambda level: calls.append(f"logging:{level}"))

    def fake_preflight(source: Path, *, source_type: str):
        calls.append(f"preflight:{source_type}:{source}")
        return {"ok": "yes"}

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)
    monkeypatch.setitem(__import__("sys").modules, "src.worker", type("W", (), {"Worker": lambda: fake_worker}))

    cli.main()

    submitted_job = fake_worker.submit.call_args.args[0]
    assert calls[:2] == ["env", "logging:INFO"]
    assert calls[2] == "preflight:audio:meeting.wav"
    assert submitted_job.source_type == "audio"
    assert submitted_job.source_path == "meeting.wav"
    fake_worker.close.assert_called_once_with()
    assert "status: completed" in capsys.readouterr().out


def test_main_uses_json_extension_source_type_and_closes_worker_on_failure(monkeypatch):
    fake_worker = Mock()
    fake_worker.submit.side_effect = RuntimeError("boom")

    monkeypatch.setattr(cli, "build_parser", lambda: _parser_for(Path("input.json")))
    monkeypatch.setattr(cli, "load_env_file_if_present", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda level: None)
    monkeypatch.setattr(cli, "run_preflight", lambda source, *, source_type: {"source_type": source_type})
    monkeypatch.setitem(__import__("sys").modules, "src.worker", type("W", (), {"Worker": lambda: fake_worker}))

    with pytest.raises(RuntimeError, match="boom"):
        cli.main()

    submitted_job = fake_worker.submit.call_args.args[0]
    assert submitted_job.source_type == "json"
    fake_worker.close.assert_called_once_with()


def test_main_does_not_create_worker_if_preflight_fails(monkeypatch):
    worker_factory = Mock(side_effect=AssertionError("worker should not be constructed"))

    monkeypatch.setattr(cli, "build_parser", lambda: _parser_for(Path("missing.wav")))
    monkeypatch.setattr(cli, "load_env_file_if_present", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda level: None)
    monkeypatch.setattr(cli, "run_preflight", Mock(side_effect=PreflightError("nope")))
    monkeypatch.setitem(__import__("sys").modules, "src.worker", type("W", (), {"Worker": worker_factory}))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    worker_factory.assert_not_called()



def _parser_for(source: Path):
    class Parser:
        def parse_args(self):
            return type("Args", (), {"source": source})

        def exit(self, *, status: int, message: str):
            raise SystemExit(status)

    return Parser()
