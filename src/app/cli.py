import argparse
import logging
from pathlib import Path

from src.app.preflight import PreflightError, run_preflight
from src.config.env import load_env_file_if_present
from src.core.jobs.models import Job
from src.infrastructure.logging.setup import setup_logging


AUDIO_SOURCE_TYPE = "audio"
JSON_SOURCE_TYPE = "json"


def _detect_source_type(source: Path) -> str:
    return JSON_SOURCE_TYPE if source.suffix.lower() == ".json" else AUDIO_SOURCE_TYPE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Meeting pipeline (job-based)",
        description=(
            "Run the canonical pipeline from an audio file or a transcription "
            "segments JSON artifact. Audio runs transcription + analysis; JSON "
            "runs analysis only."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to an audio file or transcription segments JSON artifact",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    load_env_file_if_present()
    setup_logging(level="INFO")

    logging.getLogger("bootstrap").info(
        "logging_initialized",
        extra={
            "extra": {
                "event": "logging_initialized",
                "component": "cli",
            }
        },
    )

    source_type = _detect_source_type(args.source)

    try:
        preflight_summary = run_preflight(args.source, source_type=source_type)
    except PreflightError as exc:
        parser.exit(status=2, message=f"Preflight failed: {exc}\n")

    logging.getLogger("bootstrap").info(
        "preflight_completed",
        extra={"extra": {"event": "preflight_completed", **preflight_summary}},
    )

    from src.worker import Worker

    job = Job(
        source_type=source_type,
        source_path=str(args.source),
    )

    worker = Worker()
    try:
        result = worker.submit(job)
    finally:
        worker.close()

    print("\n=== JOB RESULT ===")
    print(f"id: {result.id}")
    print(f"status: {result.status}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
