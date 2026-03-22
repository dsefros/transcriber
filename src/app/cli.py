import argparse
import logging
from pathlib import Path

from src.app.preflight import PreflightError, run_preflight
from src.core.jobs.models import Job
from src.infrastructure.logging.setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Meeting pipeline (job-based)")
    parser.add_argument("source", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Logging stays available even when preflight fails early.
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

    source_type = "json" if args.json else "audio"

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
    result = worker.submit(job)

    print("\n=== JOB RESULT ===")
    print(f"id: {result.id}")
    print(f"status: {result.status}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
