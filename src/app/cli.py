import argparse
from pathlib import Path

from src.worker import Worker
from src.core.jobs.models import Job



def main():
    parser = argparse.ArgumentParser("Meeting pipeline (job-based)")
    parser.add_argument("source", type=Path)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    job = Job(
        source_type="json" if args.json else "audio",
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
