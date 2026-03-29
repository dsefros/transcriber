# Runtime contract

## Architecture
This repository is a **CLI/job pipeline**, not an HTTP service.
No `/health` or `/ready` endpoints are part of runtime design.

## Production model
- Server has a fixed bundle directory (default `/opt/transcriber`).
- Server does not clone repo, does not build images, and does not run pip/venv setup.
- Images are built in GitHub Actions and pulled from GHCR by exact tag.
- Resident dependency: Postgres.
- Runtime jobs execute as one-shot containers via `run_job.sh`.

## Deployment contract
- Install/update command: `install.sh <tag>`
- Run job command: `run_job.sh <container-source-path>`
- Rollback command: `rollback.sh`
- Runtime validation command: `validate.sh` (with optional sample run)

## Validation guarantees
- `validate.sh` checks CLI/runtime doctor in deployed image.
- `run_job.sh` enforces fresh artifact validation (new artifact diff or marker-based freshness) and validates artifact schema keys.
