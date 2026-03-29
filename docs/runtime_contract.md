# Runtime contract

## Detected architecture
This repository is a **job-style CLI pipeline**, not a long-running HTTP API.

Canonical execution path:
`python -m src.app.cli <source>` → `Worker` → `JobRunner` → pipeline steps.

No HTTP server and no `/health` or `/ready` endpoints exist.

## Production execution model (chosen)
This PR uses a **deployed image + resident dependency stack** model:

- Resident in production:
  - PostgreSQL (docker compose service `postgres`)
  - host-mounted runtime directories (`runtime/input`, `runtime/output`, `runtime/logs`, `runtime/postgres`)
  - operator-provided runtime config (`models.yaml` + `deployment/prod.env`)
- Job runtime image is not permanently running; jobs are executed on demand as one-shot containers.
- Deployment means switching the exact GHCR image tag used by compose (`IMAGE_TAG`) and validating runtime contract.
- Rollback means switching back to the previous exact tag with symmetric state updates and running the same validation path.

## Operational validation contract
Because this is job-oriented, validation is command-based:

1. Image starts canonical CLI (`--help`).
2. Runtime doctor executes in the deployed image (`python -m src.app.runtime_doctor --json`).
3. Optional representative sample production job via `scripts/prod/run_prod_job.sh`.
4. `run_prod_job.sh` enforces fresh-artifact validation (new file diff or newer-than-run marker) and validates required output keys.

## What is manual by design
- GitHub Actions publishes image tags and provides deployment handoff metadata.
- Production host execution (`deploy_prod.sh` / `rollback_prod.sh`) remains manual terminal execution because host credentials/network/secrets are intentionally externalized.
