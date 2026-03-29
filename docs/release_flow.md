# Release flow

## Branch model
- `main` = integration/dev
- `prod` = production
- promotion only by PR: `main` -> `prod`

## CI/CD behavior
Workflow `.github/workflows/docker.yml` does:
1. `validate`: tests + image contract check
2. `publish`: build and push GHCR tags
   - on `main`: `dev-latest`, `dev-<shortsha>`
   - on `prod`: `prod-latest`, `prod-<shortsha>`
3. `bundle`: upload `deploy/` as runtime bundle artifact, stamped with published tag

## Important operational boundary
GitHub Actions does **not** deploy on the server.
Server deployment is done by server operators with bundle commands:
- `/opt/transcriber/install.sh prod-<sha>`
- `/opt/transcriber/run_job.sh /data/input/file.json`
- `/opt/transcriber/rollback.sh`
