# Release flow

## Branching model (enforced)
- `main`: integration/development branch.
- `prod`: production branch.
- Feature branches start from `main`.
- Promotion is only `main` → `prod` via PR.

## CI/CD responsibilities
Workflow: `.github/workflows/docker.yml`

1. `validate` job (main + prod)
   - tests
   - docker build
   - CLI contract check
2. `publish` job (main + prod)
   - pushes GHCR image tags:
     - `dev-latest`, `dev-<shortsha>` on `main`
     - `prod-latest`, `prod-<shortsha>` on `prod`
3. `deploy_handoff_prod` job (`prod` only)
   - publishes deterministic handoff artifact with exact `prod-<shortsha>` tag
   - does **not** claim remote auto-deploy

## Production promotion and deployment
1. Merge feature into `main`.
2. Validate + publish dev tags.
3. Promote `main` to `prod` via PR merge.
4. Validate + publish prod tags.
5. On production host, apply exact published tag:
   - `export IMAGE_TAG_OVERRIDE=prod-<shortsha>`
   - `scripts/prod/deploy_prod.sh`
6. Validate runtime and sample job:
   - `scripts/prod/validate_prod_runtime.sh --run-job-validation`
7. Roll back with:
   - `scripts/prod/rollback_prod.sh`
