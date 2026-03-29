# Developer workflow

## Local setup
1. `cp models.yaml.example models.yaml`
2. Install deps: `pip install -r requirements/runtime.txt -r requirements/dev.txt`
3. Export DB env for runtime checks: `export DATABASE_URL=postgresql://...`

## Validation modes
- Full pre-merge checklist:
  - `scripts/checks/pre_merge_checklist.sh`
- Local working-tree image smoke only:
  - `scripts/checks/local_build_smoke.sh`
- Local build/redeploy helper:
  - `scripts/dev/redeploy_dev_local.sh`
- Redeploy from published GHCR dev artifact:
  - `export GHCR_IMAGE=ghcr.io/<owner>/<repo>`
  - `scripts/dev/redeploy_dev_from_ghcr.sh`

Optional deep runtime execution (requires live DB + model backend):
- `ENABLE_FULL_JOB_SMOKE=1 scripts/dev/redeploy_dev_local.sh`
- `ENABLE_FULL_JOB_SMOKE=1 scripts/dev/redeploy_dev_from_ghcr.sh`

## Production flow rehearsal
On a Docker-capable environment with GHCR access, use:
```bash
TAG_A=prod-<old> TAG_B=prod-<new> scripts/checks/prod_operational_proof.sh
```
This rehearses deploy/deploy/validate/run/rollback using the same operational scripts as production.
