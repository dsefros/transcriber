# Operations runbook

## 1) One-time production host bootstrap
```bash
cp deployment/prod.env.example deployment/prod.env
# edit deployment/prod.env and set:
# - IMAGE_REPOSITORY / IMAGE_NAME
# - DATABASE_URL / POSTGRES_PASSWORD
# - models.yaml path and runtime mount paths
mkdir -p runtime/input runtime/output runtime/logs runtime/postgres
cp samples/segments.sample.json runtime/input/segments.sample.json
```

## 2) Deploy exact production tag
```bash
export ENV_FILE=deployment/prod.env
export IMAGE_TAG_OVERRIDE=prod-<shortsha>
scripts/prod/deploy_prod.sh
```

Deploy evidence printed by script:
- env/compose path used
- current tag and target tag
- updated state files (`deployment/state/current_prod_tag`, `deployment/state/previous_prod_tag`)

## 3) Validate deployed runtime and run a production job
```bash
scripts/prod/validate_prod_runtime.sh --env-file deployment/prod.env --run-job-validation
scripts/prod/run_prod_job.sh --env-file deployment/prod.env --source /data/input/<segments>.json
```

`run_prod_job.sh` validates **fresh artifact creation** per run using before/after snapshot + run marker, then validates required output keys.

## 4) Rollback
```bash
export ENV_FILE=deployment/prod.env
scripts/prod/rollback_prod.sh
```

Rollback evidence printed by script:
- rollback target tag
- updated current/previous tag state after rollback
- runtime validation result

## 5) Full deterministic operational proof (A → B → validate/run → rollback)
```bash
export ENV_FILE=deployment/prod.env
export TAG_A=prod-<old>
export TAG_B=prod-<new>
scripts/checks/prod_operational_proof.sh
```

## Scope boundaries
- CI publishes and produces deterministic deploy handoff artifacts.
- Actual production host deployment execution is terminal-driven/manual by design (host credentials/network/secrets are intentionally outside repository scope).
