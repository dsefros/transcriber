# Developer workflow

## Local validation
- `scripts/checks/pre_merge_checklist.sh`
- `scripts/checks/local_build_smoke.sh`

## Deploy-bundle rehearsal from repo
Repo wrappers still exist under `scripts/prod/*`, but the primary runtime contract is now in `deploy/*`.

To rehearse bundle behavior locally without server git clone assumptions:
```bash
export TRANSCRIBER_HOME=/workspace/transcriber/deploy
cp deploy/prod.env.example deploy/prod.env
# edit deploy/prod.env + provide deploy/models.yaml
./deploy/install.sh prod-<sha>
./deploy/run_job.sh /data/input/segments.sample.json
./deploy/rollback.sh
```

## GHCR dev artifact checks
- `scripts/dev/redeploy_dev_from_ghcr.sh`
- `scripts/dev/redeploy_dev_local.sh`
