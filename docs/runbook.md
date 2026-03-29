# Operations runbook (server-bundle model)

This runtime is deployed from a **small server bundle**, not from git checkout.
Images are built in GitHub Actions and pushed to GHCR.

## Server runtime bundle files
Copy these files to `/opt/transcriber` on server:

- `docker-compose.yml`
- `prod.env.example` (rename to `prod.env`)
- `install.sh`
- `run_job.sh`
- `rollback.sh`
- `validate.sh`
- `samples/segments.sample.json`

## One-time manual preparation
```bash
cd /opt/transcriber
cp prod.env.example prod.env
# edit prod.env (IMAGE_REPOSITORY, IMAGE_NAME, DB creds, DATABASE_URL, etc.)
# place production models.yaml at /opt/transcriber/models.yaml
mkdir -p /opt/transcriber/runtime/input /opt/transcriber/runtime/output /opt/transcriber/runtime/logs /opt/transcriber/runtime/postgres
cp /opt/transcriber/samples/segments.sample.json /opt/transcriber/runtime/input/segments.sample.json
```

## Install/update one version
```bash
cd /opt/transcriber
./install.sh prod-<sha>
```

## Run one job
```bash
cd /opt/transcriber
./run_job.sh /data/input/<file>.json
```

The script validates fresh artifact creation and prints exact output artifact path.

## Roll back
```bash
cd /opt/transcriber
./rollback.sh
```

## Optional deeper validation
```bash
cd /opt/transcriber
RUN_SAMPLE_JOB=1 ./validate.sh
```

## What is intentionally manual
- Copying the deploy bundle to server.
- Setting `prod.env` values and `models.yaml`.
- Running install/update/rollback commands on the server host.
