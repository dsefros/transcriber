#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-deployment/prod.env}"
TAG_A="${TAG_A:-}"
TAG_B="${TAG_B:-}"
SOURCE_PATH="${SOURCE_PATH:-/data/input/segments.sample.json}"

if [[ -z "$TAG_A" || -z "$TAG_B" ]]; then
  echo "Set TAG_A and TAG_B for full proof (example: TAG_A=prod-abc1234 TAG_B=prod-def5678)." >&2
  exit 1
fi

echo "[proof] Step 1/5 deploy tag A=$TAG_A"
IMAGE_TAG_OVERRIDE="$TAG_A" ENV_FILE="$ENV_FILE" scripts/prod/deploy_prod.sh

echo "[proof] Step 2/5 deploy tag B=$TAG_B"
IMAGE_TAG_OVERRIDE="$TAG_B" ENV_FILE="$ENV_FILE" scripts/prod/deploy_prod.sh

echo "[proof] Step 3/5 validate runtime + representative job"
scripts/prod/validate_prod_runtime.sh --env-file "$ENV_FILE" --run-job-validation

echo "[proof] Step 4/5 run explicit production job"
scripts/prod/run_prod_job.sh --env-file "$ENV_FILE" --source "$SOURCE_PATH"

echo "[proof] Step 5/5 rollback to previous tag"
ENV_FILE="$ENV_FILE" scripts/prod/rollback_prod.sh

echo "[proof] Completed. Current state files:"
ls -1 deployment/state
cat deployment/state/current_prod_tag
cat deployment/state/previous_prod_tag
