#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="deployment/prod.env"
RUN_JOB_VALIDATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --run-job-validation)
      RUN_JOB_VALIDATION=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

COMPOSE_FILE="docker-compose.prod.yml"

echo "[validate] env_file=$ENV_FILE compose_file=$COMPOSE_FILE"
echo "[validate] Runtime doctor in target prod image"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm --no-deps \
  --entrypoint python transcriber_job -m src.app.runtime_doctor --json > /tmp/prod_runtime_doctor.json

echo "[validate] CLI contract help in target prod image"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm --no-deps transcriber_job --help >/tmp/prod_cli_help.txt

if [[ "$RUN_JOB_VALIDATION" == "1" ]]; then
  echo "[validate] Running representative production JSON job"
  "$ROOT_DIR/scripts/prod/run_prod_job.sh" --env-file "$ENV_FILE" --source /data/input/segments.sample.json
fi

echo "[validate] Runtime doctor report: /tmp/prod_runtime_doctor.json"
echo "[validate] CLI help output: /tmp/prod_cli_help.txt"
echo "[validate] Production runtime validation completed"
