#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${TRANSCRIBER_HOME:-/opt/transcriber}"
ENV_FILE="$BASE_DIR/prod.env"
COMPOSE_FILE="$BASE_DIR/docker-compose.yml"
RUN_SAMPLE="${RUN_SAMPLE_JOB:-0}"

echo "[validate] base_dir=$BASE_DIR"
echo "[validate] runtime doctor"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm --no-deps \
  --entrypoint python transcriber_job -m src.app.runtime_doctor --json --check-db-connection > "$BASE_DIR/state/runtime_doctor.json"

echo "[validate] CLI contract"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm --no-deps transcriber_job --help > "$BASE_DIR/state/cli_help.txt"

if [[ "$RUN_SAMPLE" == "1" ]]; then
  echo "[validate] running representative sample job"
  "$BASE_DIR/run_job.sh" /data/input/segments.sample.json
fi

echo "[validate] evidence: $BASE_DIR/state/runtime_doctor.json and $BASE_DIR/state/cli_help.txt"
