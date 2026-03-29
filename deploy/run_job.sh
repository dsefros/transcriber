#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <container-source-path>  (example: $0 /data/input/file.json)" >&2
  exit 1
fi

SOURCE_PATH="$1"
BASE_DIR="${TRANSCRIBER_HOME:-/opt/transcriber}"
ENV_FILE="$BASE_DIR/prod.env"
COMPOSE_FILE="$BASE_DIR/docker-compose.yml"
OUTPUT_DIR="$BASE_DIR/runtime/output"

mkdir -p "$OUTPUT_DIR" "$BASE_DIR/state"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MARKER_FILE="$OUTPUT_DIR/.run_${RUN_ID}.marker"
BEFORE_LIST="/tmp/transcriber_before_${RUN_ID}.txt"
AFTER_LIST="/tmp/transcriber_after_${RUN_ID}.txt"
NEW_LIST="/tmp/transcriber_new_${RUN_ID}.txt"

echo "[run] base_dir=$BASE_DIR run_id=$RUN_ID source=$SOURCE_PATH"
find "$OUTPUT_DIR" -maxdepth 1 -name '*_analysis.json' -print | sort > "$BEFORE_LIST"
touch "$MARKER_FILE"

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm transcriber_job "$SOURCE_PATH"

find "$OUTPUT_DIR" -maxdepth 1 -name '*_analysis.json' -print | sort > "$AFTER_LIST"
comm -13 "$BEFORE_LIST" "$AFTER_LIST" > "$NEW_LIST" || true

ARTIFACT_PATH="$(tail -n 1 "$NEW_LIST" || true)"
if [[ -z "$ARTIFACT_PATH" ]]; then
  ARTIFACT_PATH="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*_analysis.json' -newer "$MARKER_FILE" -print | sort | tail -n 1 || true)"
fi

if [[ -z "$ARTIFACT_PATH" ]]; then
  echo "No fresh output artifact detected for run_id=$RUN_ID" >&2
  exit 1
fi

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm --no-deps \
  --entrypoint python transcriber_job -c "import json; from pathlib import Path; p=Path('$ARTIFACT_PATH'); d=json.loads(p.read_text(encoding='utf-8')); required={'summary_raw','segment_count','model_backend','model_profile','prompt_id'}; missing=required-d.keys(); assert not missing, f'missing keys: {sorted(missing)}'; print(p)" > "$BASE_DIR/state/last_artifact_path"

echo "[run] fresh artifact=$(cat "$BASE_DIR/state/last_artifact_path")"
rm -f "$MARKER_FILE"
