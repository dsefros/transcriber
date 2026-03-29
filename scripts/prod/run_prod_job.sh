#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="deployment/prod.env"
SOURCE_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --source)
      SOURCE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_PATH" ]]; then
  echo "--source is required, for example /data/input/segments.sample.json" >&2
  exit 1
fi

COMPOSE_FILE="docker-compose.prod.yml"
mkdir -p runtime/output

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MARKER_FILE="runtime/output/.run_${RUN_ID}.marker"
BEFORE_LIST="/tmp/prod_run_before_${RUN_ID}.txt"
AFTER_LIST="/tmp/prod_run_after_${RUN_ID}.txt"
NEW_LIST="/tmp/prod_run_new_${RUN_ID}.txt"

echo "[run] env_file=$ENV_FILE compose_file=$COMPOSE_FILE run_id=$RUN_ID"
find runtime/output -maxdepth 1 -name '*_analysis.json' -print | sort > "$BEFORE_LIST"
touch "$MARKER_FILE"

echo "[run] Executing production job source: $SOURCE_PATH"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm transcriber_job "$SOURCE_PATH"

find runtime/output -maxdepth 1 -name '*_analysis.json' -print | sort > "$AFTER_LIST"
comm -13 "$BEFORE_LIST" "$AFTER_LIST" > "$NEW_LIST" || true

CANDIDATE_ARTIFACT="$(tail -n 1 "$NEW_LIST" || true)"
if [[ -z "$CANDIDATE_ARTIFACT" ]]; then
  CANDIDATE_ARTIFACT="$(find runtime/output -maxdepth 1 -name '*_analysis.json' -newer "$MARKER_FILE" -print | sort | tail -n 1 || true)"
fi

if [[ -z "$CANDIDATE_ARTIFACT" ]]; then
  echo "No fresh analysis artifact detected for run_id=$RUN_ID" >&2
  exit 1
fi

python - "$CANDIDATE_ARTIFACT" <<'PY'
import json
import sys
from pathlib import Path
artifact = Path(sys.argv[1])
payload = json.loads(artifact.read_text(encoding="utf-8"))
required = {"summary_raw", "segment_count", "model_backend", "model_profile", "prompt_id"}
missing = required - payload.keys()
if missing:
    raise SystemExit(f"Artifact {artifact} missing keys: {sorted(missing)}")
print(f"[run] Fresh artifact validated: {artifact}")
PY

echo "[run] run_id=$RUN_ID artifact=$CANDIDATE_ARTIFACT"
rm -f "$MARKER_FILE"
