#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TRANSCRIBER_HOME="${TRANSCRIBER_HOME:-$ROOT_DIR/deploy}"

TAG_A="${TAG_A:-}"
TAG_B="${TAG_B:-}"
SOURCE_PATH="${SOURCE_PATH:-/data/input/segments.sample.json}"

if [[ -z "$TAG_A" || -z "$TAG_B" ]]; then
  echo "Set TAG_A and TAG_B (example: TAG_A=prod-abc1234 TAG_B=prod-def5678)." >&2
  exit 1
fi

echo "[proof] install tag A=$TAG_A"
"$ROOT_DIR/deploy/install.sh" "$TAG_A"

echo "[proof] install tag B=$TAG_B"
"$ROOT_DIR/deploy/install.sh" "$TAG_B"

echo "[proof] validate + sample"
RUN_SAMPLE_JOB=1 "$ROOT_DIR/deploy/validate.sh"

echo "[proof] run explicit job"
"$ROOT_DIR/deploy/run_job.sh" "$SOURCE_PATH"

echo "[proof] rollback"
"$ROOT_DIR/deploy/rollback.sh"

echo "[proof] state files"
ls -1 "$TRANSCRIBER_HOME/state"
cat "$TRANSCRIBER_HOME/state/current_tag"
cat "$TRANSCRIBER_HOME/state/previous_tag"
