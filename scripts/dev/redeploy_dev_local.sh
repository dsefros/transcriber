#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[dev-local] Running local build smoke from working tree"
"$ROOT_DIR/scripts/checks/local_build_smoke.sh"

if [[ "${ENABLE_FULL_JOB_SMOKE:-0}" == "1" ]]; then
  echo "[dev-local] Running optional full job smoke against local image"
  : "${DATABASE_URL:?Set DATABASE_URL for full job smoke}"
  if [[ ! -f models.yaml ]]; then
    cp models.yaml.example models.yaml
  fi
  mkdir -p runtime/input runtime/output
  cp -n samples/segments.sample.json runtime/input/segments.sample.json
  docker run --rm \
    -e DATABASE_URL="$DATABASE_URL" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    transcriber:local-smoke \
    runtime/input/segments.sample.json
fi

echo "[dev-local] Done"
