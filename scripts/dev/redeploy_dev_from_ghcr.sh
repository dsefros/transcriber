#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${GHCR_IMAGE:?Set GHCR_IMAGE like ghcr.io/<owner>/<repo>}"
DEV_TAG="${DEV_TAG:-dev-latest}"
FULL_IMAGE="${GHCR_IMAGE}:${DEV_TAG}"

if [[ ! -f models.yaml ]]; then
  cp models.yaml.example models.yaml
fi

: "${DATABASE_URL:=postgresql://transcriber:transcriber@host.docker.internal:5432/transcriber}"

echo "[dev-ghcr] Pulling $FULL_IMAGE"
docker pull "$FULL_IMAGE"

echo "[dev-ghcr] Validating CLI/runtime contract from published artifact"
docker run --rm -e DATABASE_URL="$DATABASE_URL" -v "$ROOT_DIR/models.yaml:/app/models.yaml:ro" "$FULL_IMAGE" --help >/tmp/dev_ghcr_cli_help.txt

docker run --rm --entrypoint python -e DATABASE_URL="$DATABASE_URL" -v "$ROOT_DIR:/workspace" -w /workspace "$FULL_IMAGE" -m src.app.runtime_doctor --json >/tmp/dev_ghcr_runtime_doctor.json

if [[ "${ENABLE_FULL_JOB_SMOKE:-0}" == "1" ]]; then
  mkdir -p runtime/input runtime/output
  cp -n samples/segments.sample.json runtime/input/segments.sample.json
  echo "[dev-ghcr] Running optional sample job"
  docker run --rm \
    -e DATABASE_URL="$DATABASE_URL" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$FULL_IMAGE" \
    runtime/input/segments.sample.json
fi

echo "[dev-ghcr] Done"
