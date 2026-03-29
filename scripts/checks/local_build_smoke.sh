#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${IMAGE_NAME:-transcriber}"
IMAGE_TAG="${IMAGE_TAG:-local-smoke}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

if [[ ! -f models.yaml ]]; then
  cp models.yaml.example models.yaml
fi

: "${DATABASE_URL:=postgresql://transcriber:transcriber@host.docker.internal:5432/transcriber}"

echo "[smoke] Build local image $FULL_IMAGE"
docker build -t "$FULL_IMAGE" .

echo "[smoke] Validate image contract (cli --help)"
docker run --rm -e DATABASE_URL="$DATABASE_URL" -v "$ROOT_DIR/models.yaml:/app/models.yaml:ro" "$FULL_IMAGE" --help >/tmp/local_smoke_cli_help.txt

echo "[smoke] Validate runtime doctor"
docker run --rm --entrypoint python -e DATABASE_URL="$DATABASE_URL" -v "$ROOT_DIR:/workspace" -w /workspace "$FULL_IMAGE" -m src.app.runtime_doctor --json >/tmp/local_smoke_runtime_doctor.json

echo "[smoke] Completed"
