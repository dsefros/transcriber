#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <image-tag>  (example: $0 prod-abc1234)" >&2
  exit 1
fi

TARGET_TAG="$1"
BASE_DIR="${TRANSCRIBER_HOME:-/opt/transcriber}"
ENV_FILE="$BASE_DIR/prod.env"
MODELS_FILE="$BASE_DIR/models.yaml"
COMPOSE_FILE="$BASE_DIR/docker-compose.yml"
STATE_DIR="$BASE_DIR/state"
CURRENT_TAG_FILE="$STATE_DIR/current_tag"
PREVIOUS_TAG_FILE="$STATE_DIR/previous_tag"
LAST_RESULT_FILE="$STATE_DIR/last_result"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy prod.env.example to prod.env and configure it first." >&2
  exit 1
fi
if [[ ! -f "$MODELS_FILE" ]]; then
  echo "Missing $MODELS_FILE. Provide production models.yaml first." >&2
  exit 1
fi
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Missing $COMPOSE_FILE. Deploy bundle was not installed correctly." >&2
  exit 1
fi

mkdir -p "$STATE_DIR" "$BASE_DIR/runtime/input" "$BASE_DIR/runtime/output" "$BASE_DIR/runtime/logs" "$BASE_DIR/runtime/postgres"
cp -n "$BASE_DIR/samples/segments.sample.json" "$BASE_DIR/runtime/input/segments.sample.json" 2>/dev/null || true

IMAGE_REPOSITORY="$(grep -E '^IMAGE_REPOSITORY=' "$ENV_FILE" | cut -d= -f2-)"
IMAGE_NAME="$(grep -E '^IMAGE_NAME=' "$ENV_FILE" | cut -d= -f2-)"

if [[ -z "$IMAGE_REPOSITORY" || -z "$IMAGE_NAME" ]]; then
  echo "IMAGE_REPOSITORY and IMAGE_NAME must be configured in $ENV_FILE" >&2
  exit 1
fi

FULL_IMAGE="${IMAGE_REPOSITORY}/${IMAGE_NAME}:${TARGET_TAG}"
CURRENT_TAG=""
[[ -f "$CURRENT_TAG_FILE" ]] && CURRENT_TAG="$(cat "$CURRENT_TAG_FILE")"

PREVIOUS_CANDIDATE=""
if [[ -n "$CURRENT_TAG" && "$CURRENT_TAG" != "$TARGET_TAG" ]]; then
  PREVIOUS_CANDIDATE="$CURRENT_TAG"
fi

echo "[install] base_dir=$BASE_DIR"
echo "[install] current_tag=${CURRENT_TAG:-<none>} target_tag=$TARGET_TAG"
echo "[install] pulling $FULL_IMAGE"
docker pull "$FULL_IMAGE"

if grep -q '^IMAGE_TAG=' "$ENV_FILE"; then
  sed -i "s|^IMAGE_TAG=.*$|IMAGE_TAG=${TARGET_TAG}|" "$ENV_FILE"
else
  echo "IMAGE_TAG=${TARGET_TAG}" >> "$ENV_FILE"
fi

echo "[install] starting resident dependencies"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d postgres

echo "[install] validating deployed image"
"$BASE_DIR/validate.sh"

if [[ -n "$PREVIOUS_CANDIDATE" ]]; then
  echo "$PREVIOUS_CANDIDATE" > "$PREVIOUS_TAG_FILE"
fi
echo "$TARGET_TAG" > "$CURRENT_TAG_FILE"
printf "status=success action=install tag=%s utc=%s\n" "$TARGET_TAG" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$LAST_RESULT_FILE"

echo "[install] done current=$(cat "$CURRENT_TAG_FILE")"
[[ -f "$PREVIOUS_TAG_FILE" ]] && echo "[install] previous=$(cat "$PREVIOUS_TAG_FILE")"
