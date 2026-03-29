#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-deployment/prod.env}"
STATE_DIR="deployment/state"
CURRENT_TAG_FILE="$STATE_DIR/current_prod_tag"
PREVIOUS_TAG_FILE="$STATE_DIR/previous_prod_tag"
LAST_RESULT_FILE="$STATE_DIR/last_deploy_result"
COMPOSE_FILE="docker-compose.prod.yml"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Create it from deployment/prod.env.example first." >&2
  exit 1
fi

mkdir -p "$STATE_DIR" runtime/input runtime/output runtime/logs runtime/postgres
cp -n samples/segments.sample.json runtime/input/segments.sample.json

IMAGE_REPOSITORY="$(grep -E '^IMAGE_REPOSITORY=' "$ENV_FILE" | cut -d= -f2-)"
IMAGE_NAME="$(grep -E '^IMAGE_NAME=' "$ENV_FILE" | cut -d= -f2-)"
CONFIG_TAG="$(grep -E '^IMAGE_TAG=' "$ENV_FILE" | cut -d= -f2-)"
IMAGE_TAG="${IMAGE_TAG_OVERRIDE:-$CONFIG_TAG}"

if [[ -z "$IMAGE_REPOSITORY" || -z "$IMAGE_NAME" || -z "$IMAGE_TAG" ]]; then
  echo "IMAGE_REPOSITORY, IMAGE_NAME, and IMAGE_TAG must be set in $ENV_FILE" >&2
  exit 1
fi

FULL_IMAGE="${IMAGE_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
CURRENT_TAG=""
if [[ -f "$CURRENT_TAG_FILE" ]]; then
  CURRENT_TAG="$(cat "$CURRENT_TAG_FILE")"
fi

PREVIOUS_CANDIDATE=""
if [[ -n "$CURRENT_TAG" && "$CURRENT_TAG" != "$IMAGE_TAG" ]]; then
  PREVIOUS_CANDIDATE="$CURRENT_TAG"
fi

echo "[deploy] env_file=$ENV_FILE compose_file=$COMPOSE_FILE"
echo "[deploy] current_tag=${CURRENT_TAG:-<none>} target_tag=$IMAGE_TAG"
echo "[deploy] Pulling exact image tag: $FULL_IMAGE"
docker pull "$FULL_IMAGE"

echo "[deploy] Updating IMAGE_TAG in $ENV_FILE"
if grep -q '^IMAGE_TAG=' "$ENV_FILE"; then
  sed -i "s|^IMAGE_TAG=.*$|IMAGE_TAG=${IMAGE_TAG}|" "$ENV_FILE"
else
  echo "IMAGE_TAG=${IMAGE_TAG}" >> "$ENV_FILE"
fi

echo "[deploy] Ensuring resident production dependency stack is running"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d postgres

echo "[deploy] Running post-deploy runtime validation"
"$ROOT_DIR/scripts/prod/validate_prod_runtime.sh" --env-file "$ENV_FILE"

if [[ -n "$PREVIOUS_CANDIDATE" ]]; then
  echo "$PREVIOUS_CANDIDATE" > "$PREVIOUS_TAG_FILE"
fi
echo "$IMAGE_TAG" > "$CURRENT_TAG_FILE"
printf "status=success action=deploy tag=%s utc=%s\n" "$IMAGE_TAG" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$LAST_RESULT_FILE"

echo "[deploy] Deployment successful"
echo "[deploy] state.current=$(cat "$CURRENT_TAG_FILE")"
if [[ -f "$PREVIOUS_TAG_FILE" ]]; then
  echo "[deploy] state.previous=$(cat "$PREVIOUS_TAG_FILE")"
fi
