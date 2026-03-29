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
  echo "Missing $ENV_FILE. Cannot perform rollback." >&2
  exit 1
fi

if [[ ! -f "$PREVIOUS_TAG_FILE" ]]; then
  echo "No previous deployment tag recorded at $PREVIOUS_TAG_FILE" >&2
  exit 1
fi

ROLLBACK_TAG="$(cat "$PREVIOUS_TAG_FILE")"
CURRENT_TAG=""
if [[ -f "$CURRENT_TAG_FILE" ]]; then
  CURRENT_TAG="$(cat "$CURRENT_TAG_FILE")"
fi

IMAGE_REPOSITORY="$(grep -E '^IMAGE_REPOSITORY=' "$ENV_FILE" | cut -d= -f2-)"
IMAGE_NAME="$(grep -E '^IMAGE_NAME=' "$ENV_FILE" | cut -d= -f2-)"
FULL_IMAGE="${IMAGE_REPOSITORY}/${IMAGE_NAME}:${ROLLBACK_TAG}"

echo "[rollback] env_file=$ENV_FILE compose_file=$COMPOSE_FILE"
echo "[rollback] current_tag=${CURRENT_TAG:-<none>} rollback_target=$ROLLBACK_TAG"
echo "[rollback] Pulling rollback image: $FULL_IMAGE"
docker pull "$FULL_IMAGE"

echo "[rollback] Updating IMAGE_TAG in $ENV_FILE"
if grep -q '^IMAGE_TAG=' "$ENV_FILE"; then
  sed -i "s|^IMAGE_TAG=.*$|IMAGE_TAG=${ROLLBACK_TAG}|" "$ENV_FILE"
else
  echo "IMAGE_TAG=${ROLLBACK_TAG}" >> "$ENV_FILE"
fi

echo "[rollback] Ensuring resident production dependency stack is running"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d postgres

echo "[rollback] Running post-rollback runtime validation"
"$ROOT_DIR/scripts/prod/validate_prod_runtime.sh" --env-file "$ENV_FILE"

# Symmetric state transition: after rollback, current becomes rollback target,
# and previous becomes the tag we rolled back from (if it existed).
if [[ -n "$CURRENT_TAG" && "$CURRENT_TAG" != "$ROLLBACK_TAG" ]]; then
  echo "$CURRENT_TAG" > "$PREVIOUS_TAG_FILE"
fi
echo "$ROLLBACK_TAG" > "$CURRENT_TAG_FILE"
printf "status=success action=rollback tag=%s utc=%s\n" "$ROLLBACK_TAG" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$LAST_RESULT_FILE"

echo "[rollback] Rollback successful"
echo "[rollback] state.current=$(cat "$CURRENT_TAG_FILE")"
if [[ -f "$PREVIOUS_TAG_FILE" ]]; then
  echo "[rollback] state.previous=$(cat "$PREVIOUS_TAG_FILE")"
fi
