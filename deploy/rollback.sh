#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${TRANSCRIBER_HOME:-/opt/transcriber}"
ENV_FILE="$BASE_DIR/prod.env"
STATE_DIR="$BASE_DIR/state"
CURRENT_TAG_FILE="$STATE_DIR/current_tag"
PREVIOUS_TAG_FILE="$STATE_DIR/previous_tag"
LAST_RESULT_FILE="$STATE_DIR/last_result"

if [[ ! -f "$PREVIOUS_TAG_FILE" ]]; then
  echo "No previous deployed tag in $PREVIOUS_TAG_FILE" >&2
  exit 1
fi

ROLLBACK_TAG="$(cat "$PREVIOUS_TAG_FILE")"
CURRENT_TAG=""
[[ -f "$CURRENT_TAG_FILE" ]] && CURRENT_TAG="$(cat "$CURRENT_TAG_FILE")"

echo "[rollback] current=${CURRENT_TAG:-<none>} target=$ROLLBACK_TAG"
"$BASE_DIR/install.sh" "$ROLLBACK_TAG"

if [[ -n "$CURRENT_TAG" && "$CURRENT_TAG" != "$ROLLBACK_TAG" ]]; then
  echo "$CURRENT_TAG" > "$PREVIOUS_TAG_FILE"
fi
echo "$ROLLBACK_TAG" > "$CURRENT_TAG_FILE"
printf "status=success action=rollback tag=%s utc=%s\n" "$ROLLBACK_TAG" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$LAST_RESULT_FILE"

echo "[rollback] done current=$(cat "$CURRENT_TAG_FILE")"
[[ -f "$PREVIOUS_TAG_FILE" ]] && echo "[rollback] previous=$(cat "$PREVIOUS_TAG_FILE")"
