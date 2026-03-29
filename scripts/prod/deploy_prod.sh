#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TRANSCRIBER_HOME="${TRANSCRIBER_HOME:-$ROOT_DIR/deploy}"
exec "$ROOT_DIR/deploy/install.sh" "$@"
