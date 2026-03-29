#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/../.." && pwd)"
export TRANSCRIBER_HOME="${TRANSCRIBER_HOME:-$ROOT_DIR/deploy}"
if [[ "${1:-}" == "--run-job-validation" ]]; then
  export RUN_SAMPLE_JOB=1
  shift
fi
exec "$ROOT_DIR/deploy/validate.sh" "$@"
