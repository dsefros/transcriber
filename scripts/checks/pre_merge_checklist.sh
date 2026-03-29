#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONDONTWRITEBYTECODE=1

echo "[checklist] 1) unit/integration tests"
pytest

echo "[checklist] 2) local docker build smoke"
"$ROOT_DIR/scripts/checks/local_build_smoke.sh"

echo "[checklist] 2a) deploy-bundle script syntax"
bash -n deploy/install.sh deploy/run_job.sh deploy/rollback.sh deploy/validate.sh

echo "[checklist] 3) runtime diagnostics (host python)"
python -m src.app.runtime_doctor --json > /tmp/runtime_report.json

echo "[checklist] 4) CLI contract smoke (host python)"
python -m src.app.cli --help > /tmp/cli_help.txt
python -m src.app.runtime_doctor --help > /tmp/runtime_doctor_help.txt

if [[ "${ENABLE_FULL_JOB_SMOKE:-0}" == "1" ]]; then
  echo "[checklist] 5) optional full job smoke"
  : "${DATABASE_URL:?Set DATABASE_URL for full smoke.}"
  if [[ ! -f models.yaml ]]; then
    cp models.yaml.example models.yaml
  fi
  python -m src.app.cli samples/segments.sample.json
  ANALYSIS_COUNT="$(find output -maxdepth 1 -name '*_analysis.json' | wc -l | tr -d ' ')"
  if [[ "$ANALYSIS_COUNT" == "0" ]]; then
    echo "Expected analysis artifacts in output/ after full job smoke" >&2
    exit 1
  fi
fi

echo "[checklist] completed"
