#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="$SCRIPT_DIR/scripts/discover_optimal_model/discover_optimal_model.py"

cd "$SCRIPT_DIR"

if [[ ! -f "$PY_RUNNER" ]]; then
  echo "ERROR: Missing benchmark runner: $PY_RUNNER" >&2
  exit 1
fi

exec uv run python "$PY_RUNNER" run "$@"
