#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="$SCRIPT_DIR/scripts/discover_optimal_model/discover_optimal_model.py"
PYTHON_BIN="${BENCHMARK_PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"

if [[ ! -f "$PY_RUNNER" ]]; then
  echo "ERROR: Missing benchmark runner: $PY_RUNNER" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$PY_RUNNER" run "$@"
