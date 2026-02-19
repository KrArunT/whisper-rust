#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="$SCRIPT_DIR/scripts/discover_optimal_model/discover_optimal_model.py"

cd "$SCRIPT_DIR"

if [[ ! -f "$PY_RUNNER" ]]; then
  echo "ERROR: Missing benchmark runner: $PY_RUNNER" >&2
  exit 1
fi

if ! PIN_PLAN_RAW="$(uv run python "$PY_RUNNER" pinning-plan)"; then
  exit 1
fi

mapfile -t PIN_PLAN <<< "$PIN_PLAN_RAW"
if (( ${#PIN_PLAN[@]} < 3 )); then
  echo "ERROR: Invalid pinning plan output from $PY_RUNNER" >&2
  exit 1
fi

TASKSET_CPU_LIST="${PIN_PLAN[0]:-}"
RUN_CORE_COUNT="${PIN_PLAN[1]:-1}"
PINNING_DESC="${PIN_PLAN[2]:-none selected_cores=1}"

export TASKSET_CPU_LIST
export RUN_CORE_COUNT
export PINNING_DESC

if [[ -n "$TASKSET_CPU_LIST" ]]; then
  echo "INFO: CPU pinning enabled: $PINNING_DESC"
  exec taskset -c "$TASKSET_CPU_LIST" uv run python "$PY_RUNNER" run "$@"
fi

echo "INFO: CPU pinning disabled: $PINNING_DESC"
exec uv run python "$PY_RUNNER" run "$@"
