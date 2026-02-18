#!/usr/bin/env bash

# Resolve and validate the static Python metrics helper path.
resolve_metrics_python_helper() {
  local helper_path="$MODULE_DIR/discover_optimal_model_metrics.py"
  if [[ ! -f "$helper_path" ]]; then
    echo "❌ Missing Python helper: $helper_path" >&2
    exit 1
  fi
  # shellcheck disable=SC2034  # Used by sourced workflow module.
  PY_HELPER="$helper_path"
}
