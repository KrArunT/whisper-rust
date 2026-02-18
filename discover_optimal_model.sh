#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$SCRIPT_DIR/scripts/discover_optimal_model"

# shellcheck disable=SC1091
source "$MODULE_DIR/discover_optimal_model_config.sh"
# shellcheck disable=SC1091
source "$MODULE_DIR/discover_optimal_model_helpers.sh"
# shellcheck disable=SC1091
source "$MODULE_DIR/discover_optimal_model_pinning.sh"
# shellcheck disable=SC1091
source "$MODULE_DIR/discover_optimal_model_python_helper.sh"
# shellcheck disable=SC1091
source "$MODULE_DIR/discover_optimal_model_workflow.sh"

# Orchestrate the full benchmark pipeline from config load to report output.
main() {
  load_discover_config
  setup_runtime_paths
  validate_benchmark_inputs
  initialize_benchmark_workspace

  # Build taskset pinning plan once for all model runs.
  prepare_taskset_pinning

  initialize_python_helper
  initialize_merged_csv
  generate_baseline_transcripts
  benchmark_optimized_models
  select_best_models
  write_markdown_report
  print_completion_summary
}

main "$@"
