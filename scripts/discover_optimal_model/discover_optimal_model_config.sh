#!/usr/bin/env bash

# Load benchmark defaults and map legacy env vars to the strict pinning interface.
load_discover_config() {
  MODELS_ROOT="${MODELS_ROOT:-models/whisper-base-optimized}"
  AUDIO_DIR="${AUDIO_DIR:-audio}"
  RESULTS_ROOT="${RESULTS_ROOT:-results/benchmarks/without_hf_pipeline_rust}"

  # New strict pinning interface.
  #   PIN_MODE: num_cpus | cpu_set | core_list | none
  PIN_MODE_INPUT="${PIN_MODE:-}"
  NUM_CPUS="${NUM_CPUS:-${MAX_CORES_PER_RUN:-8}}"
  CPU_SET="${CPU_SET:-${CPUSET_LIST:-${PIN_CPUS:-}}}"  # supports ranges, e.g. 0-7,16-23
  CORE_LIST="${CORE_LIST:-}"                            # explicit ids only, e.g. 0,2,4,6
  NUM_BEAMS="${NUM_BEAMS:-1}"
  BENCHMARK_LANG="${BENCHMARK_LANG:-en}"
  RUST_CHUNK_PARALLELISM="${RUST_CHUNK_PARALLELISM:-1}"
  PIN_DEBUG="${PIN_DEBUG:-0}"                           # 1 prints selected CPU topology diagnostics
  PIN_STRICT_PHYSICAL_CORES="${PIN_STRICT_PHYSICAL_CORES:-0}" # 1 rejects SMT sibling selection

  # Backward-compatible mapping from legacy controls.
  if [[ -z "$PIN_MODE_INPUT" ]]; then
    if [[ "${ENABLE_TASKSET_PINNING:-1}" != "1" ]]; then
      PIN_MODE="none"
    elif [[ -n "$CPU_SET" ]]; then
      PIN_MODE="cpu_set"
    elif [[ -n "$CORE_LIST" ]]; then
      PIN_MODE="core_list"
    else
      PIN_MODE="num_cpus"
    fi
  else
    PIN_MODE="$(echo "$PIN_MODE_INPUT" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  fi

  # Whisper baseline settings.
  BASELINE_MODEL="${BASELINE_MODEL:-base}"   # whisper model name: tiny/base/small/...
  BASELINE_LANG="${BASELINE_LANG:-en}"
}
