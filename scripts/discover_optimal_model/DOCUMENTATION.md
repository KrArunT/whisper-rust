# Discover Optimal Model (Python-First Workflow)

This benchmark flow is now Python-first, with exactly one top-level shell entrypoint for CPU pinning orchestration.

## File Inventory

| File | Purpose |
|---|---|
| `discover_optimal_model.sh` | Canonical and only shell entrypoint. Resolves pinning plan, applies `taskset`, and launches Python workflow. |
| `scripts/discover_optimal_model/discover_optimal_model.py` | Main orchestrator: config loading, pinning resolution, baseline/model execution, aggregation, and report generation. |
| `scripts/discover_optimal_model/discover_optimal_model_metrics.py` | Python helper for baseline transcript generation, CSV text extraction, and WER/CER + latency utilities. |

## Execution Flow

1. `discover_optimal_model.sh`
2. `discover_optimal_model.py pinning-plan` (returns CPU list + run core count + description)
3. `taskset -c <resolved CPUs> uv run python ... discover_optimal_model.py run` (if pinning enabled)
4. `discover_optimal_model.py run`
   - Validate paths and dependencies
   - Backup old results
   - Generate/reuse baseline transcripts
   - Benchmark each model directory with Rust binary
   - Compute WER/CER for each model
   - Write merged CSV and markdown report

## CPU Pinning Notes

- Pinning is decided once in `discover_optimal_model.py pinning-plan`.
- The shell wrapper applies `taskset` to the long-running Python process.
- All child processes inherit affinity automatically, including:
  - Rust model inference (`cargo run --release -- ...`)
  - Baseline transcription helper calls
  - WER/CER helper calls
- This eliminates unpinned moving parts in metric stages on multi-socket/high-core hosts (for example AMD EPYC Turin).

## Environment Controls

Supported pinning controls:

- `PIN_MODE=num_cpus|cpu_set|core_list|none`
- `NUM_CPUS=<n>` (for `PIN_MODE=num_cpus`)
- `CPU_SET=0-7,16-23` (for `PIN_MODE=cpu_set`)
- `CORE_LIST=0,2,4,6` (for `PIN_MODE=core_list`)
- `PIN_STRICT_PHYSICAL_CORES=1` to fail when SMT siblings are selected
- `PIN_DEBUG=1` to print selected topology details

Additional benchmark controls remain unchanged (`MODELS_ROOT`, `AUDIO_DIR`, `RESULTS_ROOT`, `NUM_BEAMS`, `RUST_CHUNK_PARALLELISM`, etc.).

WER/CER parallelism controls:

- `METRICS_WORKERS=<n>` to force worker count
- default worker count uses `RUN_CORE_COUNT` (derived from selected pinned CPUs)
