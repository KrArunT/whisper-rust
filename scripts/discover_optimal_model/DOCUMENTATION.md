# Discover Optimal Model (Container-Affinity Workflow)

This benchmark flow is Python-first with one shell entrypoint. Host `taskset` pinning is removed.

## File Inventory

| File | Purpose |
|---|---|
| `discover_optimal_model.sh` | Canonical shell entrypoint; launches Python workflow directly. |
| `scripts/discover_optimal_model/discover_optimal_model.py` | Main orchestrator: config, CPU affinity detection, baseline/model execution, aggregation, report generation. |
| `scripts/discover_optimal_model/discover_optimal_model_metrics.py` | Python helper for baseline generation, CSV extraction, WER/CER, and latency utilities. |

## Execution Flow

1. `discover_optimal_model.sh`
2. `discover_optimal_model.py run`
   - Validate paths and dependencies
   - Detect process CPU affinity (`os.sched_getaffinity`) and derive `RUN_CORE_COUNT`
   - Backup old results
   - Generate/reuse baseline transcripts
   - Benchmark model directories with Rust binary
   - Compute WER/CER for each model
   - Write merged CSV and markdown report

## CPU Pinning Model

- CPU pinning is done by the container runtime, not by the benchmark scripts.
- Use Docker cpuset controls (example: `docker run --cpuset-cpus=0-7 ...`).
- The Python workflow reads visible CPUs from affinity and uses that to bound threading.
- WER/CER worker default also follows this affinity-derived `RUN_CORE_COUNT`.

## Environment Controls

- `MODELS_ROOT`, `AUDIO_DIR`, `RESULTS_ROOT`
- `NUM_BEAMS`, `BENCHMARK_LANG`
- `RUST_CHUNK_PARALLELISM`
- `MODEL_PROGRESS_INTERVAL_SEC`
- `BASELINE_MODEL`, `BASELINE_LANG`

WER/CER parallelism:

- `METRICS_WORKERS=<n>` to force worker count
- default uses affinity-derived `RUN_CORE_COUNT`
