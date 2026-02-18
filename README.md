# rust_ort_fix

Standalone repository to benchmark optimized Whisper ONNX Runtime models on CPU using:
- Rust inference pipeline (`src/main.rs`)
- Python Whisper baseline for WER/CER reference
- Strict CPU pinning controls for reproducible latency measurements

## Overview

`discover_optimal_model.sh` runs the end-to-end discovery workflow:
1. Generate baseline transcripts with OpenAI Whisper (Python).
2. Benchmark each model directory under `models/whisper-base-optimized/` with Rust.
3. Compute WER/CER against baseline.
4. Produce merged CSV + markdown report.

If `results/benchmarks/without_hf_pipeline_rust/baseline_whisper_<model>/baseline_all.txt` already exists,
the baseline transcription step is skipped and reused.

## Repository Layout

- `discover_optimal_model.sh`: main entrypoint
- `src/main.rs`: Rust benchmark CLI
- `scripts/discover_optimal_model/`: modular shell + Python helper modules
- `scripts/discover_optimal_model/discover_optimal_model_metrics.py`: baseline/metrics utility
- `models/`: optimized ONNX model variants (input set under test)
- `audio/`: input audio corpus
- `results/`: benchmark outputs
- `scripts/discover_optimal_model/DOCUMENTATION.md`: file/function reference

## Prerequisites

### System

- Rust toolchain (`cargo`, `rustc`)
- `uv`
- `/usr/bin/time`
- `taskset` (required for pinning modes other than `PIN_MODE=none`)

### Python (uv-managed)

```bash
uv sync
```

## Build

```bash
cargo build --release
```

## Run Discovery

```bash
./discover_optimal_model.sh
```

## Common Configuration

```bash
# Check CPUs available to this shell
taskset -pc $$

# Pin first N allowed CPUs
PIN_MODE=num_cpus NUM_CPUS=8 ./discover_optimal_model.sh

# Pin explicit CPU set
PIN_MODE=cpu_set CPU_SET=0-7 ./discover_optimal_model.sh

# Pin explicit core list (order preserved)
PIN_MODE=core_list CORE_LIST=0,2,4,6 ./discover_optimal_model.sh

# Disable pinning
PIN_MODE=none ./discover_optimal_model.sh

# Bound chunk-level parallelism (default: 1)
RUST_CHUNK_PARALLELISM=1 ./discover_optimal_model.sh

# Print pinning topology diagnostics (useful on SMT/EPYC systems)
PIN_MODE=cpu_set CPU_SET=0-7 PIN_DEBUG=1 ./discover_optimal_model.sh

# Fail if selection includes SMT siblings from the same physical core
PIN_MODE=cpu_set CPU_SET=0-7 PIN_STRICT_PHYSICAL_CORES=1 ./discover_optimal_model.sh

# Change Rust decode language (default: en)
BENCHMARK_LANG=en ./discover_optimal_model.sh
```

## Metrics Semantics

In `BENCHMARK_REPORT.md`:
- `Time` = average inference end-to-end latency per audio file.
  - Model rows: mean of `end_to_end_s` from each model's `inference_per_file.csv`.
  - Baseline row: total baseline wall time / number of audio files.
- `Instruction Set` = parsed from model name if present; otherwise `NA`.
- Baseline appears as a row in the markdown comparison table.

## Output Files

Default output root: `results/benchmarks/without_hf_pipeline_rust/`

Primary artifacts:
- `merged_inference_results.csv`
- `BENCHMARK_REPORT.md`
- `baseline_whisper_<model>/baseline_all.txt`
- `baseline_whisper_<model>/baseline_metrics.env`
- `per_model/<model>/inference_per_file.csv`
- `per_model/<model>/inference_per_file.json`
- `per_model/<model>/inference_summary.json`

Backups of prior runs:
- `results/benchmarks/without_hf_pipeline_rust/backups/<timestamp>/`

## Reproducibility Notes

- Rust file discovery is recursive under `AUDIO_DIR`.
- Pinning mode is strict: invalid CPU selections fail fast.
- Threading is bounded to avoid oversubscription by default:
  `intra-op * chunk-parallelism <= selected cores`.

## Push to a New Remote Repository

This folder is already initialized as an independent git repository.

```bash
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

### Git LFS

ONNX files are configured for Git LFS via `.gitattributes`.

```bash
git lfs install
git lfs push --all origin main
```
