# rust_ort_fix

Standalone repository to benchmark optimized Whisper ONNX Runtime models on CPU using:
- Rust inference pipeline (`src/main.rs`)
- Python Whisper baseline for WER/CER reference
- Docker cpuset pinning for reproducible latency measurements

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
- `Dockerfile`: reproducible benchmark environment (Rust + Python + uv + ffmpeg)
- `docker_run_benchmark.sh`: helper to build and run with Docker cpuset pinning
- `src/main.rs`: Rust benchmark CLI
- `scripts/discover_optimal_model/discover_optimal_model.py`: Python-first benchmark orchestrator
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
- Docker (for containerized reproducible runs)

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

## Common Configuration (Host)

```bash
# Bound chunk-level parallelism (default: 1)
RUST_CHUNK_PARALLELISM=1 ./discover_optimal_model.sh

# Print heartbeat while each model benchmark is running (seconds)
MODEL_PROGRESS_INTERVAL_SEC=30 ./discover_optimal_model.sh

# Set WER/CER worker count (default: container/host CPU affinity core count)
METRICS_WORKERS=8 ./discover_optimal_model.sh

# Change Rust decode language (default: en)
BENCHMARK_LANG=en ./discover_optimal_model.sh
```

## Docker Reproducible Run (Recommended)

Build image with host user mapping (normal non-root user inside container):

```bash
docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t whisper-rust-bench:latest .
```

Run with Docker CPU pinning (`--cpuset-cpus`):

```bash
docker run --rm -it \
  --cpuset-cpus="0-7" \
  -v "$PWD:/workspace" \
  -w /workspace \
  whisper-rust-bench:latest
```

The benchmark runner reads container CPU affinity and sets thread planning from that scope.

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
- CPU selection is controlled only by container affinity (for example `docker run --cpuset-cpus=...`).
- Threading is bounded to avoid oversubscription by default:
  `intra-op * chunk-parallelism <= container-visible cores`.

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
