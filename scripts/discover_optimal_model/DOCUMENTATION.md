# Discover Optimal Model (WER/CER) Documentation

This document covers all files and functions used by the discovery benchmark flow.

## File Inventory

| File | Purpose |
|---|---|
| `discover_optimal_model.sh` | Canonical entrypoint. Sources modules and runs the full pipeline. |
| `scripts/discover_optimal_model/discover_optimal_model_config.sh` | Environment/config loading and legacy env mapping. |
| `scripts/discover_optimal_model/discover_optimal_model_helpers.sh` | Shared formatting/log parsing/metadata/backup helpers. |
| `scripts/discover_optimal_model/discover_optimal_model_pinning.sh` | Strict CPU pinning and validation logic. |
| `scripts/discover_optimal_model/discover_optimal_model_python_helper.sh` | Resolves static Python helper location. |
| `scripts/discover_optimal_model/discover_optimal_model_metrics.py` | Python metrics/baseline utility used by shell workflow. |
| `scripts/discover_optimal_model/discover_optimal_model_workflow.sh` | Runtime setup, baseline/model execution, aggregation, report generation. |

## Execution Flow

`discover_optimal_model.sh -> main()`

1. `load_discover_config`
2. `setup_runtime_paths`
3. `validate_benchmark_inputs`
4. `initialize_benchmark_workspace`
5. `prepare_taskset_pinning`
6. `initialize_python_helper`
7. `initialize_merged_csv`
8. `generate_baseline_transcripts`
9. `benchmark_optimized_models`
10. `select_best_models`
11. `write_markdown_report`
12. `print_completion_summary`

## Function Reference

### `discover_optimal_model.sh`

| Function | Description |
|---|---|
| `main()` | Top-level orchestrator for config, setup, pinning, baseline/model runs, and report output. |

### `scripts/discover_optimal_model/discover_optimal_model_config.sh`

| Function | Description |
|---|---|
| `load_discover_config()` | Loads defaults, normalizes `PIN_MODE`, maps legacy env vars, and sets baseline/benchmark language, beam config, and chunk parallelism controls. |

### `scripts/discover_optimal_model/discover_optimal_model_helpers.sh`

| Function | Description |
|---|---|
| `time_to_seconds()` | Converts elapsed time tokens (`SS`, `MM:SS`, `HH:MM:SS`) to seconds. |
| `pretty_time()` | Formats seconds for report rendering. |
| `pretty_score()` | Formats WER/CER ratios as percentages while preserving `NA`. |
| `precision_from_model()` | Derives precision label from model directory name. |
| `optimization_from_model()` | Extracts optimization level (`o1`/`o2`/...) from model directory name. |
| `implementation_name()` | Returns fixed implementation label (`onnxruntime rust`). |
| `instruction_set_from_model()` | Extracts ISA suffix from model directory name. |
| `backup_old_results()` | Copies previous run artifacts to a timestamped backup directory. |
| `elapsed_wall_time_from_log()` | Reads wall time seconds from `/usr/bin/time -v` output. |
| `peak_memory_mb_from_log()` | Reads peak RSS (MB) from `/usr/bin/time -v` output. |
| `count_audio_files()` | Counts supported audio files recursively under audio input root. |

### `scripts/discover_optimal_model/discover_optimal_model_pinning.sh`

| Function | Description |
|---|---|
| `pinning_die()` | Emits pinning error and exits non-zero. |
| `is_positive_int()` | Validates positive integer values. |
| `expand_cpu_set_strict()` | Parses and validates `CPU_SET` syntax (integers + ascending ranges). |
| `expand_core_list_strict()` | Parses and validates `CORE_LIST` syntax (explicit integer IDs only). |
| `get_online_cpus()` | Returns online CPU IDs from sysfs (or `nproc` fallback). |
| `get_allowed_cpus()` | Returns affinity-allowed CPU IDs for the current process. |
| `validate_requested_cpus_or_die()` | Ensures requested CPUs are within allowed affinity set. |
| `select_num_cpus_or_die()` | Selects first `NUM_CPUS` CPUs from allowed set. |
| `select_cpu_set_or_die()` | Applies strict `cpu_set` mode selection. |
| `select_core_list_or_die()` | Applies strict `core_list` mode selection. |
| `validate_mode_inputs_or_die()` | Enforces mode-specific option exclusivity/requirements. |
| `prepare_taskset_pinning()` | Produces `TASKSET_CPU_LIST`, `RUN_CORE_COUNT`, and `PINNING_DESC`. |

### `scripts/discover_optimal_model/discover_optimal_model_python_helper.sh`

| Function | Description |
|---|---|
| `resolve_metrics_python_helper()` | Resolves and validates static Python helper path into `PY_HELPER`. |

### `scripts/discover_optimal_model/discover_optimal_model_metrics.py`

| Function | Description |
|---|---|
| `list_audio(audio_dir)` | Recursively enumerates supported audio files. |
| `ensure_whisper()` | Imports OpenAI Whisper package or raises actionable error. |
| `baseline_transcribe(audio_dir, out_dir, model_name, language)` | Generates baseline transcripts and `baseline_all.txt`. |
| `normalize_for_words(s)` | Normalization for word-level WER tokenization. |
| `normalize_for_chars(s)` | Normalization for char-level CER tokenization. |
| `levenshtein(a, b)` | Edit-distance implementation used by WER/CER. |
| `wer_cer(ref_text, hyp_text)` | Computes WER and CER metrics against baseline text. |
| `extract_text(raw)` | Heuristically strips log-like noise from raw transcript text. |
| `extract_text_from_csv(csv_path)` | Extracts transcript text from benchmark CSV columns. |
| `average_end_to_end_latency(csv_path)` | Computes mean `end_to_end_s` from per-file CSV. |
| `main()` | CLI dispatcher for `baseline`, `metrics`, `metrics_csv`, `avg_latency_csv`. |

### `scripts/discover_optimal_model/discover_optimal_model_workflow.sh`

| Function | Description |
|---|---|
| `setup_runtime_paths()` | Initializes temp/output paths for a run. |
| `require_command()` | Validates command availability in PATH. |
| `validate_benchmark_inputs()` | Performs preflight checks (dirs, model presence, required executables). |
| `initialize_benchmark_workspace()` | Backs up previous results and prepares output directories. |
| `initialize_python_helper()` | Resolves helper path and validates Python syntax. |
| `run_timed_with_optional_pinning()` | Executes a command under `/usr/bin/time`, optionally pinned with taskset. |
| `run_timed_quiet_with_optional_pinning()` | Same as above but with quiet stdout and captured stderr. |
| `compute_threading_plan()` | Computes bounded `MODEL_INTRA_OP` and `MODEL_CHUNK_PARALLELISM` so combined concurrency does not exceed pinned cores by default. |
| `initialize_merged_csv()` | Writes merged CSV header. |
| `generate_baseline_transcripts()` | Runs baseline model transcription and computes baseline metrics. |
| `run_model_benchmark(onnx_dir)` | Runs one model benchmark and appends one merged CSV row. |
| `benchmark_optimized_models()` | Iterates model directories (sorted) and benchmarks each. |
| `select_best_models()` | Selects best rows for latency, memory, WER, and CER. |
| `write_markdown_report()` | Writes markdown summary report from merged results. |
| `print_completion_summary()` | Prints final artifact paths and resolved pinning details. |
