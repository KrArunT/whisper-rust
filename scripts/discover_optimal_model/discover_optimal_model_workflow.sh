#!/usr/bin/env bash

# Initialize temp and output paths used across the benchmark run.
setup_runtime_paths() {
  mkdir -p "$RESULTS_ROOT"
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT

  MERGED_MODEL_CSV="$RESULTS_ROOT/merged_inference_results.csv"
  REPORT_MD="$RESULTS_ROOT/BENCHMARK_REPORT.md"
  PER_MODEL_RESULTS_DIR="$RESULTS_ROOT/per_model"

  BASELINE_DIR="$RESULTS_ROOT/baseline_whisper_${BASELINE_MODEL}"
  BASELINE_ALL_TXT="$BASELINE_DIR/baseline_all.txt"
  BASELINE_METRICS_ENV="$BASELINE_DIR/baseline_metrics.env"
  BASELINE_TIME_LOG="$TMP_DIR/time_baseline.txt"
}

# Fail fast when a required command is not available in PATH.
require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "❌ Required command not found: $cmd" >&2
    exit 1
  }
}

# Validate key runtime inputs and dependencies before any benchmark work starts.
validate_benchmark_inputs() {
  [[ -d "$AUDIO_DIR" ]] || { echo "❌ AUDIO_DIR does not exist: $AUDIO_DIR" >&2; exit 1; }
  [[ -d "$MODELS_ROOT" ]] || { echo "❌ MODELS_ROOT does not exist: $MODELS_ROOT" >&2; exit 1; }

  if ! find "$MODELS_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "❌ No model directories found in: $MODELS_ROOT" >&2
    exit 1
  fi

  require_command uv
  require_command cargo
  [[ -x /usr/bin/time ]] || { echo "❌ Required executable missing: /usr/bin/time" >&2; exit 1; }
}

# Create output directories and backup previous run artifacts.
initialize_benchmark_workspace() {
  backup_old_results
  mkdir -p "$BASELINE_DIR"
  mkdir -p "$PER_MODEL_RESULTS_DIR"
}

# Resolve static Python helper and validate script syntax.
initialize_python_helper() {
  resolve_metrics_python_helper
  uv run python -c 'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' "$PY_HELPER" \
    || { echo "❌ Python helper has invalid syntax: $PY_HELPER" >&2; exit 1; }
}

# Run a command under /usr/bin/time, optionally pinned with taskset.
run_timed_with_optional_pinning() {
  local time_log="$1"
  shift
  if [[ -n "$TASKSET_CPU_LIST" ]]; then
    /usr/bin/time -v -o "$time_log" taskset -c "$TASKSET_CPU_LIST" "$@"
  else
    /usr/bin/time -v -o "$time_log" "$@"
  fi
}

# Run a command under /usr/bin/time, silence stdout, and capture stderr.
run_timed_quiet_with_optional_pinning() {
  local time_log="$1"
  local stderr_log="$2"
  local interval="${MODEL_PROGRESS_INTERVAL_SEC:-30}"
  local run_label="${CURRENT_RUN_LABEL:-command}"
  local start_ts elapsed pid
  local -a cmd=()
  shift 2

  if [[ -n "$TASKSET_CPU_LIST" ]]; then
    cmd=(taskset -c "$TASKSET_CPU_LIST" "$@")
  else
    cmd=("$@")
  fi

  if [[ "$interval" =~ ^[0-9]+$ ]] && (( interval > 0 )); then
    /usr/bin/time -v -o "$time_log" "${cmd[@]}" 1>/dev/null 2>"$stderr_log" &
    pid=$!
    start_ts="$(date +%s)"
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$interval"
      if kill -0 "$pid" 2>/dev/null; then
        elapsed="$(( $(date +%s) - start_ts ))"
        echo "⏳ ${run_label} still running (${elapsed}s elapsed)"
      fi
    done
    wait "$pid"
  else
    /usr/bin/time -v -o "$time_log" "${cmd[@]}" 1>/dev/null 2>"$stderr_log"
  fi
}

# Compute per-run threading values while bounding total runnable workers to RUN_CORE_COUNT.
compute_threading_plan() {
  local desired_chunk="$RUST_CHUNK_PARALLELISM"

  if [[ ! "$RUN_CORE_COUNT" =~ ^[0-9]+$ ]] || (( RUN_CORE_COUNT < 1 )); then
    echo "❌ Invalid RUN_CORE_COUNT: $RUN_CORE_COUNT" >&2
    exit 1
  fi
  if [[ ! "$desired_chunk" =~ ^[0-9]+$ ]] || (( desired_chunk < 1 )); then
    echo "❌ RUST_CHUNK_PARALLELISM must be a positive integer (got '$RUST_CHUNK_PARALLELISM')" >&2
    exit 1
  fi

  if (( desired_chunk > RUN_CORE_COUNT )); then
    desired_chunk="$RUN_CORE_COUNT"
  fi

  MODEL_CHUNK_PARALLELISM="$desired_chunk"
  MODEL_INTRA_OP=$(( RUN_CORE_COUNT / MODEL_CHUNK_PARALLELISM ))
  if (( MODEL_INTRA_OP < 1 )); then
    MODEL_INTRA_OP=1
  fi
}

# Initialize merged CSV output with header row.
initialize_merged_csv() {
  echo "implementation,precision,optimization,instruction_set,beam_size,time_sec,ram_mb,wer,cer" \
    > "$MERGED_MODEL_CSV"
}

# Generate baseline transcripts and baseline timing/memory summary metrics.
generate_baseline_transcripts() {
  local baseline_cache_hit=0
  if [[ -s "$BASELINE_ALL_TXT" ]]; then
    baseline_cache_hit=1
    echo "♻️ Reusing existing baseline transcripts: $BASELINE_ALL_TXT"
  else
    echo "🎯 Generating baseline transcripts with OpenAI Whisper '${BASELINE_MODEL}'..."
    run_timed_with_optional_pinning \
      "$BASELINE_TIME_LOG" \
      uv run python "$PY_HELPER" baseline "$AUDIO_DIR" "$BASELINE_DIR" "$BASELINE_MODEL" "$BASELINE_LANG"
  fi

  if [[ ! -f "$BASELINE_ALL_TXT" ]]; then
    echo "❌ Baseline transcript missing: $BASELINE_ALL_TXT" >&2
    exit 1
  fi

  BASELINE_AUDIO_COUNT="$(count_audio_files "$AUDIO_DIR")"
  if [[ ! "$BASELINE_AUDIO_COUNT" =~ ^[0-9]+$ ]] || (( BASELINE_AUDIO_COUNT <= 0 )); then
    echo "❌ No supported audio files found under: $AUDIO_DIR" >&2
    exit 1
  fi

  if (( baseline_cache_hit == 0 )); then
    BASELINE_TOTAL_SEC="$(elapsed_wall_time_from_log "$BASELINE_TIME_LOG")"
    BASELINE_AVG_SEC="$(awk -v t="$BASELINE_TOTAL_SEC" -v n="$BASELINE_AUDIO_COUNT" 'BEGIN{ if (n>0) printf "%.6f", t/n; else printf "0" }')"
    BASELINE_PEAK_MB="$(peak_memory_mb_from_log "$BASELINE_TIME_LOG")"
    cat > "$BASELINE_METRICS_ENV" <<EOF
BASELINE_TOTAL_SEC=$BASELINE_TOTAL_SEC
BASELINE_AVG_SEC=$BASELINE_AVG_SEC
BASELINE_PEAK_MB=$BASELINE_PEAK_MB
EOF
  else
    if [[ -s "$BASELINE_METRICS_ENV" ]]; then
      # shellcheck disable=SC1090
      source "$BASELINE_METRICS_ENV"
    else
      BASELINE_TOTAL_SEC="NA"
      BASELINE_AVG_SEC="NA"
      BASELINE_PEAK_MB="NA"
    fi
  fi

  BASELINE_WER="0.000000"
  BASELINE_CER="0.000000"

  echo "✅ Baseline ready: $BASELINE_ALL_TXT"
}

# Benchmark a single model directory and append one row to merged CSV.
run_model_benchmark() {
  local onnx_dir="$1"
  local model_name time_log hyp_clean run_err model_tmp_dir model_csv model_json model_summary
  local wall_time_sec avg_time_sec time_sec peak_mb precision opt_label isa_label impl metrics wer cer

  model_name="$(basename "$onnx_dir")"
  if [[ -n "$TASKSET_CPU_LIST" ]]; then
    echo "🚀 Benchmarking $model_name (pinned: $TASKSET_CPU_LIST)"
  else
    echo "🚀 Benchmarking $model_name"
  fi

  time_log="$TMP_DIR/time_${model_name}.txt"
  hyp_clean="$TMP_DIR/hyp_clean_${model_name}.txt"
  run_err="$TMP_DIR/run_stderr_${model_name}.txt"
  model_tmp_dir="$PER_MODEL_RESULTS_DIR/$model_name"
  model_csv="$model_tmp_dir/inference_per_file.csv"
  model_json="$model_tmp_dir/inference_per_file.json"
  model_summary="$model_tmp_dir/inference_summary.json"
  mkdir -p "$model_tmp_dir"
  compute_threading_plan

  CURRENT_RUN_LABEL="model ${model_name}"
  run_timed_quiet_with_optional_pinning \
    "$time_log" \
    "$run_err" \
    cargo run --release -- \
      --audio-dir "$AUDIO_DIR" \
      --onnx-dir "$onnx_dir" \
      --language "$BENCHMARK_LANG" \
      --task transcribe \
      --max-new-tokens 128 \
      --num-beams "$NUM_BEAMS" \
      --intra-op "$MODEL_INTRA_OP" \
      --inter-op 1 \
      --chunk-parallelism "$MODEL_CHUNK_PARALLELISM" \
      --warmup 1 \
      --out-csv "$model_csv" \
      --out-json "$model_json" \
      --out-summary-json "$model_summary"
  unset CURRENT_RUN_LABEL

  wall_time_sec="$(elapsed_wall_time_from_log "$time_log")"
  avg_time_sec="$(uv run python "$PY_HELPER" avg_latency_csv "$model_csv" || true)"
  if [[ -n "${avg_time_sec:-}" ]]; then
    time_sec="$avg_time_sec"
  else
    time_sec="$wall_time_sec"
  fi

  peak_mb="$(peak_memory_mb_from_log "$time_log")"
  precision="$(precision_from_model "$model_name")"
  opt_label="$(optimization_from_model "$model_name")"
  isa_label="$(instruction_set_from_model "$model_name")"
  impl="$(implementation_name)"

  echo "🧮 Computing WER/CER for $model_name..."
  if [[ "${METRICS_TIMEOUT_SEC:-0}" =~ ^[0-9]+$ ]] && (( METRICS_TIMEOUT_SEC > 0 )) && command -v timeout >/dev/null 2>&1; then
    metrics="$(timeout "$METRICS_TIMEOUT_SEC" uv run python "$PY_HELPER" metrics_csv "$BASELINE_ALL_TXT" "$model_csv" "$hyp_clean" || true)"
  else
    metrics="$(uv run python "$PY_HELPER" metrics_csv "$BASELINE_ALL_TXT" "$model_csv" "$hyp_clean" || true)"
  fi
  if [[ -z "${metrics:-}" ]]; then
    if [[ -f "$hyp_clean" ]]; then
      echo "⚠️ Metrics unavailable for $model_name (timeout/error). Recording WER/CER as NA."
    fi
    wer="NA"
    cer="NA"
  else
    wer="$(cut -d, -f1 <<<"$metrics")"
    cer="$(cut -d, -f2 <<<"$metrics")"
  fi

  echo "$impl,$precision,$opt_label,$isa_label,$NUM_BEAMS,$time_sec,$peak_mb,$wer,$cer" \
    >> "$MERGED_MODEL_CSV"
}

# Iterate all model directories in sorted order and run per-model benchmarks.
benchmark_optimized_models() {
  local onnx_dir
  while IFS= read -r onnx_dir; do
    [[ -d "$onnx_dir" ]] || continue
    run_model_benchmark "$onnx_dir"
  done < <(find "$MODELS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
}

# Select best latency/memory/accuracy rows from the merged CSV.
select_best_models() {
  BEST_LATENCY="$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k6 -n | head -n1)"
  BEST_MEMORY="$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k7 -n | head -n1)"

  BEST_WER="$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$8!="NA"{print}' | sort -t, -k8 -n | head -n1)"
  BEST_CER="$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$9!="NA"{print}' | sort -t, -k9 -n | head -n1)"
}

# Render markdown report from aggregated benchmark and baseline metrics.
write_markdown_report() {
  local baseline_total_fmt baseline_avg_fmt baseline_ram_fmt baseline_ram_cell
  baseline_total_fmt="$(pretty_time "$BASELINE_TOTAL_SEC")"
  baseline_avg_fmt="$(pretty_time "$BASELINE_AVG_SEC")"
  if [[ -z "${BASELINE_PEAK_MB:-}" || "$BASELINE_PEAK_MB" == "NA" ]]; then
    baseline_ram_fmt="NA"
    baseline_ram_cell="NA"
  else
    baseline_ram_fmt="${BASELINE_PEAK_MB}MB"
    baseline_ram_cell="${BASELINE_PEAK_MB}MB"
  fi

  {
    echo "# ⚡ Whisper ONNX Inference Benchmark"
    echo
    echo "**Baseline (accuracy reference):** OpenAI Whisper \`$BASELINE_MODEL\` via python \`whisper\` library"
    echo "**CPU pinning:** \`$PINNING_DESC\`"
    echo "**Time column:** average end-to-end latency per audio from \`inference_per_file.csv\`"
    echo
    echo "## 📌 Baseline Metrics"
    echo "- Files: **$BASELINE_AUDIO_COUNT**"
    echo "- Time (total): **${baseline_total_fmt}**"
    echo "- Time (avg/audio): **${baseline_avg_fmt}**"
    echo "- RAM (peak): **${baseline_ram_fmt}**"
    echo "- WER/CER (vs baseline): **$(pretty_score "$BASELINE_WER")** / **$(pretty_score "$BASELINE_CER")**"
    echo
    echo "| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |"
    echo "|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|"

    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "openai-whisper python" "fp32" "baseline" "NA" "NA" \
      "${baseline_avg_fmt}" "${baseline_ram_cell}" \
      "$(pretty_score "$BASELINE_WER")" "$(pretty_score "$BASELINE_CER")"

    tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k2,2 -k3,3V -k4,4 | \
      while IFS=, read -r impl prec opt isa beam t ram wer cer; do
        ram_cell="NA"
        if [[ -z "${ram:-}" || "$ram" == "NA" ]]; then
          ram_cell="NA"
        else
          ram_cell="${ram}MB"
        fi
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
          "$impl" "$prec" "$opt" "$isa" "$beam" "$(pretty_time "$t")" "$ram_cell" "$(pretty_score "$wer")" "$(pretty_score "$cer")"
      done

    echo
    echo "## 🏎 Lowest Latency"
    echo "- **$(cut -d, -f1 <<<"$BEST_LATENCY")**"
    echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_LATENCY")**"
    echo "- Instruction set: **$(cut -d, -f4 <<<"$BEST_LATENCY")**"
    echo "- Time: **$(pretty_time "$(cut -d, -f6 <<<"$BEST_LATENCY")")**"
    echo "- WER/CER: **$(pretty_score "$(cut -d, -f8 <<<"$BEST_LATENCY")")** / **$(pretty_score "$(cut -d, -f9 <<<"$BEST_LATENCY")")**"

    echo
    echo "## 🧠 Lowest Memory"
    echo "- **$(cut -d, -f1 <<<"$BEST_MEMORY")**"
    echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_MEMORY")**"
    echo "- Instruction set: **$(cut -d, -f4 <<<"$BEST_MEMORY")**"
    echo "- RAM: **$(cut -d, -f7 <<<"$BEST_MEMORY")MB**"
    echo "- WER/CER: **$(pretty_score "$(cut -d, -f8 <<<"$BEST_MEMORY")")** / **$(pretty_score "$(cut -d, -f9 <<<"$BEST_MEMORY")")**"

    echo
    echo "## 🎯 Best Accuracy"
    if [[ -n "${BEST_WER:-}" ]]; then
      echo "- Lowest WER Optimization: **$(cut -d, -f3 <<<"$BEST_WER")** on **$(cut -d, -f4 <<<"$BEST_WER")** (WER **$(pretty_score "$(cut -d, -f8 <<<"$BEST_WER")")**) "
    else
      echo "- Lowest WER: **NA** (no valid WER computed)"
    fi
    if [[ -n "${BEST_CER:-}" ]]; then
      echo "- Lowest CER Optimization: **$(cut -d, -f3 <<<"$BEST_CER")** on **$(cut -d, -f4 <<<"$BEST_CER")** (CER **$(pretty_score "$(cut -d, -f9 <<<"$BEST_CER")")**) "
    else
      echo "- Lowest CER: **NA** (no valid CER computed)"
    fi
  } > "$REPORT_MD"
}

# Print final artifact locations and resolved pinning configuration.
print_completion_summary() {
  echo "✅ Benchmark completed"
  echo "📄 CSV   : $MERGED_MODEL_CSV"
  echo "📄 Report: $REPORT_MD"
  echo "📁 Baseline transcripts: $BASELINE_DIR"
  echo "📌 Pinning: $PINNING_DESC"
}
