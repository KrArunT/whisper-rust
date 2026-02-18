#!/usr/bin/env bash

# Convert elapsed time tokens (SS, MM:SS, HH:MM:SS) to seconds.
time_to_seconds() {
  awk -F: '{
    if (NF==3)      { printf "%.3f", ($1*3600)+($2*60)+$3 }
    else if (NF==2) { printf "%.3f", ($1*60)+$2 }
    else            { printf "%.3f", $1 }
  }' <<< "$1"
}

# Format seconds for human-readable report output.
pretty_time() {
  awk -v t="$1" 'BEGIN{
    if (t=="" || t=="NA") { printf "NA"; exit }
    if (t !~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/) { printf "NA"; exit }
    t=t+0
    if (t < 60) printf "%.2fs", t
    else printf "%dm%.2fs", int(t/60), (t - (int(t/60) * 60))
  }'
}

# Format metric ratios as percentages, preserving NA.
pretty_score() {
  # format float like 0.1234 -> 12.34%
  awk -v x="$1" 'BEGIN{
    if (x=="" || x=="NA") { printf "NA" }
    else { printf "%.2f%%", (x*100.0) }
  }'
}

# Infer model precision label from model directory name.
precision_from_model() {
  [[ "$1" == *int8* ]] && echo "int8" || echo "fp32"
}

# Extract optimization label (o1/o2/...) from model directory name.
optimization_from_model() {
  local name="$1"
  if [[ "$name" =~ (o[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

# Return fixed implementation label used in aggregate results.
implementation_name() {
  echo "onnxruntime rust"
}

# Extract ISA tag from int8 model directory naming convention.
instruction_set_from_model() {
  local name="$1"
  if [[ "$name" =~ _int8_([^_]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "NA"
  fi
}

# Backup prior benchmark artifacts to a timestamped folder.
backup_old_results() {
  local ts backup_dir copied=0
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_dir="$RESULTS_ROOT/backups/$ts"
  mkdir -p "$backup_dir"

  if [[ -f "$MERGED_MODEL_CSV" ]]; then
    cp -a "$MERGED_MODEL_CSV" "$backup_dir/"
    copied=1
  fi
  if [[ -f "$REPORT_MD" ]]; then
    cp -a "$REPORT_MD" "$backup_dir/"
    copied=1
  fi
  if [[ -d "$PER_MODEL_RESULTS_DIR" ]]; then
    cp -a "$PER_MODEL_RESULTS_DIR" "$backup_dir/"
    copied=1
  fi
  if [[ -d "$BASELINE_DIR" ]]; then
    cp -a "$BASELINE_DIR" "$backup_dir/"
    copied=1
  fi

  if [[ "$copied" -eq 1 ]]; then
    echo "💾 Backed up previous benchmark results to: $backup_dir"
  else
    rmdir "$backup_dir" 2>/dev/null || true
  fi
}

# Parse /usr/bin/time wall clock duration and return seconds.
elapsed_wall_time_from_log() {
  local time_log="$1"
  local raw_time
  raw_time="$(grep "Elapsed (wall clock) time" "$time_log" | awk '{print $NF}' || true)"
  if [[ -n "${raw_time:-}" ]]; then
    time_to_seconds "$raw_time"
  else
    echo "0"
  fi
}

# Parse /usr/bin/time max resident set size and return MB.
peak_memory_mb_from_log() {
  local time_log="$1"
  local peak_kb
  peak_kb="$(grep "Maximum resident set size" "$time_log" | awk '{print $6}' || true)"
  if [[ -n "${peak_kb:-}" ]]; then
    awk "BEGIN { printf \"%.0f\", $peak_kb / 1024 }"
  else
    echo "0"
  fi
}

# Count supported audio inputs recursively under the provided audio directory.
count_audio_files() {
  local audio_dir="$1"
  find "$audio_dir" -type f \
    \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.flac" -o -iname "*.ogg" \
       -o -iname "*.webm" -o -iname "*.aac" -o -iname "*.wma" -o -iname "*.opus" \) \
    | wc -l | tr -d ' '
}
