#!/usr/bin/env bash

# Print a pinning error and abort the run.
pinning_die() {
  echo "❌ [pinning] $*" >&2
  exit 1
}

# Validate that a value is a positive integer.
is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# Read package/core/node metadata for one CPU, printing "package,core,node".
cpu_topology_triplet() {
  local cpu="$1"
  local pkg core node node_path

  pkg="NA"
  core="NA"
  node="NA"

  [[ -r "/sys/devices/system/cpu/cpu${cpu}/topology/physical_package_id" ]] && pkg="$(cat "/sys/devices/system/cpu/cpu${cpu}/topology/physical_package_id")"
  [[ -r "/sys/devices/system/cpu/cpu${cpu}/topology/core_id" ]] && core="$(cat "/sys/devices/system/cpu/cpu${cpu}/topology/core_id")"

  node_path="$(ls -d "/sys/devices/system/cpu/cpu${cpu}"/node* 2>/dev/null | head -n1 || true)"
  if [[ -n "$node_path" ]]; then
    node="${node_path##*node}"
  fi

  echo "${pkg},${core},${node}"
}

# Print selected CPU topology and SMT sibling details when PIN_DEBUG=1.
emit_pinning_debug_info() {
  local allowed_file="$1"
  local selected_file="$2"
  local smt_active cpu topo pkg core node siblings

  [[ "${PIN_DEBUG:-0}" == "1" ]] || return 0

  smt_active="NA"
  [[ -r /sys/devices/system/cpu/smt/active ]] && smt_active="$(cat /sys/devices/system/cpu/smt/active)"

  echo "🔎 [pinning] debug: smt_active=${smt_active}"
  echo "🔎 [pinning] debug: allowed_cpus=$(paste -sd, "$allowed_file")"
  echo "🔎 [pinning] debug: selected_cpus=$(paste -sd, "$selected_file")"

  while IFS= read -r cpu; do
    topo="$(cpu_topology_triplet "$cpu")"
    IFS=',' read -r pkg core node <<< "$topo"
    siblings="NA"
    [[ -r "/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list" ]] && siblings="$(cat "/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list")"
    echo "🔎 [pinning] cpu=${cpu} package=${pkg} core=${core} node=${node} siblings=${siblings}"
  done < "$selected_file"
}

# Warn or fail when selected CPUs include SMT siblings from the same physical core.
validate_distinct_physical_cores_or_warn() {
  local selected_file="$1"
  local strict="${PIN_STRICT_PHYSICAL_CORES:-0}"
  local cpu topo pkg core node key first_cpu
  local conflicts=""
  declare -A seen_core_to_cpu

  [[ -r /sys/devices/system/cpu/cpu0/topology/core_id ]] || return 0

  while IFS= read -r cpu; do
    topo="$(cpu_topology_triplet "$cpu")"
    IFS=',' read -r pkg core node <<< "$topo"
    key="${pkg}:${core}"
    if [[ -n "${seen_core_to_cpu[$key]:-}" ]]; then
      first_cpu="${seen_core_to_cpu[$key]}"
      conflicts+="${pkg}/core${core}=cpu${first_cpu},cpu${cpu};"
    else
      seen_core_to_cpu["$key"]="$cpu"
    fi
  done < "$selected_file"

  if [[ -n "$conflicts" ]]; then
    if [[ "$strict" == "1" ]]; then
      pinning_die "Selected CPUs include SMT siblings on the same physical core (${conflicts}). Choose one thread per core or set PIN_STRICT_PHYSICAL_CORES=0."
    fi
    echo "⚠️ [pinning] Selected CPUs include SMT siblings on the same physical core: ${conflicts}" >&2
  fi
}

# Parse and normalize CPU_SET syntax (integers and ascending ranges only).
expand_cpu_set_strict() {
  local spec="${1//[[:space:]]/}"
  local part start end
  [[ -n "$spec" ]] || pinning_die "CPU_SET cannot be empty in PIN_MODE=cpu_set."

  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    if [[ "$part" =~ ^[0-9]+$ ]]; then
      echo "$part"
      continue
    fi
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      (( start <= end )) || pinning_die "Invalid CPU range '$part' in CPU_SET (descending ranges are not allowed)."
      seq "$start" "$end"
      continue
    fi
    pinning_die "Invalid token '$part' in CPU_SET. Use comma-separated ints/ranges, e.g. 0-7,16-23."
  done | sort -n -u
}

# Parse CORE_LIST syntax (explicit integer core IDs only).
expand_core_list_strict() {
  local spec="${1//[[:space:]]/}"
  local part
  [[ -n "$spec" ]] || pinning_die "CORE_LIST cannot be empty in PIN_MODE=core_list."

  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    [[ "$part" =~ ^[0-9]+$ ]] || pinning_die "Invalid token '$part' in CORE_LIST. Use explicit core ids only, e.g. 0,2,4,6."
    echo "$part"
  done | awk '!seen[$1]++ {print $1}'
}

# Return online CPU IDs from sysfs, with nproc fallback.
get_online_cpus() {
  if [[ -r /sys/devices/system/cpu/online ]]; then
    expand_cpu_set_strict "$(cat /sys/devices/system/cpu/online)"
  else
    seq 0 $(( $(nproc) - 1 ))
  fi
}

# Return CPUs allowed for the current process affinity.
get_allowed_cpus() {
  local affinity
  if command -v taskset >/dev/null 2>&1; then
    affinity="$(taskset -pc $$ 2>/dev/null | sed -E 's/.*: *//')"
    if [[ -n "$affinity" ]]; then
      expand_cpu_set_strict "$affinity"
      return 0
    fi
  fi
  get_online_cpus
}

# Ensure requested CPUs are all contained in the allowed CPU set.
validate_requested_cpus_or_die() {
  local requested_file="$1"
  local allowed_file="$2"
  local label="$3"
  local missing

  [[ -s "$requested_file" ]] || pinning_die "$label did not resolve to any CPUs."

  missing="$(
    awk '
      FNR==NR { allowed[$1]=1; next }
      !allowed[$1] { print $1 }
    ' "$allowed_file" "$requested_file" | paste -sd, -
  )"
  if [[ -n "$missing" ]]; then
    pinning_die "$label contains CPUs not allowed for this process: $missing"
  fi
}

# Select the first NUM_CPUS entries from allowed CPUs.
select_num_cpus_or_die() {
  local allowed_file="$1"
  local selected_file="$2"
  local allowed_count

  is_positive_int "$NUM_CPUS" || pinning_die "NUM_CPUS must be a positive integer for PIN_MODE=num_cpus (got '$NUM_CPUS')."
  allowed_count="$(wc -l < "$allowed_file" | tr -d ' ')"
  (( NUM_CPUS <= allowed_count )) || pinning_die "NUM_CPUS=$NUM_CPUS exceeds allowed CPUs ($allowed_count)."

  head -n "$NUM_CPUS" "$allowed_file" > "$selected_file"
  [[ -s "$selected_file" ]] || pinning_die "Could not select CPUs for NUM_CPUS=$NUM_CPUS."
}

# Validate and apply cpu_set mode selection.
select_cpu_set_or_die() {
  local allowed_file="$1"
  local selected_file="$2"
  local requested_file="$TMP_DIR/requested_cpu_set.txt"

  expand_cpu_set_strict "$CPU_SET" > "$requested_file"
  validate_requested_cpus_or_die "$requested_file" "$allowed_file" "CPU_SET"
  cp "$requested_file" "$selected_file"
}

# Validate and apply core_list mode selection.
select_core_list_or_die() {
  local allowed_file="$1"
  local selected_file="$2"
  local requested_file="$TMP_DIR/requested_core_list.txt"

  expand_core_list_strict "$CORE_LIST" > "$requested_file"
  validate_requested_cpus_or_die "$requested_file" "$allowed_file" "CORE_LIST"
  cp "$requested_file" "$selected_file"
}

# Ensure only mode-relevant pinning options are provided.
validate_mode_inputs_or_die() {
  local cpu_set_trimmed="${CPU_SET//[[:space:]]/}"
  local core_list_trimmed="${CORE_LIST//[[:space:]]/}"

  case "$PIN_MODE" in
    num_cpus)
      [[ -z "$cpu_set_trimmed" ]] || pinning_die "PIN_MODE=num_cpus cannot be used with CPU_SET."
      [[ -z "$core_list_trimmed" ]] || pinning_die "PIN_MODE=num_cpus cannot be used with CORE_LIST."
      ;;
    cpu_set)
      [[ -n "$cpu_set_trimmed" ]] || pinning_die "PIN_MODE=cpu_set requires CPU_SET."
      [[ -z "$core_list_trimmed" ]] || pinning_die "PIN_MODE=cpu_set cannot be used with CORE_LIST."
      ;;
    core_list)
      [[ -n "$core_list_trimmed" ]] || pinning_die "PIN_MODE=core_list requires CORE_LIST."
      [[ -z "$cpu_set_trimmed" ]] || pinning_die "PIN_MODE=core_list cannot be used with CPU_SET."
      ;;
    none)
      [[ -z "$cpu_set_trimmed" ]] || pinning_die "PIN_MODE=none cannot be used with CPU_SET."
      [[ -z "$core_list_trimmed" ]] || pinning_die "PIN_MODE=none cannot be used with CORE_LIST."
      ;;
  esac
}

# Build strict pinning plan and export TASKSET_CPU_LIST/RUN_CORE_COUNT/PINNING_DESC.
prepare_taskset_pinning() {
  local allowed_file="$TMP_DIR/allowed_cpus.txt"
  local selected_file="$TMP_DIR/pin_selected.txt"
  local mode="$PIN_MODE"
  local allowed_count

  TASKSET_CPU_LIST=""
  PINNING_DESC="none"
  RUN_CORE_COUNT=1

  case "$mode" in
    num_cpus|cpu_set|core_list|none) ;;
    *) pinning_die "Unsupported PIN_MODE='$mode'. Use one of: num_cpus, cpu_set, core_list, none." ;;
  esac

  validate_mode_inputs_or_die

  if [[ "$mode" != "none" ]] && ! command -v taskset >/dev/null 2>&1; then
    pinning_die "taskset is required for PIN_MODE=$mode."
  fi

  get_allowed_cpus > "$allowed_file"
  [[ -s "$allowed_file" ]] || pinning_die "Could not detect allowed CPUs."

  allowed_count="$(wc -l < "$allowed_file" | tr -d ' ')"
  is_positive_int "$allowed_count" || pinning_die "Detected invalid allowed CPU count: $allowed_count"

  if [[ "$mode" == "none" ]]; then
    RUN_CORE_COUNT="$allowed_count"
    PINNING_DESC="none selected_cores=${RUN_CORE_COUNT}"
    echo "📌 CPU pinning disabled: $PINNING_DESC"
    return 0
  fi

  case "$mode" in
    num_cpus)
      select_num_cpus_or_die "$allowed_file" "$selected_file"
      PINNING_DESC="num_cpus(NUM_CPUS=$NUM_CPUS)"
      ;;
    cpu_set)
      select_cpu_set_or_die "$allowed_file" "$selected_file"
      PINNING_DESC="cpu_set(CPU_SET=$CPU_SET)"
      ;;
    core_list)
      select_core_list_or_die "$allowed_file" "$selected_file"
      PINNING_DESC="core_list(CORE_LIST=$CORE_LIST)"
      ;;
  esac

  RUN_CORE_COUNT="$(wc -l < "$selected_file" | tr -d ' ')"
  is_positive_int "$RUN_CORE_COUNT" || pinning_die "Selected invalid core count: $RUN_CORE_COUNT"

  TASKSET_CPU_LIST="$(paste -sd, "$selected_file")"
  [[ -n "$TASKSET_CPU_LIST" ]] || pinning_die "Selected CPU list is empty for PIN_MODE=$mode."

  validate_distinct_physical_cores_or_warn "$selected_file"
  emit_pinning_debug_info "$allowed_file" "$selected_file"

  PINNING_DESC="$PINNING_DESC selected_cores=${RUN_CORE_COUNT} cpus=${TASKSET_CPU_LIST}"
  echo "📌 CPU pinning enabled: $PINNING_DESC"
}
