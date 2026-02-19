#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

from discover_optimal_model_metrics import average_end_to_end_latency, list_audio


class BenchmarkError(RuntimeError):
    pass


class PinningError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Config:
    models_root: Path
    audio_dir: Path
    results_root: Path
    pin_mode: str
    num_cpus_raw: str
    cpu_set: str
    core_list: str
    num_beams: int
    benchmark_lang: str
    rust_chunk_parallelism: int
    model_progress_interval_sec: int
    pin_debug: bool
    pin_strict_physical_cores: bool
    baseline_model: str
    baseline_lang: str


@dataclasses.dataclass(frozen=True)
class PinningPlan:
    taskset_cpu_list: str
    run_core_count: int
    pinning_desc: str


@dataclasses.dataclass(frozen=True)
class RuntimePaths:
    tmp_dir: Path
    merged_model_csv: Path
    report_md: Path
    per_model_results_dir: Path
    baseline_dir: Path
    baseline_all_txt: Path
    baseline_metrics_env: Path
    baseline_time_log: Path


@dataclasses.dataclass(frozen=True)
class BaselineMetrics:
    audio_count: int
    total_sec: str
    avg_sec: str
    peak_mb: str
    wer: str
    cer: str


@dataclasses.dataclass(frozen=True)
class BestRows:
    latency: dict[str, str] | None
    memory: dict[str, str] | None
    wer: dict[str, str] | None
    cer: dict[str, str] | None


MERGED_CSV_HEADER = [
    "implementation",
    "precision",
    "optimization",
    "instruction_set",
    "beam_size",
    "time_sec",
    "ram_mb",
    "wer",
    "cer",
]


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise BenchmarkError(f"{name} must be an integer (got '{raw}')") from exc


def parse_flag_env(name: str, default: str) -> bool:
    raw = os.getenv(name, default).strip()
    return raw == "1"


def resolve_pin_mode(cpu_set: str, core_list: str) -> str:
    pin_mode_input = os.getenv("PIN_MODE", "").strip()
    if pin_mode_input:
        return re.sub(r"\s+", "", pin_mode_input.lower())

    if os.getenv("ENABLE_TASKSET_PINNING", "1") != "1":
        return "none"
    if cpu_set.strip():
        return "cpu_set"
    if core_list.strip():
        return "core_list"
    return "num_cpus"


def load_config() -> Config:
    models_root = Path(os.getenv("MODELS_ROOT", "models/whisper-base-optimized"))
    audio_dir = Path(os.getenv("AUDIO_DIR", "audio"))
    results_root = Path(os.getenv("RESULTS_ROOT", "results/benchmarks/without_hf_pipeline_rust"))

    cpu_set = os.getenv("CPU_SET", os.getenv("CPUSET_LIST", os.getenv("PIN_CPUS", "")))
    core_list = os.getenv("CORE_LIST", "")

    cfg = Config(
        models_root=models_root,
        audio_dir=audio_dir,
        results_root=results_root,
        pin_mode=resolve_pin_mode(cpu_set, core_list),
        num_cpus_raw=os.getenv("NUM_CPUS", os.getenv("MAX_CORES_PER_RUN", "8")),
        cpu_set=cpu_set,
        core_list=core_list,
        num_beams=parse_int_env("NUM_BEAMS", 1),
        benchmark_lang=os.getenv("BENCHMARK_LANG", "en"),
        rust_chunk_parallelism=parse_int_env("RUST_CHUNK_PARALLELISM", 1),
        model_progress_interval_sec=parse_int_env("MODEL_PROGRESS_INTERVAL_SEC", 30),
        pin_debug=parse_flag_env("PIN_DEBUG", "0"),
        pin_strict_physical_cores=parse_flag_env("PIN_STRICT_PHYSICAL_CORES", "0"),
        baseline_model=os.getenv("BASELINE_MODEL", "base"),
        baseline_lang=os.getenv("BASELINE_LANG", "en"),
    )
    return cfg


def require_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise BenchmarkError(f"Required command not found: {cmd}")


def validate_benchmark_inputs(cfg: Config) -> None:
    if not cfg.audio_dir.is_dir():
        raise BenchmarkError(f"AUDIO_DIR does not exist: {cfg.audio_dir}")
    if not cfg.models_root.is_dir():
        raise BenchmarkError(f"MODELS_ROOT does not exist: {cfg.models_root}")
    model_dirs = [p for p in cfg.models_root.iterdir() if p.is_dir()]
    if not model_dirs:
        raise BenchmarkError(f"No model directories found in: {cfg.models_root}")

    require_command("uv")
    require_command("cargo")
    if not Path("/usr/bin/time").is_file():
        raise BenchmarkError("Required executable missing: /usr/bin/time")


def backup_old_results(runtime: RuntimePaths) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = runtime.per_model_results_dir.parent / "backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied = False
    candidates = [
        runtime.merged_model_csv,
        runtime.report_md,
        runtime.per_model_results_dir,
        runtime.baseline_dir,
    ]
    for src in candidates:
        if not src.exists():
            continue
        copied = True
        dst = backup_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    if copied:
        print(f"INFO: Backed up previous benchmark results to: {backup_dir}")
    else:
        backup_dir.rmdir()


def initialize_runtime_paths(cfg: Config) -> RuntimePaths:
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="discover_optimal_model_"))
    baseline_dir = cfg.results_root / f"baseline_whisper_{cfg.baseline_model}"
    return RuntimePaths(
        tmp_dir=tmp_dir,
        merged_model_csv=cfg.results_root / "merged_inference_results.csv",
        report_md=cfg.results_root / "BENCHMARK_REPORT.md",
        per_model_results_dir=cfg.results_root / "per_model",
        baseline_dir=baseline_dir,
        baseline_all_txt=baseline_dir / "baseline_all.txt",
        baseline_metrics_env=baseline_dir / "baseline_metrics.env",
        baseline_time_log=tmp_dir / "time_baseline.txt",
    )


def cleanup_runtime_paths(runtime: RuntimePaths) -> None:
    shutil.rmtree(runtime.tmp_dir, ignore_errors=True)


def parse_cpu_set_strict(spec: str) -> list[int]:
    cleaned = re.sub(r"\s+", "", spec)
    if not cleaned:
        raise PinningError("CPU_SET cannot be empty in PIN_MODE=cpu_set.")

    cpus: set[int] = set()
    for token in cleaned.split(","):
        if not token:
            continue
        if re.fullmatch(r"\d+", token):
            cpus.add(int(token))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                raise PinningError(
                    f"Invalid CPU range '{token}' in CPU_SET (descending ranges are not allowed)."
                )
            for value in range(start, end + 1):
                cpus.add(value)
            continue
        raise PinningError(
            f"Invalid token '{token}' in CPU_SET. Use comma-separated ints/ranges, e.g. 0-7,16-23."
        )

    return sorted(cpus)


def parse_core_list_strict(spec: str) -> list[int]:
    cleaned = re.sub(r"\s+", "", spec)
    if not cleaned:
        raise PinningError("CORE_LIST cannot be empty in PIN_MODE=core_list.")

    selected: list[int] = []
    seen: set[int] = set()
    for token in cleaned.split(","):
        if not token:
            continue
        if not re.fullmatch(r"\d+", token):
            raise PinningError(
                f"Invalid token '{token}' in CORE_LIST. Use explicit core ids only, e.g. 0,2,4,6."
            )
        value = int(token)
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
    return selected


def get_online_cpus() -> list[int]:
    online_path = Path("/sys/devices/system/cpu/online")
    if online_path.is_file():
        return parse_cpu_set_strict(online_path.read_text(encoding="utf-8").strip())
    return list(range(os.cpu_count() or 1))


def get_allowed_cpus() -> list[int]:
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            return allowed
    except AttributeError:
        pass
    return get_online_cpus()


def cpu_topology_triplet(cpu: int) -> tuple[str, str, str]:
    pkg = "NA"
    core = "NA"
    node = "NA"
    cpu_root = Path(f"/sys/devices/system/cpu/cpu{cpu}")

    pkg_file = cpu_root / "topology/physical_package_id"
    core_file = cpu_root / "topology/core_id"
    if pkg_file.is_file():
        pkg = pkg_file.read_text(encoding="utf-8").strip()
    if core_file.is_file():
        core = core_file.read_text(encoding="utf-8").strip()

    node_dirs = sorted(cpu_root.glob("node*"))
    if node_dirs:
        node = node_dirs[0].name.replace("node", "")

    return pkg, core, node


def emit_pinning_debug_info(allowed: Sequence[int], selected: Sequence[int], enabled: bool) -> None:
    if not enabled:
        return
    smt_active = "NA"
    smt_file = Path("/sys/devices/system/cpu/smt/active")
    if smt_file.is_file():
        smt_active = smt_file.read_text(encoding="utf-8").strip()

    print(f"DEBUG: [pinning] smt_active={smt_active}", file=sys.stderr)
    print(f"DEBUG: [pinning] allowed_cpus={','.join(str(c) for c in allowed)}", file=sys.stderr)
    print(f"DEBUG: [pinning] selected_cpus={','.join(str(c) for c in selected)}", file=sys.stderr)

    for cpu in selected:
        pkg, core, node = cpu_topology_triplet(cpu)
        siblings = "NA"
        siblings_file = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if siblings_file.is_file():
            siblings = siblings_file.read_text(encoding="utf-8").strip()
        print(
            f"DEBUG: [pinning] cpu={cpu} package={pkg} core={core} node={node} siblings={siblings}",
            file=sys.stderr,
        )


def validate_distinct_physical_cores_or_warn(
    selected: Sequence[int], strict: bool
) -> None:
    core_id_file = Path("/sys/devices/system/cpu/cpu0/topology/core_id")
    if not core_id_file.is_file():
        return

    seen_core: dict[str, int] = {}
    conflicts: list[str] = []
    for cpu in selected:
        pkg, core, _node = cpu_topology_triplet(cpu)
        key = f"{pkg}:{core}"
        if key in seen_core:
            conflicts.append(f"{pkg}/core{core}=cpu{seen_core[key]},cpu{cpu}")
        else:
            seen_core[key] = cpu

    if not conflicts:
        return
    message = (
        "Selected CPUs include SMT siblings on the same physical core "
        f"({';'.join(conflicts)})."
    )
    if strict:
        raise PinningError(
            f"{message} Choose one thread per core or set PIN_STRICT_PHYSICAL_CORES=0."
        )
    print(f"WARNING: [pinning] {message}", file=sys.stderr)


def validate_mode_inputs_or_die(pin_mode: str, cpu_set: str, core_list: str) -> None:
    cpu_set_trimmed = re.sub(r"\s+", "", cpu_set)
    core_list_trimmed = re.sub(r"\s+", "", core_list)

    if pin_mode == "num_cpus":
        if cpu_set_trimmed:
            raise PinningError("PIN_MODE=num_cpus cannot be used with CPU_SET.")
        if core_list_trimmed:
            raise PinningError("PIN_MODE=num_cpus cannot be used with CORE_LIST.")
        return
    if pin_mode == "cpu_set":
        if not cpu_set_trimmed:
            raise PinningError("PIN_MODE=cpu_set requires CPU_SET.")
        if core_list_trimmed:
            raise PinningError("PIN_MODE=cpu_set cannot be used with CORE_LIST.")
        return
    if pin_mode == "core_list":
        if not core_list_trimmed:
            raise PinningError("PIN_MODE=core_list requires CORE_LIST.")
        if cpu_set_trimmed:
            raise PinningError("PIN_MODE=core_list cannot be used with CPU_SET.")
        return
    if pin_mode == "none":
        if cpu_set_trimmed:
            raise PinningError("PIN_MODE=none cannot be used with CPU_SET.")
        if core_list_trimmed:
            raise PinningError("PIN_MODE=none cannot be used with CORE_LIST.")
        return

    raise PinningError(
        f"Unsupported PIN_MODE='{pin_mode}'. Use one of: num_cpus, cpu_set, core_list, none."
    )


def parse_positive_int(name: str, value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise PinningError(f"{name} must be a positive integer (got '{value}').")
    return int(value)


def prepare_taskset_pinning(cfg: Config) -> PinningPlan:
    validate_mode_inputs_or_die(cfg.pin_mode, cfg.cpu_set, cfg.core_list)

    if cfg.pin_mode != "none" and shutil.which("taskset") is None:
        raise PinningError(f"taskset is required for PIN_MODE={cfg.pin_mode}.")

    allowed = get_allowed_cpus()
    if not allowed:
        raise PinningError("Could not detect allowed CPUs.")

    if cfg.pin_mode == "none":
        run_core_count = len(allowed)
        desc = f"none selected_cores={run_core_count}"
        return PinningPlan(taskset_cpu_list="", run_core_count=run_core_count, pinning_desc=desc)

    if cfg.pin_mode == "num_cpus":
        num_cpus = parse_positive_int("NUM_CPUS", cfg.num_cpus_raw.strip())
        if num_cpus > len(allowed):
            raise PinningError(
                f"NUM_CPUS={num_cpus} exceeds allowed CPUs ({len(allowed)})."
            )
        selected = allowed[:num_cpus]
        mode_desc = f"num_cpus(NUM_CPUS={num_cpus})"
    elif cfg.pin_mode == "cpu_set":
        requested = parse_cpu_set_strict(cfg.cpu_set)
        missing = [cpu for cpu in requested if cpu not in set(allowed)]
        if missing:
            raise PinningError(
                "CPU_SET contains CPUs not allowed for this process: "
                + ",".join(str(v) for v in missing)
            )
        selected = requested
        mode_desc = f"cpu_set(CPU_SET={cfg.cpu_set})"
    else:
        requested = parse_core_list_strict(cfg.core_list)
        missing = [cpu for cpu in requested if cpu not in set(allowed)]
        if missing:
            raise PinningError(
                "CORE_LIST contains CPUs not allowed for this process: "
                + ",".join(str(v) for v in missing)
            )
        selected = requested
        mode_desc = f"core_list(CORE_LIST={cfg.core_list})"

    if not selected:
        raise PinningError(f"Selected CPU list is empty for PIN_MODE={cfg.pin_mode}.")

    validate_distinct_physical_cores_or_warn(selected, strict=cfg.pin_strict_physical_cores)
    emit_pinning_debug_info(allowed, selected, enabled=cfg.pin_debug)

    selected_str = ",".join(str(cpu) for cpu in selected)
    desc = f"{mode_desc} selected_cores={len(selected)} cpus={selected_str}"
    return PinningPlan(taskset_cpu_list=selected_str, run_core_count=len(selected), pinning_desc=desc)


def elapsed_wall_time_from_log(time_log: Path) -> float:
    if not time_log.is_file():
        return 0.0
    pattern = re.compile(r"Elapsed \(wall clock\) time.*:\s*(\S+)\s*$")
    with time_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return time_to_seconds(match.group(1))
    return 0.0


def peak_memory_mb_from_log(time_log: Path) -> int:
    if not time_log.is_file():
        return 0
    pattern = re.compile(r"Maximum resident set size.*:\s*(\d+)\s*$")
    with time_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                peak_kb = int(match.group(1))
                return int(round(peak_kb / 1024.0))
    return 0


def time_to_seconds(token: str) -> float:
    parts = token.strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def pretty_time(value: str | float | int) -> str:
    if value in ("", "NA", None):
        return "NA"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "NA"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}m{remainder:.2f}s"


def pretty_score(value: str | float | int) -> str:
    if value in ("", "NA", None):
        return "NA"
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{ratio * 100.0:.2f}%"


def precision_from_model(model_name: str) -> str:
    return "int8" if "int8" in model_name else "fp32"


def optimization_from_model(model_name: str) -> str:
    match = re.search(r"(o[0-9]+)", model_name)
    return match.group(1) if match else "unknown"


def instruction_set_from_model(model_name: str) -> str:
    match = re.search(r"_int8_([^_]+)$", model_name)
    return match.group(1) if match else "NA"


def implementation_name() -> str:
    return "onnxruntime rust"


def compute_threading_plan(run_core_count: int, desired_chunk: int) -> tuple[int, int]:
    if run_core_count < 1:
        raise BenchmarkError(f"Invalid RUN_CORE_COUNT: {run_core_count}")
    if desired_chunk < 1:
        raise BenchmarkError(
            f"RUST_CHUNK_PARALLELISM must be a positive integer (got '{desired_chunk}')"
        )
    model_chunk = min(desired_chunk, run_core_count)
    model_intra = run_core_count // model_chunk
    if model_intra < 1:
        model_intra = 1
    return model_chunk, model_intra


def run_timed_command(
    cmd: Sequence[str],
    time_log: Path,
    *,
    stderr_log: Path | None = None,
    quiet_stdout: bool = False,
    progress_interval_sec: int = 0,
    run_label: str = "command",
) -> None:
    full_cmd = ["/usr/bin/time", "-v", "-o", str(time_log), *cmd]
    stderr_handle = None
    try:
        if stderr_log is not None:
            stderr_handle = stderr_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.DEVNULL if quiet_stdout else None,
            stderr=stderr_handle if stderr_handle is not None else None,
        )
        started = time.time()
        if progress_interval_sec > 0:
            while True:
                try:
                    return_code = process.wait(timeout=progress_interval_sec)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.time() - started)
                    print(f"INFO: {run_label} still running ({elapsed}s elapsed)")
        else:
            return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, full_cmd)
    finally:
        if stderr_handle is not None:
            stderr_handle.close()


def write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def generate_baseline_transcripts(
    cfg: Config,
    runtime: RuntimePaths,
    metrics_script: Path,
) -> BaselineMetrics:
    baseline_cache_hit = runtime.baseline_all_txt.is_file() and runtime.baseline_all_txt.stat().st_size > 0
    if baseline_cache_hit:
        print(f"INFO: Reusing existing baseline transcripts: {runtime.baseline_all_txt}")
    else:
        print(
            f"INFO: Generating baseline transcripts with OpenAI Whisper '{cfg.baseline_model}'..."
        )
        run_timed_command(
            [
                "uv",
                "run",
                "python",
                str(metrics_script),
                "baseline",
                str(cfg.audio_dir),
                str(runtime.baseline_dir),
                cfg.baseline_model,
                cfg.baseline_lang,
            ],
            runtime.baseline_time_log,
        )

    if not runtime.baseline_all_txt.is_file():
        raise BenchmarkError(f"Baseline transcript missing: {runtime.baseline_all_txt}")

    baseline_audio_count = len(list_audio(str(cfg.audio_dir)))
    if baseline_audio_count <= 0:
        raise BenchmarkError(f"No supported audio files found under: {cfg.audio_dir}")

    if not baseline_cache_hit:
        baseline_total_sec = elapsed_wall_time_from_log(runtime.baseline_time_log)
        baseline_avg_sec = baseline_total_sec / baseline_audio_count if baseline_audio_count else 0.0
        baseline_peak_mb = peak_memory_mb_from_log(runtime.baseline_time_log)
        write_env_file(
            runtime.baseline_metrics_env,
            {
                "BASELINE_TOTAL_SEC": f"{baseline_total_sec:.6f}",
                "BASELINE_AVG_SEC": f"{baseline_avg_sec:.6f}",
                "BASELINE_PEAK_MB": str(baseline_peak_mb),
            },
        )
        total_sec = f"{baseline_total_sec:.6f}"
        avg_sec = f"{baseline_avg_sec:.6f}"
        peak_mb = str(baseline_peak_mb)
    else:
        cached = read_env_file(runtime.baseline_metrics_env)
        total_sec = cached.get("BASELINE_TOTAL_SEC", "NA")
        avg_sec = cached.get("BASELINE_AVG_SEC", "NA")
        peak_mb = cached.get("BASELINE_PEAK_MB", "NA")

    print(f"INFO: Baseline ready: {runtime.baseline_all_txt}")
    return BaselineMetrics(
        audio_count=baseline_audio_count,
        total_sec=total_sec,
        avg_sec=avg_sec,
        peak_mb=peak_mb,
        wer="0.000000",
        cer="0.000000",
    )


def compute_metrics_csv(
    metrics_script: Path,
    ref_path: Path,
    hyp_csv_path: Path,
    hyp_clean_out: Path,
) -> tuple[str, str] | None:
    cmd = [
        "uv",
        "run",
        "python",
        str(metrics_script),
        "metrics_csv",
        str(ref_path),
        str(hyp_csv_path),
        str(hyp_clean_out),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    output = completed.stdout.strip()
    if not output:
        return None
    parts = output.split(",")
    if len(parts) != 2:
        return None
    wer = parts[0].strip()
    cer = parts[1].strip()
    return wer, cer


def initialize_merged_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MERGED_CSV_HEADER)


def append_merged_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([row[h] for h in MERGED_CSV_HEADER])


def model_directories(models_root: Path) -> Iterable[Path]:
    dirs = [p for p in models_root.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name)


def run_model_benchmark(
    cfg: Config,
    runtime: RuntimePaths,
    pinning: PinningPlan,
    metrics_script: Path,
    onnx_dir: Path,
) -> dict[str, str]:
    model_name = onnx_dir.name
    print(f"INFO: Benchmarking {model_name}")

    time_log = runtime.tmp_dir / f"time_{model_name}.txt"
    hyp_clean = runtime.tmp_dir / f"hyp_clean_{model_name}.txt"
    run_err = runtime.tmp_dir / f"run_stderr_{model_name}.txt"
    model_tmp_dir = runtime.per_model_results_dir / model_name
    model_csv = model_tmp_dir / "inference_per_file.csv"
    model_json = model_tmp_dir / "inference_per_file.json"
    model_summary = model_tmp_dir / "inference_summary.json"
    model_tmp_dir.mkdir(parents=True, exist_ok=True)

    model_chunk_parallelism, model_intra_op = compute_threading_plan(
        pinning.run_core_count, cfg.rust_chunk_parallelism
    )

    run_timed_command(
        [
            "cargo",
            "run",
            "--release",
            "--",
            "--audio-dir",
            str(cfg.audio_dir),
            "--onnx-dir",
            str(onnx_dir),
            "--language",
            cfg.benchmark_lang,
            "--task",
            "transcribe",
            "--max-new-tokens",
            "128",
            "--num-beams",
            str(cfg.num_beams),
            "--intra-op",
            str(model_intra_op),
            "--inter-op",
            "1",
            "--chunk-parallelism",
            str(model_chunk_parallelism),
            "--warmup",
            "1",
            "--out-csv",
            str(model_csv),
            "--out-json",
            str(model_json),
            "--out-summary-json",
            str(model_summary),
        ],
        time_log,
        stderr_log=run_err,
        quiet_stdout=True,
        progress_interval_sec=cfg.model_progress_interval_sec,
        run_label=f"model {model_name}",
    )

    wall_time_sec = elapsed_wall_time_from_log(time_log)
    try:
        avg_time_sec = average_end_to_end_latency(str(model_csv))
        time_sec = f"{avg_time_sec:.6f}"
    except SystemExit:
        time_sec = f"{wall_time_sec:.6f}"

    peak_mb = str(peak_memory_mb_from_log(time_log))
    precision = precision_from_model(model_name)
    opt_label = optimization_from_model(model_name)
    isa_label = instruction_set_from_model(model_name)
    impl = implementation_name()

    print(f"INFO: Computing WER/CER for {model_name}...")
    metrics = compute_metrics_csv(
        metrics_script=metrics_script,
        ref_path=runtime.baseline_all_txt,
        hyp_csv_path=model_csv,
        hyp_clean_out=hyp_clean,
    )
    if metrics is None:
        if hyp_clean.is_file():
            print(
                f"WARNING: Metrics unavailable for {model_name} (error). "
                "Recording WER/CER as NA."
            )
        wer = "NA"
        cer = "NA"
    else:
        wer, cer = metrics

    row = {
        "implementation": impl,
        "precision": precision,
        "optimization": opt_label,
        "instruction_set": isa_label,
        "beam_size": str(cfg.num_beams),
        "time_sec": time_sec,
        "ram_mb": peak_mb,
        "wer": wer,
        "cer": cer,
    }
    append_merged_row(runtime.merged_model_csv, row)
    return row


def to_float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_best_models(rows: list[dict[str, str]]) -> BestRows:
    best_latency = min(rows, key=lambda r: float(r["time_sec"])) if rows else None
    best_memory = min(rows, key=lambda r: float(r["ram_mb"])) if rows else None

    valid_wer = [r for r in rows if to_float_or_none(r["wer"]) is not None]
    valid_cer = [r for r in rows if to_float_or_none(r["cer"]) is not None]

    best_wer = min(valid_wer, key=lambda r: float(r["wer"])) if valid_wer else None
    best_cer = min(valid_cer, key=lambda r: float(r["cer"])) if valid_cer else None

    return BestRows(latency=best_latency, memory=best_memory, wer=best_wer, cer=best_cer)


def markdown_row_from_result(row: dict[str, str]) -> str:
    ram = row.get("ram_mb", "NA")
    if ram in ("", "NA"):
        ram_cell = "NA"
    else:
        ram_cell = f"{ram}MB"
    return (
        f"| {row['implementation']} | {row['precision']} | {row['optimization']} | "
        f"{row['instruction_set']} | {row['beam_size']} | {pretty_time(row['time_sec'])} | "
        f"{ram_cell} | {pretty_score(row['wer'])} | {pretty_score(row['cer'])} |"
    )


def write_markdown_report(
    runtime: RuntimePaths,
    baseline: BaselineMetrics,
    pinning: PinningPlan,
    rows: list[dict[str, str]],
    best: BestRows,
    baseline_model: str,
) -> None:
    baseline_peak = baseline.peak_mb
    baseline_ram_cell = "NA" if baseline_peak in ("", "NA") else f"{baseline_peak}MB"
    baseline_ram_fmt = baseline_ram_cell

    report_lines: list[str] = []
    report_lines.append("# Whisper ONNX Inference Benchmark")
    report_lines.append("")
    report_lines.append(
        f"**Baseline (accuracy reference):** OpenAI Whisper `{baseline_model}` via python `whisper` library"
    )
    report_lines.append(f"**CPU pinning:** `{pinning.pinning_desc}`")
    report_lines.append(
        "**Time column:** average end-to-end latency per audio from `inference_per_file.csv`"
    )
    report_lines.append("")
    report_lines.append("## Baseline Metrics")
    report_lines.append(f"- Files: **{baseline.audio_count}**")
    report_lines.append(f"- Time (total): **{pretty_time(baseline.total_sec)}**")
    report_lines.append(f"- Time (avg/audio): **{pretty_time(baseline.avg_sec)}**")
    report_lines.append(f"- RAM (peak): **{baseline_ram_fmt}**")
    report_lines.append(
        f"- WER/CER (vs baseline): **{pretty_score(baseline.wer)}** / **{pretty_score(baseline.cer)}**"
    )
    report_lines.append("")
    report_lines.append(
        "| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |"
    )
    report_lines.append(
        "|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|"
    )
    report_lines.append(
        f"| openai-whisper python | fp32 | baseline | NA | NA | {pretty_time(baseline.avg_sec)} | {baseline_ram_cell} | {pretty_score(baseline.wer)} | {pretty_score(baseline.cer)} |"
    )

    sorted_rows = sorted(
        rows,
        key=lambda r: (r["precision"], r["optimization"], r["instruction_set"]),
    )
    for row in sorted_rows:
        report_lines.append(markdown_row_from_result(row))

    report_lines.append("")
    report_lines.append("## Lowest Latency")
    if best.latency is None:
        report_lines.append("- **NA**")
    else:
        report_lines.append(f"- **{best.latency['implementation']}**")
        report_lines.append(f"- Optimization: **{best.latency['optimization']}**")
        report_lines.append(f"- Instruction set: **{best.latency['instruction_set']}**")
        report_lines.append(f"- Time: **{pretty_time(best.latency['time_sec'])}**")
        report_lines.append(
            f"- WER/CER: **{pretty_score(best.latency['wer'])}** / **{pretty_score(best.latency['cer'])}**"
        )

    report_lines.append("")
    report_lines.append("## Lowest Memory")
    if best.memory is None:
        report_lines.append("- **NA**")
    else:
        ram_value = best.memory["ram_mb"]
        ram_fmt = f"{ram_value}MB" if ram_value not in ("", "NA") else "NA"
        report_lines.append(f"- **{best.memory['implementation']}**")
        report_lines.append(f"- Optimization: **{best.memory['optimization']}**")
        report_lines.append(f"- Instruction set: **{best.memory['instruction_set']}**")
        report_lines.append(f"- RAM: **{ram_fmt}**")
        report_lines.append(
            f"- WER/CER: **{pretty_score(best.memory['wer'])}** / **{pretty_score(best.memory['cer'])}**"
        )

    report_lines.append("")
    report_lines.append("## Best Accuracy")
    if best.wer is None:
        report_lines.append("- Lowest WER: **NA** (no valid WER computed)")
    else:
        report_lines.append(
            f"- Lowest WER Optimization: **{best.wer['optimization']}** on **{best.wer['instruction_set']}** (WER **{pretty_score(best.wer['wer'])}**)"
        )
    if best.cer is None:
        report_lines.append("- Lowest CER: **NA** (no valid CER computed)")
    else:
        report_lines.append(
            f"- Lowest CER Optimization: **{best.cer['optimization']}** on **{best.cer['instruction_set']}** (CER **{pretty_score(best.cer['cer'])}**)"
        )

    runtime.report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def resolve_pinning_from_env(cfg: Config) -> PinningPlan | None:
    taskset_cpu_list = os.getenv("TASKSET_CPU_LIST", "")
    run_core_count_raw = os.getenv("RUN_CORE_COUNT", "")
    pinning_desc = os.getenv("PINNING_DESC", "")

    if not run_core_count_raw:
        return None
    try:
        run_core_count = int(run_core_count_raw)
    except ValueError:
        raise BenchmarkError(f"Invalid RUN_CORE_COUNT: {run_core_count_raw}")
    if run_core_count < 1:
        raise BenchmarkError(f"Invalid RUN_CORE_COUNT: {run_core_count_raw}")
    if not pinning_desc:
        pinning_desc = "none selected_cores={run_core_count}".format(
            run_core_count=run_core_count
        )
    if cfg.pin_mode != "none" and not taskset_cpu_list:
        raise BenchmarkError("TASKSET_CPU_LIST is empty but pinning mode is enabled.")
    return PinningPlan(
        taskset_cpu_list=taskset_cpu_list,
        run_core_count=run_core_count,
        pinning_desc=pinning_desc,
    )


def run_benchmark() -> int:
    cfg = load_config()
    validate_benchmark_inputs(cfg)

    pinning = resolve_pinning_from_env(cfg)
    if pinning is None:
        pinning = prepare_taskset_pinning(cfg)
        if pinning.taskset_cpu_list:
            print(f"INFO: CPU pinning enabled: {pinning.pinning_desc}")
        else:
            print(f"INFO: CPU pinning disabled: {pinning.pinning_desc}")
    else:
        print(f"INFO: CPU pinning context: {pinning.pinning_desc}")

    runtime = initialize_runtime_paths(cfg)
    metrics_script = Path(__file__).with_name("discover_optimal_model_metrics.py")
    if not metrics_script.is_file():
        raise BenchmarkError(f"Missing Python helper: {metrics_script}")

    try:
        backup_old_results(runtime)
        runtime.baseline_dir.mkdir(parents=True, exist_ok=True)
        runtime.per_model_results_dir.mkdir(parents=True, exist_ok=True)
        initialize_merged_csv(runtime.merged_model_csv)

        baseline = generate_baseline_transcripts(cfg, runtime, metrics_script)

        rows: list[dict[str, str]] = []
        for onnx_dir in model_directories(cfg.models_root):
            rows.append(run_model_benchmark(cfg, runtime, pinning, metrics_script, onnx_dir))

        best = select_best_models(rows)
        write_markdown_report(
            runtime=runtime,
            baseline=baseline,
            pinning=pinning,
            rows=rows,
            best=best,
            baseline_model=cfg.baseline_model,
        )

        print("INFO: Benchmark completed")
        print(f"INFO: CSV   : {runtime.merged_model_csv}")
        print(f"INFO: Report: {runtime.report_md}")
        print(f"INFO: Baseline transcripts: {runtime.baseline_dir}")
        print(f"INFO: Pinning: {pinning.pinning_desc}")
        return 0
    finally:
        cleanup_runtime_paths(runtime)


def print_pinning_plan_tsv() -> int:
    cfg = load_config()
    plan = prepare_taskset_pinning(cfg)
    print(plan.taskset_cpu_list)
    print(plan.run_core_count)
    print(plan.pinning_desc)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover optimal whisper model benchmark runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run the full benchmark workflow.")
    sub.add_parser("pinning-plan", help="Emit pinning plan as TSV (cpu_list, run_core_count, desc).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_benchmark()
        if args.command == "pinning-plan":
            return print_pinning_plan_tsv()
        parser.error("Unknown command")
        return 2
    except (BenchmarkError, PinningError) as exc:
        die(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
