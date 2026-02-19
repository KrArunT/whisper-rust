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


@dataclasses.dataclass(frozen=True)
class Config:
    models_root: Path
    audio_dir: Path
    results_root: Path
    num_beams: int
    benchmark_lang: str
    rust_chunk_parallelism: int
    model_progress_interval_sec: int
    baseline_model: str
    baseline_lang: str


@dataclasses.dataclass(frozen=True)
class CpuAffinityContext:
    run_core_count: int
    cpu_list_desc: str


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


def load_config() -> Config:
    models_root = Path(os.getenv("MODELS_ROOT", "models/whisper-base-optimized"))
    audio_dir = Path(os.getenv("AUDIO_DIR", "audio"))
    results_root = Path(os.getenv("RESULTS_ROOT", "results/benchmarks/without_hf_pipeline_rust"))

    cfg = Config(
        models_root=models_root,
        audio_dir=audio_dir,
        results_root=results_root,
        num_beams=parse_int_env("NUM_BEAMS", 1),
        benchmark_lang=os.getenv("BENCHMARK_LANG", "en"),
        rust_chunk_parallelism=parse_int_env("RUST_CHUNK_PARALLELISM", 1),
        model_progress_interval_sec=parse_int_env("MODEL_PROGRESS_INTERVAL_SEC", 30),
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


def detect_cpu_affinity_context() -> CpuAffinityContext:
    cpus: list[int]
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = list(range(os.cpu_count() or 1))
    if not cpus:
        cpus = [0]
    if len(cpus) > 32:
        desc = ",".join(str(c) for c in cpus[:32]) + ",..."
    else:
        desc = ",".join(str(c) for c in cpus)
    return CpuAffinityContext(run_core_count=len(cpus), cpu_list_desc=desc)


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
    cpu_context: CpuAffinityContext,
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
        cpu_context.run_core_count, cfg.rust_chunk_parallelism
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
    cpu_context: CpuAffinityContext,
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
    report_lines.append(
        f"**Container CPU affinity:** `selected_cores={cpu_context.run_core_count} cpus={cpu_context.cpu_list_desc}`"
    )
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


def run_benchmark() -> int:
    cfg = load_config()
    validate_benchmark_inputs(cfg)
    cpu_context = detect_cpu_affinity_context()
    print(
        "INFO: Container CPU affinity context: "
        f"selected_cores={cpu_context.run_core_count} cpus={cpu_context.cpu_list_desc}"
    )
    os.environ["RUN_CORE_COUNT"] = str(cpu_context.run_core_count)

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
            rows.append(run_model_benchmark(cfg, runtime, cpu_context, metrics_script, onnx_dir))

        best = select_best_models(rows)
        write_markdown_report(
            runtime=runtime,
            baseline=baseline,
            cpu_context=cpu_context,
            rows=rows,
            best=best,
            baseline_model=cfg.baseline_model,
        )

        print("INFO: Benchmark completed")
        print(f"INFO: CSV   : {runtime.merged_model_csv}")
        print(f"INFO: Report: {runtime.report_md}")
        print(f"INFO: Baseline transcripts: {runtime.baseline_dir}")
        print(
            "INFO: CPU affinity: "
            f"selected_cores={cpu_context.run_core_count} cpus={cpu_context.cpu_list_desc}"
        )
        return 0
    finally:
        cleanup_runtime_paths(runtime)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover optimal whisper model benchmark runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run the full benchmark workflow.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_benchmark()
        parser.error("Unknown command")
        return 2
    except BenchmarkError as exc:
        die(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
