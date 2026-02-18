#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".wma", ".opus"}


def list_audio(audio_dir: str):
    p = Path(audio_dir)
    files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXTS]
    files.sort(key=lambda x: str(x).lower())
    return files


def ensure_whisper():
    try:
        import whisper  # type: ignore

        return whisper
    except Exception:
        print("ERROR: python package 'whisper' not found (OpenAI Whisper).", file=sys.stderr)
        print("Install it, e.g.: pip install -U openai-whisper", file=sys.stderr)
        raise


def baseline_transcribe(audio_dir: str, out_dir: str, model_name: str, language: str):
    whisper = ensure_whisper()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    files = list_audio(audio_dir)
    if not files:
        raise SystemExit(f"No audio files found under: {audio_dir}")

    model = whisper.load_model(model_name)
    all_text_parts = []

    for f in files:
        res = model.transcribe(
            str(f),
            language=language,
            task="transcribe",
            fp16=False,
            verbose=False,
        )
        text = (res.get("text") or "").strip()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f.name)
        (outp / f"{safe_name}.txt").write_text(text + "\n", encoding="utf-8")
        all_text_parts.append(text)

    (outp / "baseline_all.txt").write_text("\n".join(all_text_parts).strip() + "\n", encoding="utf-8")


def normalize_for_words(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_for_chars(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", "", s)
    return s


def levenshtein(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def wer_cer(ref_text: str, hyp_text: str):
    ref_w = normalize_for_words(ref_text).split()
    hyp_w = normalize_for_words(hyp_text).split()
    if len(ref_w) == 0:
        wer = 0.0 if len(hyp_w) == 0 else 1.0
    else:
        wer = levenshtein(ref_w, hyp_w) / float(len(ref_w))

    ref_c = list(normalize_for_chars(ref_text))
    hyp_c = list(normalize_for_chars(hyp_text))
    if len(ref_c) == 0:
        cer = 0.0 if len(hyp_c) == 0 else 1.0
    else:
        cer = levenshtein(ref_c, hyp_c) / float(len(ref_c))

    return wer, cer


def extract_text(raw: str) -> str:
    lines = raw.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^(INFO|WARN|WARNING|DEBUG|TRACE)\b", s):
            continue
        if re.search(r"\b(elapsed|wall clock|warmup|max resident|tokens\/s|rtf|throughput)\b", s, re.I):
            continue
        if re.match(r"^\[?\d{1,2}:\d{2}(:\d{2})?(\.\d+)?", s):
            continue
        s = re.sub(r"^(file|audio|input)\s*:\s*", "", s, flags=re.I)
        kept.append(s)
    return "\n".join(kept).strip()


def extract_text_from_csv(csv_path: str) -> str:
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [x.strip() for x in (reader.fieldnames or [])]
        if not fieldnames:
            raise SystemExit(f"CSV has no header: {csv_path}")

        candidates = ["transcription", "trascription", "text"]
        text_col = None
        for c in candidates:
            if c in fieldnames:
                text_col = c
                break
        if text_col is None:
            raise SystemExit(
                f"No transcription column found in {csv_path}. "
                f"Expected one of: {', '.join(candidates)}. "
                f"Found: {', '.join(fieldnames)}"
            )

        rows = list(reader)
        rows.sort(key=lambda r: (r.get("file") or "").lower())
        parts = []
        for row in rows:
            t = (row.get(text_col) or "").strip()
            parts.append(t)
        return "\n".join(parts).strip()


def average_end_to_end_latency(csv_path: str) -> float:
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            v = (row.get("end_to_end_s") or "").strip()
            if not v:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    if not vals:
        raise SystemExit(f"No valid end_to_end_s values in: {csv_path}")
    return sum(vals) / len(vals)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: discover_optimal_model_metrics.py <cmd> ...")

    cmd = sys.argv[1]
    if cmd == "baseline":
        audio_dir, out_dir, model_name, lang = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        baseline_transcribe(audio_dir, out_dir, model_name, lang)
        return
    if cmd == "metrics":
        ref_path, hyp_raw_path, hyp_clean_out = sys.argv[2], sys.argv[3], sys.argv[4]
        ref = Path(ref_path).read_text(encoding="utf-8", errors="ignore")
        hyp_raw = Path(hyp_raw_path).read_text(encoding="utf-8", errors="ignore")
        hyp = extract_text(hyp_raw)
        Path(hyp_clean_out).write_text(hyp + "\n", encoding="utf-8")
        wer, cer = wer_cer(ref, hyp)
        print(f"{wer:.6f},{cer:.6f}")
        return
    if cmd == "metrics_csv":
        ref_path, hyp_csv_path, hyp_clean_out = sys.argv[2], sys.argv[3], sys.argv[4]
        ref = Path(ref_path).read_text(encoding="utf-8", errors="ignore")
        hyp = extract_text_from_csv(hyp_csv_path)
        Path(hyp_clean_out).write_text(hyp + "\n", encoding="utf-8")
        wer, cer = wer_cer(ref, hyp)
        print(f"{wer:.6f},{cer:.6f}")
        return
    if cmd == "avg_latency_csv":
        csv_path = sys.argv[2]
        avg = average_end_to_end_latency(csv_path)
        print(f"{avg:.6f}")
        return
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
