use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use csv::Writer;
use ndarray::{Array2, Array3, ArrayD, Axis};
use ordered_float::OrderedFloat;
use ort::{
    AllocationDevice, AllocatorType, CPUExecutionProvider, MemoryInfo, MemoryType, Session, SessionBuilder,
    SessionInputValue, Tensor, Value,
};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

const SUPPORTED_AUDIO_EXTS: &[&str] = &["wav", "flac", "mp3", "m4a", "ogg", "webm", "aac", "wma", "opus"];

// -------------------------
// CLI
// -------------------------
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "audio")]
    audio_dir: String,

    #[arg(long, default_value = "openai/whisper-base")]
    model_id: String,

    #[arg(long, default_value = "whisper-base-with-past")]
    onnx_dir: String,

    #[arg(long, default_value = "en")]
    language: String,

    #[arg(long, default_value = "transcribe")]
    task: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 1)]
    num_beams: usize,

    #[arg(long, default_value_t = 0)]
    warmup: usize,

    #[arg(long, default_value_t = 0)]
    limit_files: usize,

    #[arg(long, default_value = "")]
    discovery_best_json: String,

    #[arg(long, default_value = "results/benchmarks/inference_per_file.csv")]
    out_csv: String,

    #[arg(long, default_value = "results/benchmarks/inference_per_file.json")]
    out_json: String,

    #[arg(long, default_value = "results/benchmarks/inference_summary.json")]
    out_summary_json: String,

    #[arg(long, default_value_t = 0)]
    intra_op: usize,

    #[arg(long, default_value_t = 0)]
    inter_op: usize,

    #[arg(long, default_value_t = false)]
    write_txt: bool,

    #[arg(long, default_value = "")]
    tokenizer_json: String,

    #[arg(long, default_value_t = false)]
    timestamps: bool,

    #[arg(long, default_value_t = 0)]
    chunk_parallelism: usize,

    // Chunking params (Rust long-form approximation)
    #[arg(long, default_value_t = 30.0)]
    chunk_length_s: f32,

    #[arg(long, default_value_t = 5.0)]
    overlap_s: f32,
}

// -------------------------
// Config (similar to your Python)
// -------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrtCfg {
    intra_op: usize,
    inter_op: usize,
    execution_mode: String, // "SEQUENTIAL" | "PARALLEL"
    graph_opt: String,      // "ENABLE_ALL" | "ENABLE_EXTENDED"
    cpu_mem_arena: bool,
    mem_pattern: bool,
    allow_spinning: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct GenerationCfg {
    suppress_tokens: Option<Vec<i64>>,
    begin_suppress_tokens: Option<Vec<i64>>,
}

fn suggested_optimum_cfg() -> OrtCfg {
    let cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let intra = cpu.min(16);
    OrtCfg {
        intra_op: intra,
        inter_op: 1,
        execution_mode: "SEQUENTIAL".to_string(),
        graph_opt: "ENABLE_ALL".to_string(),
        cpu_mem_arena: true,
        mem_pattern: true,
        allow_spinning: true,
    }
}

fn load_best_cfg_from_discovery(path: &str) -> Result<OrtCfg> {
    #[derive(Deserialize)]
    struct Outer {
        best: Option<HashMap<String, serde_json::Value>>,
    }
    let txt = fs::read_to_string(path)?;
    let outer: Outer = serde_json::from_str(&txt)?;
    let best = outer.best.unwrap_or_default();

    let get_bool = |k: &str, default: bool| -> bool {
        match best.get(k) {
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(0) != 0,
            Some(serde_json::Value::String(s)) => {
                matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "y" | "on")
            }
            _ => default,
        }
    };
    let get_usize = |k: &str, default: usize| -> usize {
        match best.get(k) {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(default as u64) as usize,
            Some(serde_json::Value::String(s)) => s.parse::<usize>().unwrap_or(default),
            _ => default,
        }
    };
    let get_string = |k: &str, default: &str| -> String {
        match best.get(k) {
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => default.to_string(),
        }
    };

    let fallback = suggested_optimum_cfg();
    Ok(OrtCfg {
        intra_op: get_usize("intra_op", fallback.intra_op),
        inter_op: get_usize("inter_op", 1),
        execution_mode: get_string("execution_mode", "SEQUENTIAL"),
        graph_opt: get_string("graph_opt", "ENABLE_ALL"),
        cpu_mem_arena: get_bool("cpu_mem_arena", true),
        mem_pattern: get_bool("mem_pattern", true),
        allow_spinning: get_bool("allow_spinning", true),
    })
}

fn build_session(model_path: &Path, cfg: &OrtCfg) -> Result<Session> {
    let mut builder = SessionBuilder::new()?;

    builder = builder
        .with_intra_threads(cfg.intra_op)?
        .with_inter_threads(cfg.inter_op)?;

    // Execution mode
    let parallel = cfg.execution_mode.to_uppercase() == "PARALLEL";
    builder = builder.with_parallel_execution(parallel)?;

    // Graph opt
    if cfg.graph_opt.to_uppercase() == "ENABLE_ALL" {
        builder = builder.with_optimization_level(ort::GraphOptimizationLevel::Level3)?;
    } else {
        builder = builder.with_optimization_level(ort::GraphOptimizationLevel::Level2)?;
    }

    // CPU arena / mem pattern
    builder = builder.with_memory_pattern(cfg.mem_pattern)?;
    let cpu_ep = if cfg.cpu_mem_arena {
        CPUExecutionProvider::default().with_arena_allocator().build()
    } else {
        CPUExecutionProvider::default().build()
    };
    builder = builder.with_execution_providers([cpu_ep])?;

    // Allow spinning
    builder = builder
        .with_intra_op_spinning(cfg.allow_spinning)?
        .with_inter_op_spinning(cfg.allow_spinning)?;

    Ok(builder.commit_from_file(model_path)?)
}

// -------------------------
// Audio decode + resample
// -------------------------
fn resample_linear(x: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out {
        return x.to_vec();
    }
    let ratio = sr_out as f64 / sr_in as f64;
    let n_out = ((x.len() as f64) * ratio).round() as usize;

    let mut y = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t = (i as f64) / ratio; // position in input
        let i0 = t.floor() as isize;
        let i1 = i0 + 1;
        let a = t - (i0 as f64);

        let s0 = if i0 < 0 { 0.0 } else if (i0 as usize) >= x.len() { 0.0 } else { x[i0 as usize] };
        let s1 = if i1 < 0 { 0.0 } else if (i1 as usize) >= x.len() { 0.0 } else { x[i1 as usize] };
        y.push(((1.0 - a) as f32) * s0 + (a as f32) * s1);
    }
    y
}

fn load_audio_16k_mono(path: &Path) -> Result<(Vec<f32>, u32, f64)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::default::{get_codecs, get_probe};

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open audio: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = get_probe().format(
        &Default::default(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format.default_track().ok_or_else(|| anyhow!("No default track"))?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let sr_in = track.codec_params.sample_rate.ok_or_else(|| anyhow!("Unknown sample rate"))?;
    let channels = track.codec_params.channels.ok_or_else(|| anyhow!("Unknown channels"))?.count();

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };

        let decoded = decoder.decode(&packet)?;
        match decoded {
            AudioBufferRef::F32(buf) => {
                let frames = buf.frames();
                for f in 0..frames {
                    let mut acc = 0.0f32;
                    for ch in 0..channels {
                        acc += buf.chan(ch)[f];
                    }
                    samples.push(acc / channels as f32);
                }
            }
            AudioBufferRef::U8(buf) => {
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for ch in 0..channels {
                        acc += (buf.chan(ch)[f] as f32 - 128.0) / 128.0;
                    }
                    samples.push(acc / channels as f32);
                }
            }
            AudioBufferRef::U16(buf) => {
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for ch in 0..channels {
                        acc += (buf.chan(ch)[f] as f32 - 32768.0) / 32768.0;
                    }
                    samples.push(acc / channels as f32);
                }
            }
            AudioBufferRef::S16(buf) => {
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for ch in 0..channels {
                        acc += (buf.chan(ch)[f] as f32) / 32768.0;
                    }
                    samples.push(acc / channels as f32);
                }
            }
            _ => bail!("Unsupported decoded sample format"),
        }
    }

    let mut audio_16k = samples;
    let mut sr = sr_in;
    if sr_in != 16_000 {
        audio_16k = resample_linear(&audio_16k, sr_in, 16_000);
        sr = 16_000;
    }

    let dur = audio_16k.len() as f64 / sr as f64;
    Ok((audio_16k, sr, dur))
}

// -------------------------
// Whisper log-mel (approx OpenAI impl)
// n_fft=400, hop=160, win=400, mel=80, sr=16000
// log10, clamp(max-8), then (x+4)/4
// -------------------------
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (std::f32::consts::PI * 2.0 * i as f32) / (n as f32);
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4_f32.ln();
    let mut mel = 3.0 * hz / 200.0;
    if hz >= min_log_hz {
        mel = min_log_mel + (hz / min_log_hz).ln() * logstep;
    }
    mel
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4_f32.ln() / 27.0;
    let mut hz = 200.0 * mel / 3.0;
    if mel >= min_log_mel {
        hz = min_log_hz * (logstep * (mel - min_log_mel)).exp();
    }
    hz
}

fn build_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Array2<f32> {
    let n_freq = n_fft / 2 + 1;
    let fmax = fmax.min(sr as f32 / 2.0);
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let m = mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32);
        mel_points.push(m);
    }
    let mut freq_points = Vec::with_capacity(n_mels + 2);
    for &m in &mel_points {
        freq_points.push(mel_to_hz_slaney(m));
    }

    let mut fb = Array2::<f32>::zeros((n_mels, n_freq));
    let mut fft_freqs = Vec::with_capacity(n_freq);
    let max_hz = sr as f32 / 2.0;
    for k in 0..n_freq {
        let f = (k as f32) * max_hz / ((n_freq - 1) as f32);
        fft_freqs.push(f);
    }

    for m in 0..n_mels {
        let f_left = freq_points[m];
        let f_center = freq_points[m + 1];
        let f_right = freq_points[m + 2];

        let denom_left = (f_center - f_left).max(1e-6);
        let denom_right = (f_right - f_center).max(1e-6);

        for k in 0..n_freq {
            let f = fft_freqs[k];
            let lower = (f - f_left) / denom_left;
            let upper = (f_right - f) / denom_right;
            let w = lower.min(upper).max(0.0);
            fb[(m, k)] = w;
        }
    }

    // Slaney normalization
    for m in 0..n_mels {
        let f_left = freq_points[m];
        let f_right = freq_points[m + 2];
        let enorm = 2.0 / (f_right - f_left).max(1e-6);
        for k in 0..n_freq {
            fb[(m, k)] *= enorm;
        }
    }
    fb
}

fn whisper_log_mel_80(audio_16k: &[f32]) -> Result<Array2<f32>> {
    let sr = 16_000u32;
    let n_fft = 400usize;
    let hop = 160usize;
    let win = 400usize;
    let n_mels = 80usize;

    if audio_16k.is_empty() {
        bail!("Empty audio");
    }

    // Match torch.stft(center=True, pad_mode="reflect") by reflect-padding n_fft/2 on both sides.
    let pad = n_fft / 2;
    let mut padded: Vec<f32> = Vec::with_capacity(audio_16k.len() + 2 * pad);
    if audio_16k.len() >= 2 && pad > 0 {
        for i in 0..pad {
            let idx = pad - i;
            let src = idx.min(audio_16k.len() - 1);
            padded.push(audio_16k[src]);
        }
        padded.extend_from_slice(audio_16k);
        for i in 0..pad {
            let idx = audio_16k.len().saturating_sub(2 + i);
            padded.push(audio_16k[idx]);
        }
    } else {
        padded.extend_from_slice(audio_16k);
        padded.resize(audio_16k.len() + 2 * pad, 0.0);
    }

    let window = hann_window(win);
    let mel_fb = build_mel_filterbank(sr, n_fft, n_mels, 0.0, 8000.0);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // number of frames
    let mut n_frames = if padded.len() < win {
        1
    } else {
        1 + (padded.len() - win) / hop
    };
    // torch.stft returns one extra frame; HF drops the last frame via stft[..., :-1]
    if n_frames > 1 {
        n_frames -= 1;
    }

    let mut spec = Array2::<f32>::zeros((n_mels, n_frames));

    let mut fft_in: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
    let mut fft_out = fft_in.clone();

    for frame in 0..n_frames {
        let start = frame * hop;

        // windowed frame into fft_in (zero-pad to n_fft)
        for i in 0..n_fft {
            fft_in[i] = Complex::new(0.0, 0.0);
        }
        for i in 0..win {
            let idx = start + i;
            let s = if idx < padded.len() { padded[idx] } else { 0.0 };
            fft_in[i] = Complex::new(s * window[i], 0.0);
        }

        fft_out.copy_from_slice(&fft_in);
        fft.process(&mut fft_out);

        // power spectrum (n_fft/2+1)
        let n_freq = n_fft / 2 + 1;
        let mut pows = vec![0.0f32; n_freq];
        for k in 0..n_freq {
            let c = fft_out[k];
            pows[k] = c.re * c.re + c.im * c.im;
        }

        // mel energies = mel_fb dot pows
        for m in 0..n_mels {
            let mut e = 0.0f32;
            for k in 0..n_freq {
                e += mel_fb[(m, k)] * pows[k];
            }
            spec[(m, frame)] = e.max(1e-10);
        }
    }

    // log10 + clamp(max-8) + (x+4)/4 (Whisper normalization)
    let mut max_log = f32::NEG_INFINITY;
    for v in spec.iter() {
        let lv = v.log10();
        if lv > max_log {
            max_log = lv;
        }
    }

    for v in spec.iter_mut() {
        let lv = v.log10();
        let clamped = lv.max(max_log - 8.0);
        *v = (clamped + 4.0) / 4.0;
    }

    Ok(spec)
}

// -------------------------
// Tokenizer/prompt (minimal)
// -------------------------
// For a true 1:1 port, you’d load HF tokenizer.json and map special tokens.
// Here we provide a minimal approach: hardcode Whisper special token IDs for OpenAI Whisper *base*.
// NOTE: These IDs match the standard Whisper tokenizer used by OpenAI/HF for multilingual models.
// If you use a different tokenizer or custom vocab, adjust accordingly.
#[derive(Clone, Debug)]
struct WhisperSpecial {
    sot: i64,
    eot: i64,
    // Language token like <|en|> depends on mapping; for simplicity we provide a small map.
    lang: i64,
    task: i64,
    no_timestamps: i64,
}

fn special_tokens(language: &str, task: &str, tokenizer: Option<&Tokenizer>) -> Result<WhisperSpecial> {
    if let Some(tok) = tokenizer {
        let get_id = |t: &str| -> Result<i64> {
            tok.token_to_id(t)
                .map(|v| v as i64)
                .ok_or_else(|| anyhow!("Tokenizer missing token: {t}"))
        };
        let sot = get_id("<|startoftranscript|>")?;
        let eot = get_id("<|endoftext|>")?;
        let lang = get_id(&format!("<|{}|>", language))?;
        let task_tok = get_id(&format!("<|{}|>", task))?;
        let no_ts = get_id("<|notimestamps|>")?;
        return Ok(WhisperSpecial { sot, eot, lang, task: task_tok, no_timestamps: no_ts });
    }

    // These are standard for Whisper multilingual tokenizer:
    // <|startoftranscript|> = 50258, <|endoftext|> = 50257
    // Language tokens are in a contiguous block; <|en|> = 50259 for multilingual.
    // Task: <|transcribe|>, <|translate|> are also in that block.
    //
    // If your export/tokenizer differs, replace this with tokenizer.json loading.
    let sot = 50258;
    let eot = 50257;

    // Common mapping for multilingual:
    // <|en|> 50259, <|transcribe|> 50359, <|translate|> 50358 (HF uses these)
    let lang = match language {
        "en" => 50259,
        "hi" => 50276, // example; verify for your tokenizer if needed
        _ => 50259,    // default to en
    };

    let task_tok = match task {
        "transcribe" => 50359,
        "translate" => 50358,
        _ => 50359,
    };

    let no_ts = 50363;

    Ok(WhisperSpecial { sot, eot, lang, task: task_tok, no_timestamps: no_ts })
}

// -------------------------
// ONNX: encoder + decoder_with_past greedy
// -------------------------
fn resolve_tokenizer(args: &Args) -> Result<Option<(Tokenizer, PathBuf)>> {
    if !args.tokenizer_json.trim().is_empty() {
        let p = PathBuf::from(args.tokenizer_json.trim());
        if !p.is_file() {
            bail!("tokenizer_json not found: {}", p.display());
        }
        let tok = Tokenizer::from_file(&p)
            .map_err(|e| anyhow!("Failed to load tokenizer {}: {e}", p.display()))?;
        return Ok(Some((tok, p)));
    }

    let candidates = [
        PathBuf::from(&args.onnx_dir).join("tokenizer.json"),
        PathBuf::from(&args.model_id).join("tokenizer.json"),
        PathBuf::from("tokenizer.json"),
    ];
    for p in candidates.iter() {
        if p.is_file() {
            let tok = Tokenizer::from_file(p)
                .map_err(|e| anyhow!("Failed to load tokenizer {}: {e}", p.display()))?;
            return Ok(Some((tok, p.clone())));
        }
    }

    // Try Hugging Face cache (if the model was previously downloaded).
    if args.model_id.contains('/') {
        let mut parts = args.model_id.splitn(2, '/');
        let org = parts.next().unwrap_or("");
        let name = parts.next().unwrap_or("");
        if !org.is_empty() && !name.is_empty() {
            let hf_home = std::env::var("HF_HOME").ok();
            let base = hf_home
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                    PathBuf::from(home).join(".cache/huggingface")
                });
            let hub = base.join("hub");
            let model_dir = hub.join(format!("models--{}--{}", org, name));
            let snaps = model_dir.join("snapshots");
            if snaps.is_dir() {
                let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
                for entry in fs::read_dir(&snaps)? {
                    let entry = entry?;
                    let path = entry.path().join("tokenizer.json");
                    if path.is_file() {
                        let m = entry.metadata()?.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        match &best {
                            Some((best_m, _)) if *best_m >= m => {}
                            _ => best = Some((m, path)),
                        }
                    }
                }
                if let Some((_, p)) = best {
                    let tok = Tokenizer::from_file(&p)
                        .map_err(|e| anyhow!("Failed to load tokenizer {}: {e}", p.display()))?;
                    return Ok(Some((tok, p)));
                }
            }
        }
    }
    Ok(None)
}

fn decode_tokens(tokens: &[i64], tokenizer: Option<&Tokenizer>) -> Result<String> {
    if let Some(tok) = tokenizer {
        let ids: Vec<u32> = tokens.iter().filter_map(|&t| u32::try_from(t).ok()).collect();
        let text = tok.decode(&ids, true)
            .map_err(|e| anyhow!("Tokenizer decode failed: {e}"))?;
        return Ok(text);
    }
    Ok(format!(
        "[TOKENS:{}]",
        tokens.iter().take(200).map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
    ))
}

fn load_generation_cfg(path: &Path) -> Result<GenerationCfg> {
    if !path.is_file() {
        return Ok(GenerationCfg::default());
    }
    let txt = fs::read_to_string(path)?;
    let cfg: GenerationCfg = serde_json::from_str(&txt)?;
    Ok(cfg)
}

fn stitch_texts(chunks: &[String]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        let t = chunk.trim();
        if t.is_empty() {
            continue;
        }
        if out.is_empty() {
            out.push_str(t);
            continue;
        }
        let overlap = word_overlap(&out, t, 16);
        if overlap > 0 {
            let words: Vec<&str> = t.split_whitespace().collect();
            let remaining = words[overlap..].join(" ");
            if !remaining.is_empty() {
                out.push(' ');
                out.push_str(&remaining);
            }
        } else {
            out.push(' ');
            out.push_str(t);
        }
    }
    out
}

fn word_overlap(a: &str, b: &str, max_words: usize) -> usize {
    let a_words: Vec<String> = a.split_whitespace().map(|w| w.to_lowercase()).collect();
    let b_words: Vec<String> = b.split_whitespace().map(|w| w.to_lowercase()).collect();
    let max = max_words.min(a_words.len()).min(b_words.len());
    for k in (1..=max).rev() {
        if a_words[a_words.len() - k..] == b_words[..k] {
            return k;
        }
    }
    0
}

fn run_encoder(encoder: &Session, input_features: Array3<f32>) -> Result<ArrayD<f32>> {
    // input_features: [1, 80, 3000]
    let input_val = Value::from_array(input_features)?;
    let input_name = encoder.inputs.get(0).map(|i| i.name.as_str())
        .ok_or_else(|| anyhow!("Encoder has no inputs"))?;
    let outputs = encoder.run(vec![(input_name, input_val)])?;
    // Assume first output is encoder_hidden_states
    let out0: ArrayD<f32> = outputs[0].try_extract_tensor::<f32>()?.to_owned();
    Ok(out0)
}

fn argmax_last_dim_raw(shape: &[i64], data: &[f32], suppress: Option<&HashSet<i64>>) -> Result<i64> {
    if shape.len() < 2 {
        bail!("Unexpected logits shape: {:?}", shape);
    }
    let vocab_dim = *shape.last().unwrap() as usize;
    if vocab_dim == 0 || data.len() < vocab_dim {
        bail!("Invalid logits shape/data: vocab_dim={vocab_dim}, len={}", data.len());
    }
    let rows = data.len() / vocab_dim;
    let row_start = (rows - 1) * vocab_dim;
    let row = &data[row_start..row_start + vocab_dim];

    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if let Some(set) = suppress {
            if set.contains(&(i as i64)) {
                continue;
            }
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(best_i as i64)
}

fn topk_logprobs_last_dim_raw(
    shape: &[i64],
    data: &[f32],
    k: usize,
    suppress: Option<&HashSet<i64>>,
) -> Result<Vec<(i64, f64)>> {
    if shape.len() < 2 {
        bail!("Unexpected logits shape: {:?}", shape);
    }
    let vocab_dim = *shape.last().unwrap() as usize;
    if vocab_dim == 0 || data.len() < vocab_dim {
        bail!("Invalid logits shape/data: vocab_dim={vocab_dim}, len={}", data.len());
    }

    let rows = data.len() / vocab_dim;
    let row_start = (rows - 1) * vocab_dim;
    let row = &data[row_start..row_start + vocab_dim];

    let mut candidates: Vec<(usize, f32)> = row
        .iter()
        .enumerate()
        .filter(|(i, _)| suppress.map(|s| !s.contains(&(*i as i64))).unwrap_or(true))
        .map(|(i, &v)| (i, v))
        .collect();
    if candidates.is_empty() {
        bail!("All logits were suppressed");
    }

    let max_logit = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = candidates
        .iter()
        .map(|(_, v)| ((*v - max_logit).exp()) as f64)
        .sum();
    let log_denom = (max_logit as f64) + sum_exp.ln();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let take_n = k.max(1).min(candidates.len());
    Ok(candidates
        .into_iter()
        .take(take_n)
        .map(|(i, v)| (i as i64, (v as f64) - log_denom))
        .collect())
}

fn insert_present_as_past(
    outputs: &mut ort::SessionOutputs<'_, '_>,
    target: &mut HashMap<String, Value>,
    output_specs: &[ort::Output],
) {
    for out in output_specs.iter() {
        let name = out.name.as_str();
        if name.starts_with("present.") {
            if let Some(v) = outputs.remove(name) {
                let past_name = name.replacen("present.", "past_key_values.", 1);
                target.insert(past_name, v);
            }
        }
    }
}

fn greedy_decode_with_past(
    decoder: &Session,
    decoder_with_past: &Session,
    encoder_hidden_states: &ArrayD<f32>,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    eot: i64,
    gen_cfg: &GenerationCfg,
) -> Result<Vec<i64>> {
    let mut tokens: Vec<i64> = prompt_ids.to_vec();
    let enc_val = Value::from_array(encoder_hidden_states.to_owned())?;

    let base_suppress: HashSet<i64> = gen_cfg.suppress_tokens.clone().unwrap_or_default().into_iter().collect();
    let begin_suppress: HashSet<i64> = gen_cfg.begin_suppress_tokens.clone().unwrap_or_default().into_iter().collect();
    let mut suppress_first = base_suppress.clone();
    suppress_first.extend(begin_suppress.iter());

    // First step uses decoder (with encoder_hidden_states) to get initial past, including encoder K/V.
    let input_ids = Array2::<i64>::from_shape_vec((1, tokens.len()), tokens.clone())?;
    let input_ids_val = Value::from_array(input_ids)?;
    let mut outputs = decoder.run(vec![
        ("input_ids", SessionInputValue::from(input_ids_val)),
        ("encoder_hidden_states", SessionInputValue::from(enc_val.view())),
    ])?;
    let (shape, data) = outputs[0].try_extract_raw_tensor::<f32>()?;
    let next_id = argmax_last_dim_raw(&shape, data, Some(&suppress_first))?;
    tokens.push(next_id);

    if next_id == eot {
        return Ok(tokens);
    }

    // Initialize past from first decoder outputs (both decoder + encoder).
    let mut past: HashMap<String, Value> = HashMap::new();
    insert_present_as_past(&mut outputs, &mut past, &decoder.outputs);

    let mut step_input_ids = Tensor::<i64>::from_array(Array2::<i64>::zeros((1, 1)))?;
    let mem_info = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?;
    let mut binding = decoder_with_past.create_binding()?;

    for _ in 1..max_new_tokens {
        let step_token = *tokens.last().unwrap_or(&prompt_ids[0]);
        let (_shape, data_mut) = step_input_ids.try_extract_raw_tensor_mut::<i64>()?;
        data_mut[0] = step_token;

        binding.clear_inputs();
        binding.clear_outputs();
        for out in decoder_with_past.outputs.iter() {
            binding.bind_output_to_device(&out.name, &mem_info)?;
        }
        binding.bind_input("input_ids", &step_input_ids)?;
        for inp in decoder_with_past.inputs.iter() {
            if inp.name.starts_with("past_key_values.") {
                if let Some(v) = past.get(&inp.name) {
                    binding.bind_input(&inp.name, v)?;
                } else {
                    bail!("Missing cached decoder input: {}", inp.name);
                }
            }
        }

        let mut outputs = binding.run()?;
        let logits = outputs.remove("logits").ok_or_else(|| anyhow!("Missing logits output"))?;
        let (shape, data) = logits.try_extract_raw_tensor::<f32>()?;
        let next_id = argmax_last_dim_raw(&shape, data, Some(&base_suppress))?;
        tokens.push(next_id);

        if next_id == eot {
            break;
        }

        // Update decoder past; encoder past stays constant.
        insert_present_as_past(&mut outputs, &mut past, &decoder_with_past.outputs);
    }

    Ok(tokens)
}

fn beam_search_decode(
    decoder: &Session,
    encoder_hidden_states: &ArrayD<f32>,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    num_beams: usize,
    eot: i64,
    gen_cfg: &GenerationCfg,
) -> Result<Vec<i64>> {
    #[derive(Clone)]
    struct BeamHyp {
        tokens: Vec<i64>,
        score: f64,
        finished: bool,
    }

    let k = num_beams.max(1);
    if k == 1 {
        bail!("beam_search_decode called with num_beams=1");
    }

    let enc_val = Value::from_array(encoder_hidden_states.to_owned())?;
    let base_suppress: HashSet<i64> = gen_cfg
        .suppress_tokens
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let begin_suppress: HashSet<i64> = gen_cfg
        .begin_suppress_tokens
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let mut suppress_first = base_suppress.clone();
    suppress_first.extend(begin_suppress.iter());

    let mut beams = vec![BeamHyp {
        tokens: prompt_ids.to_vec(),
        score: 0.0,
        finished: false,
    }];

    for step in 0..max_new_tokens {
        let suppress = if step == 0 {
            Some(&suppress_first)
        } else {
            Some(&base_suppress)
        };
        let mut next_beams: Vec<BeamHyp> = Vec::new();

        for beam in &beams {
            if beam.finished {
                next_beams.push(beam.clone());
                continue;
            }

            let input_ids = Array2::<i64>::from_shape_vec((1, beam.tokens.len()), beam.tokens.clone())?;
            let input_ids_val = Value::from_array(input_ids)?;
            let outputs = decoder.run(vec![
                ("input_ids", SessionInputValue::from(input_ids_val)),
                ("encoder_hidden_states", SessionInputValue::from(enc_val.view())),
            ])?;

            let (shape, data) = outputs[0].try_extract_raw_tensor::<f32>()?;
            let topk = topk_logprobs_last_dim_raw(&shape, data, k, suppress)?;
            for (tok, logp) in topk {
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(tok);
                next_beams.push(BeamHyp {
                    tokens: new_tokens,
                    score: beam.score + logp,
                    finished: tok == eot,
                });
            }
        }

        next_beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next_beams.truncate(k);
        if next_beams.is_empty() {
            bail!("Beam search produced no candidates");
        }
        let all_finished = next_beams.iter().all(|b| b.finished);
        beams = next_beams;
        if all_finished {
            break;
        }
    }

    let best = beams
        .into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("No beam hypothesis produced"))?;
    Ok(best.tokens)
}

fn decode_with_strategy(
    decoder: &Session,
    decoder_with_past: &Session,
    encoder_hidden_states: &ArrayD<f32>,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    num_beams: usize,
    eot: i64,
    gen_cfg: &GenerationCfg,
) -> Result<Vec<i64>> {
    if num_beams <= 1 {
        greedy_decode_with_past(
            decoder,
            decoder_with_past,
            encoder_hidden_states,
            prompt_ids,
            max_new_tokens,
            eot,
            gen_cfg,
        )
    } else {
        beam_search_decode(
            decoder,
            encoder_hidden_states,
            prompt_ids,
            max_new_tokens,
            num_beams,
            eot,
            gen_cfg,
        )
    }
}

// -------------------------
// Chunked “long-form” transcription (Rust approximation)
// -------------------------
fn transcribe_longform_chunked(
    encoder: &Session,
    decoder: &Session,
    decoder_with_past: &Session,
    audio_16k: &[f32],
    language: &str,
    task: &str,
    max_new_tokens: usize,
    num_beams: usize,
    chunk_length_s: f32,
    overlap_s: f32,
    tokenizer: Option<&Tokenizer>,
    timestamps: bool,
    gen_cfg: &GenerationCfg,
    chunk_parallelism: usize,
) -> Result<(String, Timing)> {
    let t0 = Instant::now();

    let special = special_tokens(language, task, tokenizer)?;
    let mut prompt = vec![special.sot, special.lang, special.task];
    if !timestamps {
        prompt.push(special.no_timestamps);
    }

    // Chunking
    let sr = 16_000usize;
    let chunk_len = (chunk_length_s * sr as f32).round() as usize;
    let overlap = (overlap_s * sr as f32).round() as usize;
    let step = chunk_len.saturating_sub(overlap).max(1);
    let hop = 160usize;

    let mut texts: Vec<String> = Vec::new();

    let mut preprocess_total = 0.0;
    let mut model_total = 0.0;
    let mut decode_total = 0.0;

    let tp0 = Instant::now();
    let mel_full = whisper_log_mel_80(audio_16k)?;
    preprocess_total += tp0.elapsed().as_secs_f64();
    let total_frames = mel_full.len_of(Axis(1));

    let mut chunks: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    while pos < audio_16k.len() {
        let end = (pos + chunk_len).min(audio_16k.len());
        chunks.push(pos);
        if end == audio_16k.len() { break; }
        pos += step;
    }

    if chunk_parallelism > 0 && chunks.len() > 1 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(chunk_parallelism)
            .build()
            .map_err(|e| anyhow!("Failed to build thread pool: {e}"))?;
        let tm0 = Instant::now();
        let chunk_tokens: Vec<(usize, Vec<i64>)> = pool.install(|| {
            chunks
                .par_iter()
                .enumerate()
                .map(|(idx, &chunk_pos)| {
                    let frame_start = chunk_pos / hop;
                    let frame_end = frame_start + 3000;
                    let available_end = frame_end.min(total_frames);

                    let mut mel = Array2::<f32>::zeros((80, 3000));
                    if frame_start < total_frames {
                        let frames = available_end - frame_start;
                        mel.slice_mut(ndarray::s![.., 0..frames])
                            .assign(&mel_full.slice(ndarray::s![.., frame_start..available_end]));
                    }
                    let input_features = mel.insert_axis(Axis(0));
                    let enc = run_encoder(encoder, input_features)?;
                    let tokens = decode_with_strategy(
                        decoder,
                        decoder_with_past,
                        &enc,
                        &prompt,
                        max_new_tokens,
                        num_beams,
                        special.eot,
                        gen_cfg,
                    )?;
                    Ok::<_, anyhow::Error>((idx, tokens))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        model_total += tm0.elapsed().as_secs_f64();

        let td0 = Instant::now();
        let mut chunk_tokens = chunk_tokens;
        chunk_tokens.sort_by_key(|(idx, _)| *idx);
        for (_idx, tokens) in chunk_tokens {
            let mut gen_tokens = if tokens.len() > prompt.len() {
                tokens[prompt.len()..].to_vec()
            } else {
                Vec::new()
            };
            if let Some(last) = gen_tokens.last() {
                if *last == special.eot {
                    gen_tokens.pop();
                }
            }
            let mut text = decode_tokens(&gen_tokens, tokenizer)?;
            if text.is_empty() {
                text = "[EMPTY]".to_string();
            }
            if text != "[EMPTY]" {
                texts.push(text);
            }
        }
        decode_total += td0.elapsed().as_secs_f64();
    } else {
        let mut pos = 0usize;
        while pos < audio_16k.len() {
            let end = (pos + chunk_len).min(audio_16k.len());
            // Preprocess: slice log-mel by frame and pad to 3000 frames
            let frame_start = pos / hop;
            let frame_end = frame_start + 3000;
            let available_end = frame_end.min(total_frames);

            let mut mel = Array2::<f32>::zeros((80, 3000));
            if frame_start < total_frames {
                let frames = available_end - frame_start;
                mel.slice_mut(ndarray::s![.., 0..frames])
                    .assign(&mel_full.slice(ndarray::s![.., frame_start..available_end]));
            }
            // [1, 80, 3000]
            let input_features = mel.insert_axis(Axis(0));

            // Model: encoder + decoder
            let tm0 = Instant::now();
            let enc = run_encoder(encoder, input_features)?;
            let tokens = decode_with_strategy(
                decoder,
                decoder_with_past,
                &enc,
                &prompt,
                max_new_tokens,
                num_beams,
                special.eot,
                gen_cfg,
            )?;
            model_total += tm0.elapsed().as_secs_f64();

            // Decode: use tokenizer if available, otherwise output token IDs.
            let td0 = Instant::now();
            let mut gen_tokens = if tokens.len() > prompt.len() {
                tokens[prompt.len()..].to_vec()
            } else {
                Vec::new()
            };
            if let Some(last) = gen_tokens.last() {
                if *last == special.eot {
                    gen_tokens.pop();
                }
            }
            let mut text = decode_tokens(&gen_tokens, tokenizer)?;
            if text.is_empty() {
                text = "[EMPTY]".to_string();
            }
            decode_total += td0.elapsed().as_secs_f64();

            if text != "[EMPTY]" {
                texts.push(text);
            }

            if end == audio_16k.len() { break; }
            pos += step;
        }
    }

    let full_text = stitch_texts(&texts);
    let end_to_end = t0.elapsed().as_secs_f64();

    Ok((
        full_text,
        Timing {
            preprocess_s: preprocess_total,
            model_only_s: model_total,
            decode_s: decode_total,
            end_to_end_s: end_to_end,
        },
    ))
}

#[derive(Debug, Clone, Serialize)]
struct Timing {
    preprocess_s: f64,
    model_only_s: f64,
    decode_s: f64,
    end_to_end_s: f64,
}

// -------------------------
// Stats helpers
// -------------------------
fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by_key(|v| OrderedFloat(*v));
    let k = (xs.len() as f64 - 1.0) * (p / 100.0);
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c { return xs[f]; }
    xs[f] + (xs[c] - xs[f]) * (k - f as f64)
}

fn stat_block(xs: &[f64]) -> serde_json::Value {
    let mut v = xs.to_vec();
    v.sort_by_key(|x| OrderedFloat(*x));
    let minv = *v.first().unwrap_or(&f64::NAN);
    let maxv = *v.last().unwrap_or(&f64::NAN);
    let mean = if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    let median = if v.is_empty() { f64::NAN } else { v[v.len()/2] };
    serde_json::json!({
        "min": minv,
        "median": median,
        "p90": percentile(xs.to_vec(), 90.0),
        "p95": percentile(xs.to_vec(), 95.0),
        "max": maxv,
        "mean": mean
    })
}

// -------------------------
// Output rows
// -------------------------
#[derive(Debug, Clone, Serialize)]
struct RowOut {
    file: String,
    duration_s: f64,
    end_to_end_s: f64,
    rtf: f64,
    text: String,
}

fn ensure_parent_dir(path: &str) -> Result<()> {
    if let Some(p) = Path::new(path).parent() {
        fs::create_dir_all(p)?;
    }
    Ok(())
}

fn prepare_output_dirs(args: &Args) -> Result<()> {
    ensure_parent_dir(&args.out_csv)?;
    ensure_parent_dir(&args.out_json)?;
    ensure_parent_dir(&args.out_summary_json)?;
    Ok(())
}

fn is_supported_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            SUPPORTED_AUDIO_EXTS.iter().any(|v| *v == lower)
        })
        .unwrap_or(false)
}

fn collect_audio_files_recursive(dir: &Path, root: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_audio_files_recursive(&path, root, out)?;
            continue;
        }
        if !path.is_file() || !is_supported_audio_file(&path) {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|e| anyhow!("Failed to compute relative path for {}: {e}", path.display()))?;
        out.push(rel.to_string_lossy().replace('\\', "/"));
    }
    Ok(())
}

fn list_audio_files(audio_dir: &str, limit_files: usize) -> Result<Vec<String>> {
    let root = Path::new(audio_dir);
    if !root.is_dir() {
        bail!("Audio directory does not exist: {}", audio_dir);
    }

    let mut files: Vec<String> = Vec::new();
    collect_audio_files_recursive(root, root, &mut files)?;
    files.sort_by_cached_key(|s| s.to_ascii_lowercase());
    if limit_files > 0 {
        files.truncate(limit_files);
    }
    if files.is_empty() {
        bail!("No audio files found in {}", audio_dir);
    }
    Ok(files)
}

fn tokenizer_ref(tokenizer: &Option<(Tokenizer, PathBuf)>) -> Option<&Tokenizer> {
    tokenizer.as_ref().map(|t| &t.0)
}

fn round_to(value: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    (value * factor).round() / factor
}

// -------------------------
// Main
// -------------------------
fn main() -> Result<()> {
    let args = Args::parse();

    prepare_output_dirs(&args)?;

    let cfg = if !args.discovery_best_json.is_empty() {
        load_best_cfg_from_discovery(&args.discovery_best_json)?
    } else {
        suggested_optimum_cfg()
    };
    let mut cfg = cfg;
    if args.intra_op > 0 {
        cfg.intra_op = args.intra_op;
    }
    if args.inter_op > 0 {
        cfg.inter_op = args.inter_op;
    }

    let tokenizer = resolve_tokenizer(&args)?;
    let gen_cfg = load_generation_cfg(&PathBuf::from(&args.onnx_dir).join("generation_config.json"))?;

    // ORT env (global)
    let _env = ort::init()
        .with_name("whisper_ort_bench")
        .commit()?;

    let onnx_dir = PathBuf::from(&args.onnx_dir);
    if !onnx_dir.is_dir() {
        bail!("onnx_dir does not exist or is not a directory: {}", onnx_dir.display());
    }

    let encoder_path = onnx_dir.join("encoder_model.onnx");
    let decoder_path = onnx_dir.join("decoder_model.onnx");
    let decoder_with_past_path = onnx_dir.join("decoder_with_past_model.onnx");

    let encoder = build_session(&encoder_path, &cfg)
        .with_context(|| format!("Failed to load {}", encoder_path.display()))?;
    let decoder = build_session(&decoder_path, &cfg)
        .with_context(|| format!("Failed to load {}", decoder_path.display()))?;
    let decoder_with_past = build_session(&decoder_with_past_path, &cfg)
        .with_context(|| format!("Failed to load {}", decoder_with_past_path.display()))?;

    let files = list_audio_files(&args.audio_dir, args.limit_files)?;
    let tokenizer_for_decode = tokenizer_ref(&tokenizer);

    // Warmup
    if args.warmup > 0 {
        let p0 = Path::new(&args.audio_dir).join(&files[0]);
        let (a0, sr0, _dur) = load_audio_16k_mono(&p0)?;
        if sr0 != 16_000 { bail!("Unexpected sample rate after resample"); }
        for _ in 0..args.warmup {
            let _ = transcribe_longform_chunked(
                &encoder,
                &decoder,
                &decoder_with_past,
                &a0,
                &args.language,
                &args.task,
                args.max_new_tokens,
                args.num_beams,
                args.chunk_length_s,
                args.overlap_s,
                tokenizer_for_decode,
                args.timestamps,
                &gen_cfg,
                args.chunk_parallelism,
            )?;
        }
    }

    let mut rows: Vec<RowOut> = Vec::new();
    let mut end2end_list = Vec::<f64>::new();
    let mut load_list = Vec::<f64>::new();
    let mut preprocess_list = Vec::<f64>::new();
    let mut model_only_list = Vec::<f64>::new();
    let mut decode_list = Vec::<f64>::new();
    let mut rtf_end2end_list = Vec::<f64>::new();

    let txt_dir = Path::new(&args.out_csv).parent().unwrap().to_path_buf();

    for fnm in &files {
        let path = Path::new(&args.audio_dir).join(fnm);

        // load time
        let tl0 = Instant::now();
        let (audio, sr, dur) = load_audio_16k_mono(&path)?;
        let load_s = tl0.elapsed().as_secs_f64();
        if sr != 16_000 { bail!("Unexpected sample rate after resample"); }

        // model/pre/post
        let (text, t) = transcribe_longform_chunked(
            &encoder,
            &decoder,
            &decoder_with_past,
            &audio,
            &args.language,
            &args.task,
            args.max_new_tokens,
            args.num_beams,
            args.chunk_length_s,
            args.overlap_s,
            tokenizer_for_decode,
            args.timestamps,
            &gen_cfg,
            args.chunk_parallelism,
        )?;

        let end_to_end_s = load_s + t.end_to_end_s;
        let rtf = end_to_end_s / dur.max(1e-9);

        rows.push(RowOut {
            file: fnm.clone(),
            duration_s: round_to(dur, 3),
            end_to_end_s: round_to(end_to_end_s, 4),
            rtf: round_to(rtf, 6),
            text: text.clone(),
        });

        load_list.push(load_s);
        preprocess_list.push(t.preprocess_s);
        model_only_list.push(t.model_only_s);
        decode_list.push(t.decode_s);
        end2end_list.push(end_to_end_s);
        rtf_end2end_list.push(rtf);

        if args.write_txt {
            let base = Path::new(fnm).file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let out_txt = txt_dir.join(format!("{base}.transcript.txt"));
            fs::write(out_txt, format!("{}\n", text.trim()))?;
        }
    }

    // CSV
    {
        let mut w = Writer::from_path(&args.out_csv)?;
        w.write_record(["file", "duration_s", "end_to_end_s", "rtf", "text"])?;
        for r in &rows {
            w.write_record([
                r.file.as_str(),
                &format!("{:.3}", r.duration_s),
                &format!("{:.4}", r.end_to_end_s),
                &format!("{:.6}", r.rtf),
                r.text.as_str(),
            ])?;
        }
        w.flush()?;
    }

    // JSON per-file
    fs::write(&args.out_json, serde_json::to_string_pretty(&rows)?)?;

    // Summary
    let summary = serde_json::json!({
        "config_used": {
            "intra_op": cfg.intra_op,
            "inter_op": cfg.inter_op,
            "execution_mode": cfg.execution_mode,
            "graph_opt": cfg.graph_opt,
            "cpu_mem_arena": cfg.cpu_mem_arena,
            "mem_pattern": cfg.mem_pattern,
            "allow_spinning": cfg.allow_spinning,
            "num_beams": args.num_beams
        },
        "n_files": rows.len(),
        "latency_end_to_end_s": stat_block(&end2end_list),
        "breakdown_s": {
            "load_s": stat_block(&load_list),
            "preprocess_s": stat_block(&preprocess_list),
            "model_only_s": stat_block(&model_only_list),
            "decode_s": stat_block(&decode_list),
        },
        "rtf_end_to_end": stat_block(&rtf_end2end_list),
        "model_id": args.model_id,
        "onnx_dir": args.onnx_dir,
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "tokenizer_json": tokenizer.as_ref().map(|t| t.1.display().to_string()).unwrap_or_else(|| "".to_string()),
        "timestamps": args.timestamps,
        "notes": {
            "longform": if args.num_beams > 1 {
                "Rust approximation: chunked 30s windows with overlap; beam search decode via decoder"
            } else {
                "Rust approximation: chunked 30s windows with overlap; greedy decode via decoder_with_past"
            },
            "token_decode": if tokenizer.is_some() { "Tokenizer decode (skip_special_tokens=true)" } else { "Prints token IDs unless you provide tokenizer.json." }
        }
    });

    fs::write(&args.out_summary_json, serde_json::to_string_pretty(&summary)?)?;

    println!("DONE");
    println!("Config used:\n{}", serde_json::to_string_pretty(&cfg)?);
    println!("Per-file CSV: {}", args.out_csv);
    println!("Per-file JSON: {}", args.out_json);
    println!("Summary JSON: {}", args.out_summary_json);
    if let Some(p95) = summary["latency_end_to_end_s"]["p95"].as_f64() {
        println!("End-to-end p95(s): {:.6}", p95);
    }

    Ok(())
}
