use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::{Deserialize, Serialize};
use std::{fs::{self, File}, io::{Read, Seek}, path::{Path, PathBuf}, sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex}, time::Instant};
use sysinfo::System;
use tauri::{AppHandle, Emitter, State};
use tokenizers::Tokenizer;

const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Default)]
pub struct EngineState {
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub size: String,
    pub bytes: u64,
    pub architecture: String,
    pub quantization: String,
    pub context: usize,
    pub status: String,
    pub parameter_count: Option<String>,
    pub tokenizer: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct SystemStats {
    pub cpu_usage: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub process_memory_mb: u64,
    pub gpu_available: bool,
    pub gpu_note: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens: usize,
    pub seconds: f64,
    pub tokens_per_sec: f64,
    pub cancelled: bool,
}

#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    pub request_id: String,
    pub model_path: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub context_length: usize,
    pub seed: u64,
}

#[derive(Debug, Serialize, Clone)]
struct TokenEvent {
    request_id: String,
    text: String,
}

#[tauri::command]
pub fn engine_status() -> String {
    "Candle Rust engine ready".into()
}

#[tauri::command]
pub fn system_stats() -> SystemStats {
    let mut system = System::new_all();
    system.refresh_all();
    let cpu = system.global_cpu_usage();
    let memory_total_mb = system.total_memory() / 1024 / 1024;
    let memory_used_mb = system.used_memory() / 1024 / 1024;
    let process_memory_mb = std::process::id();
    let process_memory_mb = system
        .process(sysinfo::Pid::from_u32(process_memory_mb))
        .map(|p| p.memory() / 1024 / 1024)
        .unwrap_or(0);
    SystemStats { cpu_usage: cpu, memory_used_mb, memory_total_mb, process_memory_mb, gpu_available: false, gpu_note: "CPU Candle backend is bundled by default. CUDA is available as an optional build feature.".into() }
}

#[tauri::command]
pub fn inspect_model(path: String) -> Result<ModelInfo, String> {
    inspect_model_inner(Path::new(&path)).map_err(|e| e.to_string())
}

fn inspect_model_inner(path: &Path) -> Result<ModelInfo> {
    if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("gguf")) != Some(true) {
        bail!("Synth Studio accepts GGUF model files.");
    }
    let meta = fs::metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    if !meta.is_file() { bail!("Selected model is not a file."); }
    if meta.len() > MAX_MODEL_BYTES { bail!("Model exceeds Synth Studio's 2 GB lightweight model limit."); }
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let architecture = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| "unknown".into());
    let quantization = content.metadata.get("general.file_type").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| "GGUF".into());
    let key = format!("{}.context_length", architecture);
    let context = content.metadata.get(&key).and_then(|v| v.to_u64().ok()).map(|v| v as usize).unwrap_or(4096);
    let parameter_count = content.metadata.get("general.parameter_count").and_then(|v| v.to_u64().ok()).map(|v| format_count(v));
    let tokenizer = if has_embedded_tokenizer(&content) { "embedded" } else if sidecar_tokenizer(path).is_some() { "sidecar" } else { "missing" };
    let status = if architecture.eq_ignore_ascii_case("llama") && tokenizer != "missing" { "ready" } else if architecture.eq_ignore_ascii_case("llama") { "ready" } else { "unsupported" };
    let name = content.metadata.get("general.name").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or("GGUF model").into());
    Ok(ModelInfo { id: format!("{}-{}", meta.len(), path.display()), name, path: path.display().to_string(), size: format_size(meta.len()), bytes: meta.len(), architecture, quantization, context, status: status.into(), parameter_count, tokenizer: tokenizer.into() })
}

fn has_embedded_tokenizer(content: &gguf_file::Content) -> bool {
    ["tokenizer.huggingface.json", "tokenizer.ggml.hf_json", "tokenizer.huggingface.tokenizer_json"].iter().any(|key| content.metadata.get(*key).and_then(|v| v.to_string().ok()).is_some())
}

fn load_tokenizer(content: &gguf_file::Content, model_path: &Path) -> Result<Tokenizer> {
    for key in ["tokenizer.huggingface.json", "tokenizer.ggml.hf_json", "tokenizer.huggingface.tokenizer_json"] {
        if let Some(json) = content.metadata.get(key).and_then(|v| v.to_string().ok()) {
            return Tokenizer::from_bytes(json.as_bytes()).map_err(|e| anyhow::anyhow!(e.to_string()));
        }
    }
    if let Some(sidecar) = sidecar_tokenizer(model_path) {
        return Tokenizer::from_file(sidecar).map_err(|e| anyhow::anyhow!(e.to_string()));
    }
    bail!("This GGUF does not contain an embedded tokenizer. Put tokenizer.json beside the GGUF file (or a <model>.tokenizer.json sidecar) and retry.")
}

fn sidecar_tokenizer(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = model_path.file_stem()?.to_string_lossy();
    let candidates = [parent.join("tokenizer.json"), parent.join(format!("{}.tokenizer.json", stem))];
    candidates.into_iter().find(|p| p.is_file())
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1_000_000_000.0) } else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1_000_000.0) } else if n >= 1_000 { format!("{:.1}K", n as f64 / 1_000.0) } else { n.to_string() }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 { format!("{:.2} GB", bytes as f64 / 1024_f64.powi(3)) } else { format!("{:.0} MB", bytes as f64 / 1024_f64.powi(2)) }
}

fn load_llama(path: &Path, device: &Device) -> Result<(ModelWeights, Tokenizer, gguf_file::Content, File)> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)?;
    let architecture = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
    if !architecture.eq_ignore_ascii_case("llama") { bail!("Candle's Phase 1 runtime currently loads Llama-family GGUFs. Detected architecture: {}.", architecture); }
    let tokenizer = load_tokenizer(&content, path)?;
    let model = ModelWeights::from_gguf(content.clone(), &mut file, device)?;
    Ok((model, tokenizer, content, file))
}

fn make_sampling(temperature: f64, top_p: f64, top_k: usize, seed: u64) -> LogitsProcessor {
    let sampling = if temperature <= 0.0 { Sampling::ArgMax } else if top_k > 0 { Sampling::TopKThenTopP { k: top_k, p: top_p.clamp(0.05, 1.0), temperature } } else if top_p < 1.0 { Sampling::TopP { p: top_p.clamp(0.05, 1.0), temperature } } else { Sampling::All { temperature } };
    LogitsProcessor::from_sampling(seed, sampling)
}

#[tauri::command]
pub fn cancel_generation(state: State<'_, EngineState>) -> bool {
    state.cancel.store(true, Ordering::SeqCst); true
}

#[tauri::command]
pub fn start_generation(app: AppHandle, state: State<'_, EngineState>, request: GenerationConfig) -> Result<GenerationResult, String> {
    state.cancel.store(false, Ordering::SeqCst);
    generate_inner(&app, &state.cancel, request).map_err(|e| e.to_string())
}

fn generate_inner(app: &AppHandle, cancel: &AtomicBool, cfg: GenerationConfig) -> Result<GenerationResult> {
    let path = Path::new(&cfg.model_path);
    let meta = fs::metadata(path)?;
    if meta.len() > MAX_MODEL_BYTES { bail!("Model exceeds 2 GB."); }
    let device = if cfg!(feature = "cuda") { Device::new_cuda(0).unwrap_or(Device::Cpu) } else { Device::Cpu };
    let (mut model, tokenizer, content, _file) = load_llama(path, &device)?;
    let mut tokens = tokenizer.encode(cfg.prompt.as_str(), true).map_err(|e| anyhow::anyhow!(e.to_string()))?.get_ids().to_vec();
    if tokens.len() + cfg.max_tokens >= cfg.context_length { tokens.truncate(cfg.context_length.saturating_sub(cfg.max_tokens + 1)); }
    let mut sampler = make_sampling(cfg.temperature, cfg.top_p, cfg.top_k, cfg.seed);
    let eos = ["</s>", "<|end_of_text|>", "<|eot_id|>", "<|im_end|>"].iter().filter_map(|t| tokenizer.get_vocab(true).get(*t).copied()).collect::<Vec<_>>();
    let mut generated = Vec::<u32>::new();
    let start = Instant::now();
    let mut position = 0usize;
    let mut next = {
        let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, position)?.squeeze(0)?;
        sampler.sample(&logits)?
    };
    for _ in 0..cfg.max_tokens {
        if cancel.load(Ordering::SeqCst) { break; }
        generated.push(next);
        tokens.push(next);
        let text = tokenizer.decode(&generated, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let _ = app.emit("generation-token", TokenEvent { request_id: cfg.request_id.clone(), text });
        if eos.contains(&next) { break; }
        position += 1;
        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, position)?.squeeze(0)?;
        let logits = if (cfg.repeat_penalty - 1.0).abs() < f32::EPSILON { logits } else { candle_transformers::utils::apply_repeat_penalty(&logits, cfg.repeat_penalty, &tokens[tokens.len().saturating_sub(64)..])? };
        next = sampler.sample(&logits)?;
        if tokens.len() >= cfg.context_length { break; }
    }
    let seconds = start.elapsed().as_secs_f64();
    let text = tokenizer.decode(&generated, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let token_count = generated.len();
    Ok(GenerationResult { text, tokens: token_count, seconds, tokens_per_sec: if seconds > 0.0 { token_count as f64 / seconds } else { 0.0 }, cancelled: cancel.load(Ordering::SeqCst) })
}

#[tauri::command]
pub fn validate_runtime() -> String {
    let _ = Mutex::new(());
    let _ = std::mem::size_of::<GenerationConfig>();
    "Candle + GGUF + Llama runtime configured".into()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![engine_status, system_stats, inspect_model, cancel_generation, start_generation, validate_runtime])
        .run(tauri::generate_context!())
        .expect("error while running Synth Studio");
}
