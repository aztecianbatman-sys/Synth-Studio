use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::{Deserialize, Serialize};
use std::{fs::{self, File}, path::{Path, PathBuf}, sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Instant};
use sysinfo::System;
use tauri::{AppHandle, Emitter, State};
use tokenizers::Tokenizer;

const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Default)]
pub struct EngineState { cancel: Arc<AtomicBool> }

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct GenerationResult { pub text: String, pub tokens: usize, pub seconds: f64, pub tokens_per_sec: f64, pub cancelled: bool }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
struct TokenEvent { request_id: String, text: String }

#[tauri::command]
fn engine_status() -> String { "Candle Rust engine ready".into() }

#[tauri::command]
fn system_stats() -> SystemStats {
    let mut system = System::new_all();
    system.refresh_all();
    let total = system.total_memory() / 1024 / 1024;
    let used = system.used_memory() / 1024 / 1024;
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let process_memory = system.process(pid).map(|p| p.memory() / 1024 / 1024).unwrap_or(0);
    SystemStats {
        cpu_usage: system.global_cpu_usage(), memory_used_mb: used, memory_total_mb: total, process_memory_mb: process_memory,
        gpu_available: false, gpu_note: "Candle CPU backend is bundled by default. CUDA is an optional build feature.".into(),
    }
}

#[tauri::command]
fn inspect_model(path: String) -> Result<ModelInfo, String> { inspect_model_inner(Path::new(&path)).map_err(|e| e.to_string()) }

fn inspect_model_inner(path: &Path) -> Result<ModelInfo> {
    if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("gguf")) != Some(true) { bail!("Synth Studio accepts GGUF model files."); }
    let meta = fs::metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    if !meta.is_file() { bail!("Selected model is not a file."); }
    if meta.len() > MAX_MODEL_BYTES { bail!("Model exceeds Synth Studio's 2 GB lightweight model limit."); }
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let architecture = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| "unknown".into());
    let quantization = content.metadata.get("general.file_type").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| "GGUF".into());
    let context = content.metadata.get(&format!("{}.context_length", architecture)).and_then(|v| v.to_u64().ok()).map(|v| v as usize).unwrap_or(4096);
    let parameter_count = content.metadata.get("general.parameter_count").and_then(|v| v.to_u64().ok()).map(format_count);
    let tokenizer = if has_embedded_tokenizer(&content) { "embedded" } else if sidecar_tokenizer(path).is_some() { "sidecar" } else { "missing" };
    let status = if architecture.eq_ignore_ascii_case("llama") { if tokenizer == "missing" { "missing" } else { "ready" } } else { "unsupported" };
    let name = content.metadata.get("general.name").and_then(|v| v.to_string().ok()).cloned().unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or("GGUF model").into());
    Ok(ModelInfo { id: format!("{}-{}", meta.len(), path.display()), name, path: path.display().to_string(), size: format_size(meta.len()), bytes: meta.len(), architecture, quantization, context, status: status.into(), parameter_count, tokenizer: tokenizer.into() })
}

fn has_embedded_tokenizer(content: &gguf_file::Content) -> bool {
    ["tokenizer.huggingface.json", "tokenizer.ggml.hf_json", "tokenizer.huggingface.tokenizer_json"].iter().any(|key| content.metadata.get(*key).and_then(|v| v.to_string().ok()).is_some())
}
fn sidecar_tokenizer(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = model_path.file_stem()?.to_string_lossy();
    [parent.join("tokenizer.json"), parent.join(format!("{}.tokenizer.json", stem))].into_iter().find(|p| p.is_file())
}
fn load_tokenizer(content: &gguf_file::Content, model_path: &Path) -> Result<Tokenizer> {
    for key in ["tokenizer.huggingface.json", "tokenizer.ggml.hf_json", "tokenizer.huggingface.tokenizer_json"] {
        if let Some(json) = content.metadata.get(key).and_then(|v| v.to_string().ok()) { return Tokenizer::from_bytes(json.as_bytes()).map_err(|e| anyhow::anyhow!(e.to_string())); }
    }
    if let Some(sidecar) = sidecar_tokenizer(model_path) { return Tokenizer::from_file(sidecar).map_err(|e| anyhow::anyhow!(e.to_string())); }
    bail!("No tokenizer found. Put tokenizer.json beside the GGUF (or <model>.tokenizer.json) and retry.")
}
fn format_count(n: u64) -> String { if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) } else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) } else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) } else { n.to_string() } }
fn format_size(bytes: u64) -> String { if bytes >= 1024 * 1024 * 1024 { format!("{:.2} GB", bytes as f64 / 1024_f64.powi(3)) } else { format!("{:.1} MB", bytes as f64 / 1024_f64.powi(2)) } }

fn load_llama(path: &Path, device: &Device) -> Result<(ModelWeights, Tokenizer)> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)?;
    let architecture = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
    if !architecture.eq_ignore_ascii_case("llama") { bail!("Candle runtime currently targets Llama-family GGUF. Detected architecture: {}.", architecture); }
    let tokenizer = load_tokenizer(&content, path)?;
    let model = ModelWeights::from_gguf(content, &mut file, device)?;
    Ok((model, tokenizer))
}

#[cfg(feature = "cuda")]
fn select_device() -> Device { Device::new_cuda(0).unwrap_or(Device::Cpu) }
#[cfg(not(feature = "cuda"))]
fn select_device() -> Device { Device::Cpu }

fn make_sampling(temperature: f64, top_p: f64, top_k: usize, seed: u64) -> LogitsProcessor {
    let sampling = if temperature <= 0.0 { Sampling::ArgMax } else if top_k > 0 { Sampling::TopKThenTopP { k: top_k, p: top_p.clamp(0.05, 1.0), temperature } } else if top_p < 1.0 { Sampling::TopP { p: top_p.clamp(0.05, 1.0), temperature } } else { Sampling::All { temperature } };
    LogitsProcessor::from_sampling(seed, sampling)
}

#[tauri::command]
fn cancel_generation(state: State<'_, EngineState>) -> bool { state.cancel.store(true, Ordering::SeqCst); true }

#[tauri::command]
fn start_generation(app: AppHandle, state: State<'_, EngineState>, request: GenerationConfig) -> Result<GenerationResult, String> {
    state.cancel.store(false, Ordering::SeqCst);
    generate_inner(&app, &state.cancel, request).map_err(|e| e.to_string())
}

fn generate_inner(app: &AppHandle, cancel: &AtomicBool, cfg: GenerationConfig) -> Result<GenerationResult> {
    let path = Path::new(&cfg.model_path);
    let meta = fs::metadata(path)?;
    if meta.len() > MAX_MODEL_BYTES { bail!("Model exceeds 2 GB."); }
    let device = select_device();
    let (mut model, tokenizer) = load_llama(path, &device)?;
    let encoded = tokenizer.encode(cfg.prompt.as_str(), true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut prompt_tokens = encoded.get_ids().to_vec();
    let context = cfg.context_length.clamp(512, 8192);
    let max_tokens = cfg.max_tokens.clamp(1, 2048);
    if prompt_tokens.len() + max_tokens >= context { prompt_tokens.truncate(context.saturating_sub(max_tokens + 1)); }
    let mut sampler = make_sampling(cfg.temperature, cfg.top_p, cfg.top_k, cfg.seed);
    let vocab = tokenizer.get_vocab(true);
    let eos = ["</s>", "<|end_of_text|>", "<|eot_id|>", "<|im_end|>"].iter().filter_map(|t| vocab.get(*t).copied()).collect::<Vec<_>>();
    let mut generated = Vec::<u32>::new();
    let start = Instant::now();
    let mut position = 0usize;
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, position)?.squeeze(0)?;
    let mut next = sampler.sample(&logits)?;
    for _ in 0..max_tokens {
        if cancel.load(Ordering::SeqCst) { break; }
        generated.push(next);
        let text = tokenizer.decode(&generated, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let _ = app.emit("generation-token", TokenEvent { request_id: cfg.request_id.clone(), text });
        if eos.contains(&next) { break; }
        prompt_tokens.push(next);
        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, position + 1)?.squeeze(0)?;
        if (cfg.repeat_penalty - 1.0).abs() >= f32::EPSILON { let start_at = prompt_tokens.len().saturating_sub(64); logits = candle_transformers::utils::apply_repeat_penalty(&logits, cfg.repeat_penalty, &prompt_tokens[start_at..])?; }
        next = sampler.sample(&logits)?;
        position += 1;
        if prompt_tokens.len() >= context { break; }
    }
    let seconds = start.elapsed().as_secs_f64();
    let text = tokenizer.decode(&generated, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokens = generated.len();
    Ok(GenerationResult { text, tokens, seconds, tokens_per_sec: if seconds > 0.0 { tokens as f64 / seconds } else { 0.0 }, cancelled: cancel.load(Ordering::SeqCst) })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().with_shortcuts(["CommandOrControl+Shift+Space"]).expect("valid shortcut").with_handler(|app, _shortcut, event| {
            use tauri_plugin_global_shortcut::ShortcutState;
            if event.state == ShortcutState::Pressed { let _ = app.emit("synth-quick", "open"); }
        }).build())
        .invoke_handler(tauri::generate_handler![engine_status, system_stats, inspect_model, cancel_generation, start_generation])
        .run(tauri::generate_context!())
        .expect("error while running Synth Studio");
}
