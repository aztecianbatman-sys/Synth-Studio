use serde::Serialize;
use std::{fs, path::Path};

#[derive(Debug, Serialize)]
struct ModelInfo {
    name: String,
    size: String,
    status: String,
    quant: String,
    architecture: String,
}

#[tauri::command]
fn engine_status() -> String {
    "Native Rust bridge ready".to_string()
}

#[tauri::command]
fn local_inference(prompt: String, model: String) -> Result<String, String> {
    if model == "No model imported" {
        return Err("No GGUF model is installed. Open Models and import a compatible .gguf file.".into());
    }
    Err(format!("The inference runtime is ready to be connected to llama.cpp for model '{model}'. Prompt received ({} characters).", prompt.chars().count()))
}

#[tauri::command]
fn register_model(path: String) -> Result<ModelInfo, String> {
    let p = Path::new(&path);
    if p.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("gguf")) != Some(true) {
        return Err("Synth Studio accepts GGUF model files.".into());
    }
    if !p.is_file() {
        return Err("The selected model file could not be found.".into());
    }
    let metadata = fs::metadata(p).map_err(|e| e.to_string())?;
    let size = format_size(metadata.len());
    let name = p.file_stem().and_then(|x| x.to_str()).unwrap_or("Imported GGUF").to_string();
    Ok(ModelInfo { name, size, status: "installed".into(), quant: "GGUF".into(), architecture: "Detected at runtime".into() })
}

fn format_size(bytes: u64) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    if bytes as f64 >= GB { format!("{:.2} GB", bytes as f64 / GB) }
    else { format!("{:.0} MB", bytes as f64 / MB) }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![engine_status, local_inference, register_model])
        .run(tauri::generate_context!())
        .expect("error while running Synth Studio");
}
