# Synth Studio

A lightweight, local-first AI studio by Fulltrack.

## Phase 1

- Tauri 2 + Rust core
- React + TypeScript UI
- GGUF/llama.cpp integration foundation
- Local conversations, personas, model management, settings, files, and system monitoring
- Light theme by default with an Ollama-inspired, Apple-clean visual language

## Development

```bash
npm install
npm run tauri dev
```

The current Phase 1 UI includes the complete application shell and a native Rust command bridge. The local inference adapter is structured for llama.cpp/GGUF integration and reports a clear unavailable state until the runtime/model is installed.
