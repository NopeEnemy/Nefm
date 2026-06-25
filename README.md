# NEFM (Neural Enhanced Framework Model)

An experimental Large Language Model (LLM) with KV-cache support, written in Rust using the **Burn** framework and **WGPU** (WebGPU) backend.

## Features
- **Architecture:** Transformer Block with Rotary Position Embeddings (RoPE).
- **Optimizations:** KV-Cache support during inference for accelerated text generation.
- **Training:** Built-in supervised training loop with a Cosine Annealing LR scheduler.
- **Tokenizer:** Custom Byte Pair Encoding (BPE) tokenizer powered by the `tokenizers` crate.

## Getting Started

### 1. Train the Tokenizer
```bash
cargo run --release -- tokenizer ./artifacts 32000
```

2. Train the Model
```bash
cargo run --release -- train ./artifacts 10 6 8 256 1024 512 32 4 42 0.0001 0.00001 ./artifacts/tokenizer.json
```
(Arguments: dir, epochs, layers, heads, d_model, d_hidden, context, batch, workers, seed, lr, min_lr, tokenizer_path)


3. Model Inference (Text Generation)
```bash
cargo run --release -- run ./artifacts 0.7 50 "Hello, I am a language model"
```

4. View Model Parameters
```bash
cargo run --release -- params ./artifacts
```
