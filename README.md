# NEFM (Neural Enhanced Framework Model)

An experimental Large Language Model (LLM) with KV-cache support, written in Rust using the **Burn** framework and **WGPU** (WebGPU) backend.

## Features
- **Architecture:** Transformer Block with Rotary Position Embeddings (RoPE).
- **Optimizations:** KV-Cache support during inference for accelerated text generation.
- **Training:** Built-in supervised training loop with a Cosine Annealing LR scheduler.
- **Tokenizer:** Custom Byte Pair Encoding (BPE) tokenizer powered by the `tokenizers` crate.

## Getting Started

### Installation
Go to [releases](https://github.com/NopeEnemy/Nefm/releases) and install the latest version or compile sources.

Instruction if you want compile sources:
1. Install [Rust](https://rust-lang.org/tools/install/), if you haven't.
2. Get the sources
```bash
git clone https://github.com/NopeEnemy/Nefm
cd Nefm
```
3. Build (this may take some minutes)
```
cargo build --release
```
4. Pick up the binary from .../Nefm/target/release

### Using

1. Train the Tokenizer
```bash
nefm tokenizer ./artifacts 32000
```

2. Train the Model
```bash
nefm train ./artifacts 10 6 8 256 1024 512 32 4 42 0.0001 0.00001 ./artifacts/tokenizer.json
```
(Arguments: dir, epochs, layers, heads, d_model, d_hidden, context, batch, workers, seed, lr, min_lr, tokenizer_path)


3. Model Inference (Text Generation)
```bash
nefm run ./artifacts 0.7 50 "Hello, I am a language model"
```

4. View Model Parameters
```bash
nefm params ./artifacts
```
