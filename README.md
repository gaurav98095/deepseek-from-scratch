# DeepSeek From Scratch

A clean PyTorch workspace for building and understanding a small DeepSeek-style language model from the ground up.

Start with [Article 1 — Understanding DeepSeek](docs/01-understanding-deepseek.md), continue with [Setting Up the DeepSeek Codebase](docs/02-setting-up-codebase.md), then read [Building a Tokenizer From Scratch](docs/03-build-tokenizer-from-scratch.md), [Preparing the Pretraining Dataset](docs/04-prepare-pretraining-dataset.md), [Building the PyTorch DataLoader](docs/05-building-data-loader.md), [Token Embeddings and RMSNorm](docs/06-embeddings-and-rms-norm.md), [Causal Self-Attention](docs/07-causal-self-attention.md), [Rotary Position Embeddings](docs/08-rope-from-scratch.md), [Building a Complete GPT-Style Transformer](docs/09-first-complete-gpt.md), [Why DeepSeek Uses MLA](docs/10-why-mla-exists.md), [Low-Rank Query and KV Compression](docs/11-low-rank-q-kv-compression.md), [Decoupled RoPE in MLA](docs/12-decoupled-rope.md), [Complete MLA Integration](docs/13-complete-mla-integration.md), and [Compressed KV Cache and Matrix Absorption](docs/14-absorbed-mla-kv-cache.md) alongside the implementation and notebooks.

## Repository layout

- `src/model/` — model configuration, embeddings, RMSNorm, RoPE, causal attention, SwiGLU, MLA projections, and DeepSeek components.
- `src/experiments/` — small architecture comparison experiments.
- `src/inference/` — KV-cache sizing utilities.
- `src/model/mla_cache.py` — stateful compressed-cache MLA inference.
- `src/data/` — tokenizer, dataset preparation, memory-mapped dataset, and DataLoader tools.
- `src/tests/` — shape, numerical, gradient, and integration tests.
- `commands/` — commands for notebooks, smoke tests, and each article's checks.
- `src/smoke_test.py` — small forward-pass check.
- `src/train_tokenizer.py` — tokenizer training demonstration.
- `src/train.py` — dense-baseline training entry point.
- `src/generate.py` — checkpoint loading and autoregressive generation.
- `docs/` — written explanations and architecture diagrams.
- `notebooks/` — executable experiments and tensor-shape demonstrations.
- `requirements.txt` — Python dependencies.

The implementation will stay readable and modular: each component should be understandable on its own, tested, and then integrated into the growing model.

Generated datasets, checkpoints, logs, and Python cache files are kept out of version control.
