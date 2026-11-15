# DeepSeek From Scratch

A clean PyTorch workspace for building and understanding a small DeepSeek-style language model from the ground up.

Start with [Article 1 — Understanding DeepSeek](docs/01-understanding-deepseek.md), continue with [Setting Up the DeepSeek Codebase](docs/02-setting-up-codebase.md), and then read [Building a Tokenizer From Scratch](docs/03-build-tokenizer-from-scratch.md) alongside the implementation and notebooks.

## Repository layout

- `src/model/` — model configuration, embeddings, and DeepSeek components.
- `src/data/` — byte-level BPE tokenizer and data-processing components.
- `src/tests/` — shape, numerical, gradient, and integration tests.
- `src/commands/` — commands for notebooks, smoke tests, and each article's checks.
- `src/smoke_test.py` — small forward-pass check.
- `src/train_tokenizer.py` — tokenizer training demonstration.
- `docs/` — written explanations and architecture diagrams.
- `notebooks/` — executable experiments and tensor-shape demonstrations.
- `requirements.txt` — Python dependencies.

The implementation will stay readable and modular: each component should be understandable on its own, tested, and then integrated into the growing model.

Generated datasets, checkpoints, logs, and Python cache files are kept out of version control.
