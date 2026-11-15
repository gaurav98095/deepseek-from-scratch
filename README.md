# DeepSeek From Scratch

A clean PyTorch workspace for building and understanding a small DeepSeek-style language model from the ground up.

Start with [Article 1 — Understanding DeepSeek](docs/01-understanding-deepseek.md), then read [Setting Up the DeepSeek Codebase](docs/02-setting-up-codebase.md) while using the notebook and implementation files alongside them.

## Repository layout

- `src/model/` — model configuration, embeddings, and DeepSeek components.
- `src/tests/` — shape, numerical, gradient, and integration tests.
- `src/smoke_test.py` — small forward-pass check.
- `docs/` — written explanations and architecture diagrams.
- `notebooks/` — executable experiments and tensor-shape demonstrations.
- `requirements.txt` — Python dependencies.

The implementation will stay readable and modular: each component should be understandable on its own, tested, and then integrated into the growing model.

Generated datasets, checkpoints, logs, and Python cache files are kept out of version control.
