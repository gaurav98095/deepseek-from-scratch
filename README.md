# DeepSeek From Scratch

A clean PyTorch workspace for building and understanding a small DeepSeek-style language model from the ground up.

Start with [Article 1 — Understanding DeepSeek](docs/01-understanding-deepseek.md), then use the notebook and implementation files alongside it.

## Repository layout

- `configs/` — model and experiment configuration files.
- `data/` — tokenization, dataset preparation, and data loading.
- `model/` — model components such as RMSNorm, RoPE, attention, MLA, MoE, and transformer blocks.
- `training/` — losses, optimization, training, evaluation, and checkpoints.
- `inference/` — KV-cache handling and text generation.
- `tests/` — shape, numerical, gradient, and integration tests.
- `docs/` — written explanations and architecture diagrams.
- `notebooks/` — executable experiments that use the implementation in `model/`.
- `train.py` — entry point for training runs.

The implementation will stay readable and modular: each component should be understandable on its own, tested, and then integrated into the growing model.

## Configurations

- `configs/tiny.yaml` — small configuration for fast local checks.
- `configs/mini.yaml` — larger configuration for GPU experiments.

Generated datasets, checkpoints, logs, and Python cache files are kept out of version control.
