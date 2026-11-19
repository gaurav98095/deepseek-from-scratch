# Command Guide

Run these commands from the repository root unless a command says otherwise.

## One-time environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA machine, check the driver and GPU before using CUDA:

```bash
nvidia-smi
```

## Article 1 — Understanding DeepSeek

Open the executable notebook:

```bash
jupyter lab notebooks/01-understanding-deepseek.ipynb
```

The notebook currently demonstrates configuration values, token IDs, embedding lookup, output shapes, and parameter counts.

## Article 2 — Setting Up the Codebase

Run the model skeleton's forward-pass smoke test from the repository root:

```bash
python -m src.smoke_test
```

The same script can be run from inside `src/`:

```bash
cd src
python smoke_test.py
cd ..
```

Run only the model tests:

```bash
python -m pytest src/tests/test_model.py -v
```

These checks cover output shape, parameter count, gradient flow, and sequence-length validation.

## Article 3 — Building a Tokenizer

Train and inspect the toy byte-level BPE tokenizer:

```bash
python -m src.train_tokenizer
```

Run only the tokenizer tests:

```bash
python -m pytest src/tests/test_tokenizer.py -v
```

These checks cover pair merging, text round-tripping, Unicode, EOS handling, vocabulary size, and invalid configuration values.

## Article 4 — Preparing the Pretraining Dataset

Prepare a bounded FineWeb-Edu sample from the repository root:

```bash
python -m src.data.prepare_data \
  --max-documents 5000 \
  --tokenizer-documents 1000 \
  --tokenizer-vocab-size 350
```

Use a smaller smoke run first when checking the pipeline:

```bash
python -m src.data.prepare_data \
  --max-documents 20 \
  --tokenizer-documents 10 \
  --tokenizer-vocab-size 300
```

Inspect the generated files:

```bash
python -m src.data.inspect_data
```

The generated files will live under `data/processed/`:

```text
tokenizer.json
train.bin
val.bin
metadata.json
```

Do not commit generated datasets or binary token files.

## Article 5 — Building the PyTorch DataLoader

The data loader implementation lives under `src/data/dataset.py` and `src/data/dataloader.py`. Run the dataset checks with:

```bash
python -m pytest src/tests/test_dataset.py -v
python -m pytest src/tests/test_dataloader.py -v
```

The loader reads `data/processed/train.bin` and `val.bin` with memory mapping and returns `[batch, sequence]` input-target pairs.

## Article 6 — Token Embeddings and RMSNorm

The RMSNorm implementation lives under `src/model/rmsnorm.py`. Run:

```bash
python -m pytest src/tests/test_rmsnorm.py -v
```

The tests cover the numerical formula, zero inputs, gradient flow, scale behavior, and BF16-safe statistics.

## Article 7 — Causal Self-Attention

The attention layer belongs in `src/model/attention.py`, with transformer integration in `src/model/transformer.py` and focused checks in `src/tests/test_attention.py`. Run the attention checks with:

```bash
python -m pytest src/tests/test_attention.py -v
```

The important checks cover output shapes, causal future-token isolation, finite gradients, and projection parameter counts.

## Article 8 — Rotary Position Embeddings

RoPE is implemented in `src/model/rope.py` and integrated into `src/model/attention.py`. Run the rotation and causal-attention checks with:

```bash
python -m pytest src/tests/test_rope.py -v
```

The checks cover shape preservation, length preservation, relative-position behavior, gradients, and causal masking after adding RoPE.

The notebook adds the repository root to `sys.path` automatically, so its imports work when Jupyter is opened from either the repository root or the `notebooks/` directory.

## Article 9 — Complete GPT-Style Transformer

The dense baseline adds `SwiGLUFFN` in `src/model/ffn.py` and combines it with attention and RMSNorm in `src/model/transformer.py`. Run the focused checks with:

```bash
python -m pytest src/tests/test_transformer.py -v
```

These checks cover the FFN shape, complete block shape, model logits, and gradient flow. The companion notebook is `notebooks/09-first-complete-gpt.ipynb`.

Train the dense baseline after preparing `data/processed`:

```bash
python -m src.train --steps 300 --batch-size 8 --seq-len 128
```

Generate from its checkpoint:

```bash
python -m src.generate --checkpoint checkpoints/gpt_baseline.pt --prompt "The history of"
```

The training and generation scripts are educational baseline implementations. They do not use a KV cache yet.

## Run all current tests

```bash
python -m pytest src/tests -v
```

## Useful checks

Inspect the repository files without running the model:

```bash
find src -maxdepth 3 -type f | sort
git status --short
```

Check Python formatting at the Git level:

```bash
git diff --check
```

## Import and path note

Use the `python -m src...` form from the repository root. It keeps `src` on the import path and makes imports such as `from src.model import DeepSeekConfig` work consistently.

## Current scope

The repository currently contains the complete dense attention-and-RoPE baseline, tokenizer, dataset preparation, data-loading foundations, baseline training, and autoregressive generation. MLA, MoE, KV-cache inference, and other scaling commands will be added when those parts exist in `src/`.
