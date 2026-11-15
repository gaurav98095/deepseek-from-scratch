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

The repository currently contains the model skeleton and tokenizer work. Full attention, MLA, MoE, dataset preparation, training, and inference commands will be added when those parts exist in `src/`.
