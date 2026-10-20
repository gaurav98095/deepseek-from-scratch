# Commands reference

Run these commands from the repository root:

```bash
cd /Users/gaurav/Desktop/deepseek-from-scratch
```

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation

Full TinyStories training split:

```bash
python -m src.data.prepare_tinystories \
  --split train \
  --output data/processed/tinystories-train.jsonl \
  --cache-dir data/cache
```

Small smoke-test dataset:

```bash
python -m src.data.prepare_tinystories \
  --max-stories 100 \
  --output data/processed/tinystories-smoke.jsonl \
  --cache-dir data/cache
```

## Training

Short CPU smoke test:

```bash
python -m src.train \
  --data data/processed/tinystories-smoke.jsonl \
  --config configs/tiny_train.json \
  --steps 10 \
  --warmup-steps 0 \
  --batch-size 2 \
  --block-size 128 \
  --device cpu
```

Local training configuration:

```bash
python -m src.train \
  --data data/processed/tinystories-train.jsonl \
  --config configs/tiny_train.json \
  --steps 1000 \
  --batch-size 8 \
  --block-size 128 \
  --device auto \
  --checkpoint checkpoints/deepseek-tiny.pt
```

CUDA explicitly:

```bash
python -m src.train \
  --data data/processed/tinystories-train.jsonl \
  --config configs/tiny_train.json \
  --device cuda
```

The full-scale reference configuration is `configs/deepseek_v3.json`. It is a
dimension reference and should not be instantiated on a normal personal
machine.

## Tests

```bash
pytest
pytest tests/test_prepare_tinystories.py
pytest tests/test_attention.py
```

## Checkpoints

Training saves model weights, optimizer state, scheduler state, and global-step
state to:

```text
checkpoints/deepseek-tiny.pt
```

Checkpoint loading is available through `DeepSeekTrainer.load_checkpoint` in
`src/model/deepseek.py`.

## Git workflow

```bash
git status --short
git branch --list
```

Create and publish the persistent data-preparation branch:

```bash
git switch -c 01-data-preparation
git push -u origin 01-data-preparation
```

Return to main, merge it, and publish main:

```bash
git switch main
git merge --ff-only 01-data-preparation
git push origin main
```

Create and publish a later implementation branch:

```bash
git switch -c 02-model-integration
git add .
git commit -m "feat(model): add current model integration"
git push -u origin 02-model-integration
```

Inspect history:

```bash
git log --oneline --decorate --all
```

## Commit conventions

```text
feat(data): add streaming TinyStories preparation
feat(config): add centralized DeepSeek model configuration
feat(model): add standard causal multi-head attention
feat(train): add runnable attention-based training pipeline
```

Each implementation step should update its numbered document before committing.
