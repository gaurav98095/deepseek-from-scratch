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
  --tensorboard-logdir runs/tensorboard \
  --checkpoint checkpoints/deepseek-tiny.pt
```

Open TensorBoard in another terminal:

```bash
tensorboard --logdir runs/tensorboard
```

Then open the displayed local URL. The training script records loss,
perplexity, learning rate, gradient norm, throughput, step time, token counts,
logit statistics, model parameter counts, and CUDA memory when using CUDA.

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

## Inference

Inference does not download a model checkpoint. A checkpoint is created only
after training. For the local model, run:

```bash
make train-model-dev
```

That command writes the checkpoint to:

```text
checkpoints/deepseek-tiny.pt
```

Then generate from the newest local checkpoint:

```bash
make inference INFERENCE_PROMPT="Once upon a time"
```

Inference automatically builds a per-layer KV cache during prompt prefill.
After that first pass, each normal decoding step processes only the newest
token. The cache exists only in runtime memory and is not written to the
checkpoint.

Each generation step prints the live cache size:

```text
generation_step=1 kv_cache_tokens_per_layer=18 kv_cache_elements=36864 kv_cache_bytes=147456 kv_cache_mib=0.1406
```

The reported bytes come from the actual K/V tensors and therefore reflect the
loaded model's layer count, dimensions, context length, and tensor dtype.

If the checkpoint was deleted, repeat `make train-model-dev`; it must be
trained again because model weights are not stored in the repository.

The inference command uses `configs/tiny_train.json` by default. Use the
matching configuration when loading a production checkpoint:

```bash
make inference \
  INFERENCE_CONFIG=configs/tiny_train.json \
  CHECKPOINT=checkpoints/deepseek-tiny.pt \
  INFERENCE_PROMPT="Once upon a time" \
  INFERENCE_TOKENS=100 \
  INFERENCE_TEMPERATURE=0.7
```

The checkpoint must have been trained with the same architecture configuration.
The inference entry point is the root-level [inference.py](./inference.py),
outside the `src/` directory.
The current byte tokenizer is intentionally the simple tokenizer used by this
educational stage.

## Tests

```bash
pytest
pytest tests/test_prepare_tinystories.py
pytest tests/test_attention.py
pytest tests/test_inference.py
```

## Checkpoints

Training saves model weights, optimizer state, scheduler state, and global-step
state to:

```text
checkpoints/deepseek-tiny.pt
```

The training command writes model, optimizer, scheduler, and global-step state
to the checkpoint. The current training pipeline does not resume optimizer
state from an older checkpoint. The root-level `inference.py` loads the saved
`model` weights for text generation.

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
feat(inference): add KV-cached generation
```

Each implementation step should update its numbered document before committing.

## Makefile shortcuts

The Makefile provides a short local workflow and explicit production targets:

```bash
make help
make setup
make prepare-data
make train-model-dev
make inference INFERENCE_PROMPT="Once upon a time"
make tensorboard
make test
```

`make prepare-data` streams the full TinyStories training split to
`data/processed/tinystories-train.jsonl`. It can be run ahead of time or on
another machine and copied to the production training environment. Production
training will prepare it automatically if it is missing.

```bash
make train-model-prod
```

Development training uses a small dataset and the tiny config. Production
training uses the full dataset, the DeepSeek reference config, and CUDA by
default. Override production paths or training parameters for your environment:

```bash
make train-model-prod \
  PROD_DATA=/data/tinystories-train.jsonl \
  PROD_STEPS=10000 \
  PROD_WARMUP_STEPS=500 \
  PROD_BATCH_SIZE=2
```

Inference defaults to the development config. When using a production
checkpoint, select its matching config and checkpoint explicitly.
