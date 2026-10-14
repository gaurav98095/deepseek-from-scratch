# Project commands

Run these commands from the project root.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Download training data

Downloads a 100 MiB TinyStories corpus and skips the download when it already exists.

```bash
./scripts/download_data.sh
```

Download a larger 1 GiB corpus:

```bash
./scripts/download_data.sh \
  --min-bytes $((1000 * 1024 * 1024)) \
  --max-bytes $((1024 * 1024 * 1024))
```

## Train

```bash
./scripts/run.sh \\
  --data data/tinystories-train.txt \\
  --steps 1000 \\
  --run-dir runs/experiment-1
```

Useful overrides:

```bash
./scripts/run.sh \\
  --batch-size 16 \\
  --context-length 128 \\
  --embedding-dim 128 \\
  --num-layers 4 \\
  --num-heads 4 \\
  --learning-rate 3e-4 \\
  --eval-interval 100
```

Resume from the latest checkpoint:

```bash
./scripts/run.sh \\
  --resume runs/experiment-1/last.pt \\
  --run-dir runs/experiment-1
```

## TensorBoard

```bash
tensorboard --logdir runs
```

Open the displayed local URL, usually `http://localhost:6006`.

Logged metrics include training and validation loss, perplexity, token accuracy, learning rate, gradient norm, parameter norm, tokens per second, step time, and tokens seen.

## Inference

```bash
./scripts/infer.sh \\
  --checkpoint runs/experiment-1/last.pt \\
  --prompt "Once upon a time" \\
  --max-new-tokens 120 \\
  --temperature 0.8 \\
  --top-k 50 \\
  --top-p 0.95
```

Greedy decoding:

```bash
./scripts/infer.sh \\
  --checkpoint runs/experiment-1/last.pt \\
  --prompt "Once upon a time" \\
  --temperature 0
```

## Git

After installing macOS Command Line Tools if `git` reports an `xcrun` error:

```bash
git add -A
git commit -m "Add training pipeline and inference setup"
git push origin main
```
