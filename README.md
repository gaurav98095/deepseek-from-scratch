# DeepSeek from scratch — pre-MLA stage

This repository implements the part of the DeepSeek-style decoder covered before Multi-Head Latent Attention:

1. character-level tokenization with an unknown token;
2. token and learned positional embeddings;
3. pre-norm transformer blocks;
4. causal standard multi-head self-attention;
5. GELU feed-forward layers and tied language-model output weights;
6. next-token loss and greedy generation.

RoPE, Multi-Head Latent Attention, MoE, and other later DeepSeek-specific components are intentionally not included yet.

## Run tests

```bash
python -m pytest -q
```

The dimensions are controlled by [`ModelConfig`](deepseek_scratch/config.py). The defaults are a compact teaching configuration; pass the vocabulary size from the tokenizer and override dimensions to match the upstream series configuration when available.

## Train with TensorBoard

Download a decent-sized local training corpus from Hugging Face. The command is idempotent: once the 100 MiB corpus exists, later runs do not download it again.

```bash
chmod +x scripts/download_data.sh scripts/run.sh
./scripts/download_data.sh
```

The default is a 100 MiB stream from [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories). Increase it when you have more disk space:

```bash
./scripts/download_data.sh --max-bytes $((1024 * 1024 * 1024)) --min-bytes $((1024 * 1024 * 1024))
```

To use the local starter corpus instead, pass `--data data/input.txt`. To force replacement of an existing Hugging Face corpus, pass `--force`.

Train with:

```bash
chmod +x scripts/run.sh
./scripts/run.sh --data data/tinystories-train.txt --steps 1000 --run-dir runs/experiment-1
```

In another terminal, launch TensorBoard:

```bash
tensorboard --logdir runs
```

The dashboard records train/validation loss, perplexity, token accuracy, learning rate, gradient norm, parameter norm, tokens per second, step time, and tokens seen. Checkpoints are written to `<run-dir>/last.pt`; resume with `./scripts/run.sh --resume runs/experiment-1/last.pt`.

## Run inference

Every training checkpoint stores the model config and tokenizer vocabulary. Generate from it with:

```bash
./scripts/infer.sh \
  --checkpoint runs/experiment-1/last.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Use `--temperature 0` for greedy decoding. The inference loop keeps context cropping and sampling separate, so a KV cache or Flash Attention implementation can replace the full-prefix forward pass later without changing the user-facing interface.
