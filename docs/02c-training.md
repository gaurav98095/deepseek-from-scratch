# Step 02c — Run the complete training and inference pipeline

This step connects the prepared TinyStories JSONL file to the current
attention-based decoder and optimizer. It intentionally uses a small byte
tokenizer and a small model configuration so the pipeline can run locally.

The flow is:

```text
TinyStories JSONL
    → UTF-8 byte tokenizer
    → fixed-length token sequences
    → token and position embeddings
    → causal attention decoder blocks
    → vocabulary logits
    → shifted cross-entropy loss
    → AdamW, warmup/cosine schedule, gradient clipping
    → checkpoint
```

## Run it

First prepare the dataset using [01-data-preparation.md](./01-data-preparation.md).
Then install dependencies and run:

```bash
python -m pip install -r requirements.txt
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

In another terminal, start TensorBoard:

```bash
tensorboard --logdir runs/tensorboard
```

The run records training loss and perplexity, optimizer learning rate and
gradient norm, batch/sequence sizes, step time, tokens per second, total token
count, logits mean and standard deviation, parameter counts, and CUDA memory
metrics when training on CUDA.

For a short smoke run:

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

The command prints periodic loss values and saves model, optimizer, scheduler,
and step state to the checkpoint path.

## Inference from a checkpoint

Inference happens only after a checkpoint has been created by training. The
checkpoint is a local generated artifact and is not downloaded from GitHub or
stored in the repository.

Run the development pipeline first:

```bash
make step-2-dev
```

This creates:

```text
checkpoints/deepseek-tiny.pt
```

The root-level `inference.py` script loads the newest `.pt` file in the
`checkpoints/` directory when no explicit checkpoint is supplied. It rebuilds
the model from the matching JSON configuration, loads the saved model weights,
encodes the prompt with the byte tokenizer, and generates one token at a time.

### `inference.py`

Create this file at the repository root, outside `src/`:

```python
"""Generate text from the newest or explicitly selected model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.byte_tokenizer import ByteTokenizer
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek


def _build_parser() -> argparse.ArgumentParser:
    # Keep inference independent from the training loop.
    parser = argparse.ArgumentParser(description=__doc__)
    # The architecture must match the configuration used to create the checkpoint.
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    # If omitted, the newest checkpoint in this directory is selected by mtime.
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    # The prompt is converted to byte IDs by the current tokenizer.
    parser.add_argument("--prompt", required=True)
    # This controls how many new tokens are appended after the prompt.
    parser.add_argument("--max-new-tokens", type=int, default=50)
    # Temperature 0 uses greedy decoding; higher values produce more variation.
    parser.add_argument("--temperature", type=float, default=1.0)
    # Top-k limits sampling to the most likely candidates when positive.
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _choose_device(requested: str) -> torch.device:
    # Fail clearly instead of silently moving a requested CUDA run to the CPU.
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _latest_checkpoint(directory: Path) -> Path:
    # Modification time selects the most recent saved checkpoint.
    checkpoints = list(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints found in {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> DeepSeek:
    # Construct the same module layout before loading the saved tensors.
    config = ModelConfig.from_json(config_path)
    model = DeepSeek(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Training checkpoints contain optimizer state as well as model weights.
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def generate(
    model: DeepSeek,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Append tokens to ``prompt`` and return the complete decoded text."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    device = next(model.parameters()).device
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # Keep only the newest context if generation reaches the position limit.
        context = token_ids[:, -model.config.max_position_embeddings :]
        logits = model(context).logits[:, -1, :]

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                limit = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, limit)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        token_ids = torch.cat((token_ids, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids[0].tolist())


def main() -> None:
    # Parse options once so command-line use and future integrations share behavior.
    args = _build_parser().parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)
    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    model = _load_model(args.config, checkpoint, device)
    text = generate(
        model,
        ByteTokenizer(),
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    print(f"Checkpoint: {checkpoint}")
    print(text)


if __name__ == "__main__":
    main()
```

Run inference through the Makefile:

```bash
make infer INFERENCE_PROMPT="Once upon a time"
```

For a specific checkpoint and longer generation:

```bash
make infer \
  INFERENCE_CONFIG=configs/tiny_train.json \
  CHECKPOINT=checkpoints/deepseek-tiny.pt \
  INFERENCE_PROMPT="The little girl went" \
  INFERENCE_TOKENS=100 \
  INFERENCE_TEMPERATURE=0.7
```

The checkpoint must be loaded with the same architecture configuration used
during training. If the checkpoint is deleted, run `make step-2-dev` again.

## Exact files in this step

### `configs/tiny_train.json`

Create this file exactly:

```json
{
  "model_name": "DeepSeek-Tiny-CurrentScaffold",
  "model_type": "decoder_only",
  "implementation_profile": "standard_attention_educational",
  "vocab_size": 258,
  "max_position_embeddings": 129,
  "hidden_size": 256,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "num_key_value_heads": 8,
  "head_dim": 32,
  "intermediate_size": 1024,
  "hidden_activation": "silu",
  "attention_dropout": 0.0,
  "attention_bias": false,
  "position_embedding_type": "learned_absolute",
  "initializer_range": 0.02,
  "rms_norm_eps": 0.000001,
  "torch_dtype": "float32",
  "tie_word_embeddings": false
}
```

### `src/data/byte_tokenizer.py`

This first tokenizer maps UTF-8 bytes directly to token IDs. It has 256 byte
tokens plus beginning-of-sequence ID 256 and end-of-sequence ID 257.

```python
"""A dependency-free byte tokenizer for the first runnable training pipeline."""

from __future__ import annotations


class ByteTokenizer:
    """Encode UTF-8 bytes plus BOS/EOS tokens for a small demo vocabulary."""

    bos_token_id = 256
    eos_token_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = [token for token in tokens if 0 <= token < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")
```

### `src/data/tinystories.py`

This dataset iterator reads prepared JSONL records, tokenizes each story, and
returns only complete sequences of `block_size + 1` tokens. The extra token is
needed because the model shifts inputs and labels by one position.

```python
"""Streaming TinyStories records into fixed-length language-model batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .byte_tokenizer import ByteTokenizer


class TinyStoriesDataset(IterableDataset[Tensor]):
    """Yield fixed-length token sequences from prepared TinyStories JSONL."""

    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, block_size: int) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self) -> Iterator[Tensor]:
        with self.path.open(encoding="utf-8") as data_file:
            for line in data_file:
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str):
                    continue
                tokens = self.tokenizer.encode(text)
                for start in range(0, len(tokens) - self.block_size, self.block_size):
                    sequence = tokens[start : start + self.block_size + 1]
                    if len(sequence) == self.block_size + 1:
                        yield torch.tensor(sequence, dtype=torch.long)
```

### `src/train.py`

The command-line entry point selects a device, loads the configuration, builds
the model and trainer, loops until `--steps`, and saves a checkpoint.

```python
"""Run the first complete DeepSeek training pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.byte_tokenizer import ByteTokenizer
from .data.tinystories import TinyStoriesDataset
from .model.config import ModelConfig
from .model.deepseek import DeepSeek, DeepSeekTrainer, TrainingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/tinystories-train.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepseek-tiny.pt"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)

    config = ModelConfig.from_json(args.config)
    if args.block_size + 1 > config.max_position_embeddings:
        raise ValueError("block_size + 1 must not exceed max_position_embeddings in the config")

    tokenizer = ByteTokenizer()
    # The byte tokenizer uses the first 258 IDs; a larger model vocabulary is allowed.
    if config.vocab_size < tokenizer.vocab_size:
        raise ValueError(
            f"config vocab_size ({config.vocab_size}) must be at least the "
            f"tokenizer vocab_size ({tokenizer.vocab_size})"
        )

    model = DeepSeek(config).to(device)
    trainer = DeepSeekTrainer(
        model,
        TrainingConfig(learning_rate=args.learning_rate, max_steps=args.steps),
    )
    dataset = TinyStoriesDataset(args.data, tokenizer, args.block_size)
    batches = DataLoader(dataset, batch_size=args.batch_size)

    while trainer.global_step < args.steps:
        made_progress = False
        for input_ids in batches:
            made_progress = True
            output = trainer.train_batch(input_ids.to(device))
            if trainer.global_step % args.log_interval == 0:
                print(f"step={trainer.global_step} loss={output.loss.item():.4f}")
            if trainer.global_step >= args.steps:
                break
        if not made_progress:
            raise RuntimeError("the dataset produced no complete training sequences")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


def main() -> None:
    parser = _build_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
```

## What is and is not included

Included in this runnable stage:

- TinyStories JSONL reading
- UTF-8 byte tokenization
- Fixed-length batches
- Token and positional embeddings
- Repeated causal attention decoder blocks
- Feed-forward layers and residual connections
- Vocabulary logits and shifted language-model loss
- AdamW, warmup/cosine scheduling, gradient clipping
- Checkpoint save/load
- TensorBoard metric logging

The large `configs/deepseek_v3.json` remains a dimension reference. Use
`configs/tiny_train.json` for local execution until the later scaling and data
engineering stages are implemented.

## Commit this step

```text
feat(train): add runnable attention-based training pipeline
```

```bash
git add configs/tiny_train.json src/data/byte_tokenizer.py src/data/tinystories.py src/train.py inference.py src/data/__init__.py README.md docs/02c-training.md
git commit -m "feat(train): add runnable attention-based training pipeline"
```
