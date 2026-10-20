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
    # Keep every run configurable from the command line so the same script can
    # handle a tiny smoke test and a longer training run.
    parser = argparse.ArgumentParser(description=__doc__)
    # JSONL produced by the dataset-preparation step is the training input.
    parser.add_argument("--data", type=Path, default=Path("data/processed/tinystories-train.jsonl"))
    # The model dimensions live in JSON so architecture and training code stay separate.
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    # A step is one optimizer update, not one individual token.
    parser.add_argument("--steps", type=int, default=1000)
    # None lets the code choose a safe warmup for short smoke tests.
    parser.add_argument("--warmup-steps", type=int, default=None)
    # More examples per batch improve throughput but require more memory.
    parser.add_argument("--batch-size", type=int, default=8)
    # Each example contains this many input tokens and one shifted target token.
    parser.add_argument("--block-size", type=int, default=128)
    # This is the peak learning rate reached after warmup.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    # Auto selects CUDA when available and otherwise falls back to the CPU.
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    # Printing periodically makes it possible to monitor progress without noisy output.
    parser.add_argument("--log-interval", type=int, default=10)
    # The checkpoint preserves progress so a later step can restore training.
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepseek-tiny.pt"))
    # Seeding makes data order and initialization more reproducible.
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _choose_device(requested: str) -> torch.device:
    # Fail clearly when the user explicitly requests CUDA but the machine cannot provide it.
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    # Auto mode prefers the accelerator because attention is computationally expensive.
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Otherwise honor the explicit CPU/CUDA choice.
    return torch.device(requested)


def train(args: argparse.Namespace) -> None:
    # A non-positive step count would make the training loop meaningless.
    if args.steps < 1:
        raise ValueError("steps must be positive")
    # Seed Python and PyTorch before creating the model for repeatable initialization.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Select the device once and reuse it for the model and every batch.
    device = _choose_device(args.device)

    # Load the architecture from JSON instead of duplicating dimensions in this script.
    config = ModelConfig.from_json(args.config)
    # The dataset yields block_size input tokens plus one target token for shifting.
    if args.block_size + 1 > config.max_position_embeddings:
        raise ValueError("block_size + 1 must not exceed max_position_embeddings in the config")

    # The first pipeline uses byte IDs, giving a tiny 258-token vocabulary.
    tokenizer = ByteTokenizer()
    # This prevents token IDs from exceeding the embedding table's vocabulary size.
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("the selected config vocab_size must be 258 for ByteTokenizer")

    # Build the decoder and move all parameters to the selected device.
    model = DeepSeek(config).to(device)
    # For short runs, scale warmup down so it never consumes the entire run.
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = min(100, args.steps // 10)
    # The trainer owns AdamW, scheduling, gradients, and checkpoint state.
    trainer = DeepSeekTrainer(
        model,
        TrainingConfig(
            learning_rate=args.learning_rate,
            warmup_steps=warmup_steps,
            max_steps=args.steps,
        ),
    )
    # This iterator reads prepared stories and emits fixed-length token tensors.
    dataset = TinyStoriesDataset(args.data, tokenizer, args.block_size)
    # DataLoader groups individual token sequences into model batches.
    batches = DataLoader(dataset, batch_size=args.batch_size)

    # Repeat the finite dataset until the requested number of optimizer steps is reached.
    while trainer.global_step < args.steps:
        made_progress = False
        for input_ids in batches:
            # Mark progress before training so an empty dataset can be detected below.
            made_progress = True
            # Move the batch to the same device as the model, then run one full update.
            output = trainer.train_batch(input_ids.to(device))
            # Loss is the main signal that training is producing numeric progress.
            if trainer.global_step % args.log_interval == 0:
                print(f"step={trainer.global_step} loss={output.loss.item():.4f}")
            # Stop inside the current epoch as soon as the requested step count is met.
            if trainer.global_step >= args.steps:
                break
        # Without this guard, an empty or overly short dataset would loop forever.
        if not made_progress:
            raise RuntimeError("the dataset produced no complete training sequences")

    # Create the parent directory only after training has produced a checkpoint to save.
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Save weights plus optimizer/scheduler progress for future continuation.
    trainer.save_checkpoint(args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


def main() -> None:
    # Parse command-line options only when this module is used as a script.
    parser = _build_parser()
    train(parser.parse_args())


# This guard keeps the module importable for notebooks and tests.
if __name__ == "__main__":
    main()
