"""Train the current educational decoder on prepared TinyStories data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.byte_tokenizer import ByteTokenizer
from .data.tinystories import TinyStoriesDataset
from .model.config import ModelConfig
from .model.deepseek import DeepSeek, DeepSeekTrainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for one training run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/tinystories-train.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepseek-tiny.pt"))
    parser.add_argument("--tensorboard-logdir", type=Path, default=Path("runs/tensorboard"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def choose_device(requested: str) -> torch.device:
    """Choose CUDA when requested or available, otherwise use CPU."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(args: argparse.Namespace) -> None:
    """Build the data/model pipeline and train until ``args.steps``."""

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.log_interval < 1:
        raise ValueError("log_interval must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = ModelConfig.from_json(args.config)

    if args.block_size + 1 > config.max_position_embeddings:
        raise ValueError("block_size + 1 must not exceed max_position_embeddings in the config")

    tokenizer = ByteTokenizer()
    if config.vocab_size < tokenizer.vocab_size:
        raise ValueError(
            f"config vocab_size ({config.vocab_size}) must be at least the "
            f"tokenizer vocab_size ({tokenizer.vocab_size})"
        )

    model = DeepSeek(config).to(device)
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = min(100, args.steps // 10)
    trainer = DeepSeekTrainer(
        model,
        TrainingConfig(
            learning_rate=args.learning_rate,
            warmup_steps=warmup_steps,
            max_steps=args.steps,
        ),
    )

    dataset = TinyStoriesDataset(args.data, tokenizer, args.block_size)
    batches = DataLoader(dataset, batch_size=args.batch_size)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir))
    writer.add_text("run/model_config", json.dumps(config.to_dict(), indent=2), 0)
    writer.add_text("run/arguments", str(vars(args)), 0)
    writer.add_scalar("model/parameters_total", sum(p.numel() for p in model.parameters()), 0)
    writer.add_scalar(
        "model/parameters_trainable",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        0,
    )

    try:
        while trainer.global_step < args.steps:
            made_progress = False

            for input_ids in batches:
                made_progress = True
                started = time.perf_counter()
                input_ids = input_ids.to(device)
                output = trainer.train_batch(input_ids)
                seconds = time.perf_counter() - started
                step = trainer.global_step
                loss = output.loss.item()
                tokens = input_ids.numel()
                tokens_per_second = tokens / seconds

                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/perplexity", math.exp(min(loss, 20.0)), step)
                writer.add_scalar("optimizer/learning_rate", trainer.optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("optimizer/gradient_norm", trainer.last_grad_norm, step)
                writer.add_scalar("data/batch_size", input_ids.shape[0], step)
                writer.add_scalar("data/sequence_length", input_ids.shape[1], step)
                writer.add_scalar("performance/step_seconds", seconds, step)
                writer.add_scalar("performance/tokens_per_second", tokens_per_second, step)
                writer.add_scalar("performance/total_tokens", step * tokens, step)
                writer.add_scalar("model/logits_mean", output.logits.detach().mean().item(), step)
                writer.add_scalar("model/logits_std", output.logits.detach().std().item(), step)

                if device.type == "cuda":
                    writer.add_scalar(
                        "system/cuda_memory_allocated_mb",
                        torch.cuda.memory_allocated() / 2**20,
                        step,
                    )
                    writer.add_scalar(
                        "system/cuda_memory_reserved_mb",
                        torch.cuda.memory_reserved() / 2**20,
                        step,
                    )
                    writer.add_scalar(
                        "system/cuda_max_memory_allocated_mb",
                        torch.cuda.max_memory_allocated() / 2**20,
                        step,
                    )

                if step % args.log_interval == 0:
                    print(f"step={step} loss={loss:.4f} tokens_per_second={tokens_per_second:.1f}")
                    writer.flush()
                if trainer.global_step >= args.steps:
                    break

            if not made_progress:
                raise RuntimeError("the dataset produced no complete training sequences")

        writer.add_text("run/status", "completed", trainer.global_step)
        writer.flush()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "global_step": trainer.global_step,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint to {args.checkpoint}")
    finally:
        writer.close()


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
