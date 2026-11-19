"""Compare short training runs for the dense GPT and naive MLA models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.dataloader import create_dataloader
from src.model import DeepSeekConfig, DeepSeekMLA, DeepSeekSkeleton


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    """Return mean next-token loss over a limited validation stream."""
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                target_ids.reshape(-1),
            )
            total_loss += float(loss)
            batches += 1
            if batches >= max_batches:
                break
    if batches == 0:
        raise ValueError("validation loader produced no batches")
    return total_loss / batches


def train_steps(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> float:
    """Run a short FP32 training comparison and return the final loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    iterator = iter(loader)
    final_loss = 0.0
    for _ in range(steps):
        try:
            input_ids, target_ids = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            input_ids, target_ids = next(iterator)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            target_ids.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def run_comparison(
    data_dir: str | Path,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Train and evaluate both architectures with matched settings."""
    metadata = json.loads(
        (Path(data_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    config = DeepSeekConfig(
        vocab_size=int(metadata["tokenizer_vocab_size"]),
        dim=256,
        n_layers=6,
        n_heads=8,
        inter_dim=768,
        max_seq_len=seq_len,
    )
    train_loader = create_dataloader(
        str(data_dir), "train", batch_size, seq_len
    )
    val_loader = create_dataloader(
        str(data_dir), "val", batch_size, seq_len
    )

    results: dict[str, dict[str, float]] = {}
    for name, model in (
        ("gpt", DeepSeekSkeleton(config)),
        ("mla", DeepSeekMLA(config)),
    ):
        model = model.to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        start = time.perf_counter()
        train_loss = train_steps(
            model,
            train_loader,
            device=device,
            steps=steps,
            learning_rate=learning_rate,
        )
        elapsed = time.perf_counter() - start
        validation_loss = evaluate(
            model,
            val_loader,
            device=device,
            max_batches=max(1, min(10, len(val_loader))),
        )
        results[name] = {
            "parameters": float(parameter_count),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "seconds": elapsed,
        }
        print(name, results[name])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_comparison(
        args.data_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        device=device,
    )


if __name__ == "__main__":
    main()
