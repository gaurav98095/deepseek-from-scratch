"""Train the dense GPT-style baseline on prepared token files."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataloader import create_dataloader
from src.model import DeepSeekConfig, DeepSeekSkeleton


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_batches(loader: torch.utils.data.DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Cycle over a finite DataLoader indefinitely."""
    while True:
        yield from loader


def load_metadata(data_dir: str | Path) -> dict:
    """Load metadata written by the dataset preparation step."""
    path = Path(data_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_config(
    metadata: dict,
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    inter_dim: int,
    max_seq_len: int,
) -> DeepSeekConfig:
    """Build a model config using the prepared tokenizer vocabulary."""
    try:
        vocab_size = int(metadata["tokenizer_vocab_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "metadata.json must contain tokenizer_vocab_size"
        ) from error

    return DeepSeekConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        inter_dim=inter_dim,
        max_seq_len=max_seq_len,
    )


def train(
    model: DeepSeekSkeleton,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    steps: int,
    log_every: int = 25,
    use_bf16: bool = False,
) -> list[float]:
    """Run a fixed number of next-token training steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    model.train()
    batches = infinite_batches(loader)
    losses: list[float] = []
    autocast_enabled = use_bf16 and device.type == "cuda"
    autocast_context = lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    )

    for step in range(1, steps + 1):
        input_ids, target_ids = next(batches)
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context() if autocast_enabled else contextlib.nullcontext():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                target_ids.reshape(-1),
            )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step {step:>6}/{steps} | loss {loss_value:.4f}")

    return losses


def save_checkpoint(
    path: str | Path,
    *,
    model: DeepSeekSkeleton,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    tokenizer_path: str | Path,
) -> None:
    """Save model, optimizer, configuration, and tokenizer metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "loss": loss,
            "config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_path": str(tokenizer_path),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/gpt_baseline.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--inter-dim", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    metadata = load_metadata(data_dir)
    config = build_config(
        metadata,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        inter_dim=args.inter_dim,
        max_seq_len=args.seq_len,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = create_dataloader(
        data_dir=str(data_dir),
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )
    model = DeepSeekSkeleton(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    losses = train(
        model,
        loader,
        optimizer,
        device=device,
        steps=args.steps,
        log_every=args.log_every,
        use_bf16=args.bf16,
    )
    save_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        loss=losses[-1],
        tokenizer_path=data_dir / "tokenizer.json",
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
