"""Train the complete small MLA + MoE language model."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from src.data.dataloader import create_dataloader
from src.model import DeepSeekConfig, TinyDeepSeek
from src.training import MoELoadBalancer
from src.train import infinite_batches, load_metadata, set_seed


def build_config(metadata: dict, args: argparse.Namespace) -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=int(metadata["tokenizer_vocab_size"]),
        dim=args.dim,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
        n_heads=args.n_heads,
        inter_dim=args.inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        max_seq_len=args.seq_len,
    )


def train(
    model: TinyDeepSeek,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    steps: int,
    gamma: float,
    log_every: int = 25,
    use_bf16: bool = False,
) -> list[float]:
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

    with MoELoadBalancer(model, gamma=gamma) as balancer:
        for step in range(1, steps + 1):
            input_ids, target_ids = next(batches)
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            balancer.clear()
            with (
                autocast_context()
                if autocast_enabled
                else contextlib.nullcontext()
            ):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, model.config.vocab_size),
                    target_ids.reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            load_ratios = balancer.update()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if step == 1 or step % log_every == 0 or step == steps:
                mean_ratio = (
                    sum(load_ratios) / len(load_ratios)
                    if load_ratios
                    else 0.0
                )
                print(
                    f"step {step:>6}/{steps} | loss {loss_value:.4f} | "
                    f"mean max-load ratio {mean_ratio:.3f}"
                )
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_deepseek_moe.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-dense-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--inter-dim", type=int, default=768)
    parser.add_argument("--moe-inter-dim", type=int, default=256)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    parser.add_argument("--n-shared-experts", type=int, default=1)
    parser.add_argument("--n-activated-experts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    metadata = load_metadata(data_dir)
    config = build_config(metadata, args)
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
    model = TinyDeepSeek(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    losses = train(
        model,
        loader,
        optimizer,
        device=device,
        steps=args.steps,
        gamma=args.gamma,
        log_every=args.log_every,
        use_bf16=args.bf16,
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": args.steps,
            "loss": losses[-1],
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": json.loads(
                (data_dir / "metadata.json").read_text(encoding="utf-8")
            ),
        },
        checkpoint,
    )
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
