"""One FP32 optimization step for the DeepSeek+MTP training wrapper."""

from __future__ import annotations

import torch

from .moe_balancer import MoELoadBalancer
from .scheduler import get_learning_rate, set_learning_rate


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    balancer: MoELoadBalancer,
    tokens: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    max_grad_norm: float = 1.0,
) -> dict[str, float | list[float]]:
    """Run forward, backward, clipping, AdamW, and router-bias updates."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    model.train()
    learning_rate = get_learning_rate(
        step,
        total_steps,
        warmup_steps,
        max_lr,
        min_lr,
    )
    set_learning_rate(optimizer, learning_rate)
    optimizer.zero_grad(set_to_none=True)
    balancer.clear()

    result = model(tokens)
    loss = result["loss"]
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )
    optimizer.step()
    load_ratios = balancer.update()

    return {
        "loss": float(loss.detach()),
        "main_loss": float(result["main_loss"].detach()),
        "mtp_loss": float(result["mtp_loss"].detach()),
        "lr": learning_rate,
        "grad_norm": float(grad_norm.detach()),
        "load_ratios": load_ratios,
    }
