"""Learning-rate warmup and cosine-decay helpers."""

from __future__ import annotations

import math

import torch


def get_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return the learning rate for a one-based optimizer step."""
    if step <= 0:
        raise ValueError("step must be positive")
    if total_steps <= 0 or warmup_steps < 0:
        raise ValueError("total_steps must be positive and warmup nonnegative")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("learning-rate bounds are invalid")

    if step <= warmup_steps:
        return max_lr * step / warmup_steps if warmup_steps else max_lr

    progress = (min(step, total_steps) - warmup_steps) / (
        total_steps - warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set the same learning rate on every optimizer parameter group."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
