"""BF16 and gradient-accumulation training step for DeepSeek + MTP."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import torch

from .moe_balancer import MoELoadBalancer
from .scheduler import get_learning_rate, set_learning_rate


def _use_bf16(device: torch.device, requested: bool | None) -> bool:
    if requested is False:
        return False
    available = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if requested is True and not available:
        raise ValueError("BF16 was requested but the device does not support it")
    return available


def efficient_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    balancer: MoELoadBalancer,
    microbatches: Iterable[torch.Tensor],
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    max_grad_norm: float = 1.0,
    use_bf16: bool | None = None,
) -> dict[str, float | list[float]]:
    """Accumulate equally weighted microbatches into one optimizer update."""
    batches = list(microbatches)
    if not batches:
        raise ValueError("microbatches cannot be empty")
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
    device = batches[0].device
    autocast_enabled = _use_bf16(device, use_bf16)
    accumulated = {"loss": 0.0, "main_loss": 0.0, "mtp_loss": 0.0}

    try:
        for tokens in batches:
            autocast_context = (
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                )
                if autocast_enabled
                else contextlib.nullcontext()
            )
            with autocast_context:
                result = model(tokens)
                loss = result["loss"] / len(batches)
            loss.backward()
            for name in accumulated:
                accumulated[name] += float(result[name].detach()) / len(batches)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        load_ratios = balancer.update()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        balancer.clear()
        raise

    return {
        **accumulated,
        "lr": learning_rate,
        "grad_norm": float(grad_norm.detach()),
        "load_ratios": load_ratios,
        "mean_max_load_ratio": (
            sum(load_ratios) / len(load_ratios) if load_ratios else 0.0
        ),
    }
