"""Held-out next-token evaluation for the main DeepSeek model."""

from __future__ import annotations

import math
import contextlib

import torch
import torch.nn.functional as F

from ..data.token_batches import TokenBatcher
from ..model.final_deepseek import TinyDeepSeek


def evaluate_main_model(
    model: TinyDeepSeek,
    batcher: TokenBatcher,
    *,
    batch_size: int,
    num_batches: int,
) -> dict[str, float | int]:
    """Return mean validation loss and perplexity without updating the model."""
    if batch_size <= 0 or num_batches <= 0:
        raise ValueError("batch_size and num_batches must be positive")

    device = next(model.parameters()).device
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    was_training = model.training
    loss_sum = 0.0
    token_count = 0

    model.eval()
    try:
        with torch.no_grad():
            for _ in range(num_batches):
                tokens = batcher.next_batch(batch_size).to(device, non_blocking=True)
                inputs = tokens[:, :-1]
                targets = tokens[:, 1:]
                autocast_context = (
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                    )
                    if use_bf16
                    else contextlib.nullcontext()
                )
                with autocast_context:
                    logits = model(inputs)
                flat_logits = logits.float().reshape(-1, logits.shape[-1])
                flat_targets = targets.reshape(-1)
                loss_sum += float(
                    F.cross_entropy(
                        flat_logits,
                        flat_targets,
                        reduction="sum",
                    )
                )
                token_count += flat_targets.numel()
    finally:
        model.train(was_training)

    average_loss = loss_sum / token_count
    return {
        "val_loss": average_loss,
        "perplexity": math.exp(average_loss),
        "evaluated_tokens": token_count,
    }
