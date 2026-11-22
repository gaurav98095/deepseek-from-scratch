"""Small GPU memory and throughput profiling helper."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch


def profile_train_step(
    train_fn: Callable[[], dict[str, float | list[float]]],
    *,
    tokens_per_step: int,
) -> dict[str, float | list[float]]:
    """Time one complete optimizer step and report CUDA peak memory."""
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metrics = train_fn()
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result: dict[str, float | list[float]] = {
        **metrics,
        "step_seconds": elapsed,
        "tokens_per_second": tokens_per_step / elapsed,
    }
    if use_cuda:
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    return result
