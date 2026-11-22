"""Deterministic, non-overlapping validation windows."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from ..model.final_deepseek import TinyDeepSeek


class FixedTokenWindows:
    """Expose fixed non-overlapping ``[B, seq_len + 1]`` token windows."""

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        dtype: str | np.dtype = np.uint16,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.tokens) < seq_len + 1:
            raise ValueError("not enough tokens for one validation window")
        self.num_windows = (len(self.tokens) - seq_len - 1) // seq_len + 1

    def __len__(self) -> int:
        return self.num_windows

    def window(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.num_windows:
            raise IndexError(index)
        start = index * self.seq_len
        values = np.asarray(
            self.tokens[start : start + self.seq_len + 1],
            dtype=np.int64,
        ).copy()
        return torch.from_numpy(values)

    def batches(
        self,
        batch_size: int,
        *,
        max_windows: int | None = None,
    ) -> Iterator[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        count = self.num_windows if max_windows is None else min(
            max_windows,
            self.num_windows,
        )
        if count <= 0:
            raise ValueError("max_windows must select at least one window")
        for begin in range(0, count, batch_size):
            yield torch.stack(
                [self.window(index) for index in range(begin, min(begin + batch_size, count))],
                dim=0,
            )


def evaluate_fixed_windows(
    model: TinyDeepSeek,
    windows: FixedTokenWindows,
    *,
    batch_size: int,
    max_windows: int | None = None,
) -> dict[str, float | int]:
    """Score identical validation positions and return token-weighted metrics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    was_training = model.training
    total_nll = 0.0
    total_targets = 0
    evaluated_windows = 0

    model.eval()
    try:
        with torch.no_grad():
            for batch in windows.batches(batch_size, max_windows=max_windows):
                batch = batch.to(device, non_blocking=True)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                context = (
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                    )
                    if use_bf16
                    else contextlib.nullcontext()
                )
                with context:
                    logits = model(inputs)
                total_nll += float(
                    F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                )
                total_targets += targets.numel()
                evaluated_windows += batch.shape[0]
    finally:
        model.train(was_training)

    loss = total_nll / total_targets
    return {
        "loss": loss,
        "val_loss": loss,
        "perplexity": math.exp(loss),
        "target_tokens": total_targets,
        "windows": evaluated_windows,
    }
