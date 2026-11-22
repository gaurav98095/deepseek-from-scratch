"""Reproducible random batches from a memory-mapped token stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenBatcher:
    """Sample random ``[batch, sequence + 1]`` windows from a token file."""

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        dtype: str | np.dtype = np.uint16,
        seed: int = 0,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.tokens) <= seq_len:
            raise ValueError("not enough tokens for one sampled window")

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self, batch_size: int) -> torch.Tensor:
        """Return a random integer tensor with shape ``[B, seq_len + 1]``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        max_start = len(self.tokens) - self.seq_len
        starts = torch.randint(
            low=0,
            high=max_start,
            size=(batch_size,),
            generator=self.generator,
        )
        rows = []
        for start in starts.tolist():
            window = np.asarray(
                self.tokens[start : start + self.seq_len + 1],
                dtype=np.int64,
            ).copy()
            rows.append(torch.from_numpy(window))
        return torch.stack(rows, dim=0)

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """Return enough state to continue the same random sequence."""
        return {
            "generator_state": self.generator.get_state().clone(),
            "seq_len": self.seq_len,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a state previously returned by :meth:`state_dict`."""
        if int(state.get("seq_len", -1)) != self.seq_len:
            raise ValueError("batcher sequence length does not match checkpoint")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("checkpoint is missing generator state")
        self.generator.set_state(generator_state.cpu())
