"""Memory-mapped token dataset for next-token prediction."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose fixed-length input-target windows from a binary token file."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = 1024,
        stride: int = 1024,
    ) -> None:
        data_dir = Path(data_dir)
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if seq_len <= 0 or stride <= 0:
            raise ValueError("seq_len and stride must be positive")

        metadata = json.loads(
            (data_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.seq_len = seq_len
        self.stride = stride
        self.tokens = np.memmap(
            data_dir / f"{split}.bin",
            dtype=np.dtype(metadata["dtype"]),
            mode="r",
        )

        self.num_tokens = len(self.tokens)
        if self.num_tokens < seq_len + 1:
            raise ValueError("Not enough tokens for one example")

        self.num_examples = (
            (self.num_tokens - seq_len - 1) // stride
        ) + 1

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.num_examples:
            raise IndexError(index)

        start = index * self.stride
        end = start + self.seq_len + 1
        window = np.array(self.tokens[start:end], dtype=np.int64, copy=True)
        window_tensor = torch.from_numpy(window)

        return window_tensor[:-1], window_tensor[1:]
