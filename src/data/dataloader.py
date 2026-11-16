"""PyTorch DataLoader construction for serialized token windows."""

import torch
from torch.utils.data import DataLoader

from .dataset import TokenDataset


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    seq_len: int = 1024,
    stride: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a batched loader for a prepared train or validation split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if stride is None:
        stride = seq_len

    dataset = TokenDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        stride=stride,
    )
    use_cuda = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=(split == "train"),
    )
