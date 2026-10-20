"""Streaming TinyStories records into fixed-length language-model batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .byte_tokenizer import ByteTokenizer


class TinyStoriesDataset(IterableDataset[Tensor]):
    """Yield fixed-length token sequences from prepared TinyStories JSONL."""

    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, block_size: int) -> None:
        super().__init__()
        # At least two positions are needed to create shifted input/target pairs.
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self) -> Iterator[Tensor]:
        # Streaming the file avoids loading the complete prepared dataset into RAM.
        with self.path.open(encoding="utf-8") as data_file:
            for line in data_file:
                # Each line is one normalized story written by the preparation step.
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str):
                    continue
                # Tokenization happens lazily so training can start from large files.
                tokens = self.tokenizer.encode(text)
                # Move by block_size so each yielded sample is a non-overlapping chunk.
                for start in range(0, len(tokens) - self.block_size, self.block_size):
                    # The extra token becomes the final next-token target.
                    sequence = tokens[start : start + self.block_size + 1]
                    if len(sequence) == self.block_size + 1:
                        # Long tensors are required by nn.Embedding input IDs.
                        yield torch.tensor(sequence, dtype=torch.long)
