"""Token embedding layers."""

import torch
from torch import nn

from .config import DeepSeekConfig


class TokenEmbedding(nn.Module):
    """Map token IDs to learned hidden vectors."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)
