"""Minimal language-model skeleton."""

import torch
from torch import nn

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm


class DeepSeekSkeleton(nn.Module):
    """Embedding and output projection with stable tensor interfaces."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
