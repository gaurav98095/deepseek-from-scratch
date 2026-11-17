"""Attention-only language-model baseline."""

import torch
from torch import nn

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm
from .transformer import AttentionBlock


class DeepSeekSkeleton(nn.Module):
    """Embedding, causal-attention blocks, and output projection."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

        hidden_states = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
