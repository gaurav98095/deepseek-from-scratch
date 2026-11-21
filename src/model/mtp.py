"""One-stage sequential Multi-Token Prediction for the final model."""

from __future__ import annotations

import torch
from torch import nn

from .config import DeepSeekConfig
from .final_deepseek import FinalDeepSeekBlock
from .rmsnorm import RMSNorm


class MTPModule(nn.Module):
    """Predict the token after a teacher-forced next-token embedding.

    At each position ``i`` the module receives the main hidden state ``h_i``
    and the embedding of the actual token ``t_(i+1)``. It predicts ``t_(i+2)``.
    The embedding table and vocabulary head are supplied by the main model so
    their parameters are shared rather than copied.
    """

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        dim = config.dim
        self.hidden_norm = RMSNorm(dim, eps=1e-6)
        self.embed_norm = RMSNorm(dim, eps=1e-6)
        self.project = nn.Linear(2 * dim, dim, bias=False)

        # Use an MoE block when the configuration has an MoE suffix. If a
        # caller configures every main layer as dense, use the final valid
        # layer ID and retain a valid dense MTP block.
        mtp_layer_id = (
            config.n_dense_layers
            if config.n_dense_layers < config.n_layers
            else config.n_layers - 1
        )
        self.block = FinalDeepSeekBlock(config, layer_id=mtp_layer_id)
        self.norm = RMSNorm(dim, eps=1e-6)

    def forward(
        self,
        main_hidden: torch.Tensor,
        next_token_ids: torch.Tensor,
        *,
        shared_embedding: nn.Module,
        shared_lm_head: nn.Module,
    ) -> torch.Tensor:
        """Return vocabulary logits for the two-ahead MTP targets."""
        if main_hidden.ndim != 3:
            raise ValueError("main_hidden must have shape [batch, sequence, dim]")
        if next_token_ids.ndim != 2:
            raise ValueError(
                "next_token_ids must have shape [batch, sequence]"
            )
        if main_hidden.shape[:2] != next_token_ids.shape:
            raise ValueError("hidden states and token IDs must have matching shapes")
        if main_hidden.shape[-1] != self.project.out_features:
            raise ValueError("main_hidden has the wrong hidden dimension")
        if main_hidden.shape[1] == 0:
            raise ValueError("MTP requires at least one aligned position")

        next_embeddings = shared_embedding(next_token_ids)
        fused = torch.cat(
            [self.hidden_norm(main_hidden), self.embed_norm(next_embeddings)],
            dim=-1,
        )
        hidden = self.project(fused)
        hidden = self.block(hidden)
        return shared_lm_head(self.norm(hidden))
