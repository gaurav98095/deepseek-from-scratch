"""Sparse DeepSeek-style MoE with token dispatch and aggregation."""

from __future__ import annotations

import torch
from torch import nn

from .experts import ExpertBank
from .router import TopKRouter


class DeepSeekMoE(nn.Module):
    """Shared expert plus token-dependent top-k routed experts."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.top_k = config.n_activated_experts
        self.router = TopKRouter(config)
        self.experts = ExpertBank(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run sparse MoE for hidden states shaped ``[B, T, D]``."""
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected hidden dimension {self.dim}, got {x.shape[-1]}"
            )

        batch, sequence, _ = x.shape
        flat_x = x.reshape(batch * sequence, self.dim)
        indices, weights = self.router(flat_x)

        output = self.experts.forward_shared(flat_x)
        for expert_id in range(self.n_experts):
            token_ids, top_k_slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue

            expert_input = flat_x.index_select(0, token_ids)
            expert_output = self.experts.forward_expert(
                expert_id,
                expert_input,
            )
            expert_weights = weights[token_ids, top_k_slots].to(
                dtype=expert_output.dtype,
            )
            contribution = expert_output * expert_weights.unsqueeze(-1)
            output = output.index_add(0, token_ids, contribution)

        return output.reshape(batch, sequence, self.dim)
