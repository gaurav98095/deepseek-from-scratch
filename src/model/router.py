"""Minimal sigmoid top-k router for the educational DeepSeekMoE layer."""

from __future__ import annotations

import torch
from torch import nn


class TopKRouter(nn.Module):
    """Score routed experts independently and select the top-k per token."""

    def __init__(self, config) -> None:
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.top_k = config.n_activated_experts
        self.route_scale = config.route_scale
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("Invalid top-K setting")

        self.gate = nn.Linear(
            config.dim,
            self.n_experts,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected expert IDs and normalized routing weights.

        Args:
            x: Tensor with shape ``[..., hidden_dim]``.

        Returns:
            ``indices`` with shape ``[..., top_k]`` and dtype ``torch.long``;
            ``weights`` with the same shape and floating-point dtype.
        """
        if x.ndim < 1:
            raise ValueError("Expected x to have at least one dimension")
        if x.shape[-1] != self.gate.in_features:
            raise ValueError(
                f"Expected hidden dimension {self.gate.in_features}, "
                f"got {x.shape[-1]}"
            )

        logits = self.gate(x)
        scores = torch.sigmoid(logits.float())
        selected_scores, indices = torch.topk(
            scores,
            k=self.top_k,
            dim=-1,
        )
        denominator = selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = selected_scores / denominator
        return indices, weights * self.route_scale
