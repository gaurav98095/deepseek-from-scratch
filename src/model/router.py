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
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.n_experts),
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
        selection_scores = scores + self.expert_bias.to(scores.device)
        _, indices = torch.topk(
            selection_scores,
            k=self.top_k,
            dim=-1,
        )
        selected_scores = scores.gather(dim=-1, index=indices)
        denominator = selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = selected_scores / denominator
        return indices, (weights * self.route_scale).to(dtype=x.dtype)

    @torch.no_grad()
    def update_bias(
        self,
        indices: torch.Tensor,
        gamma: float = 0.01,
    ) -> torch.Tensor:
        """Adjust expert-selection biases from one routing observation.

        The update is deliberately outside autograd. Overloaded experts move
        down, underloaded experts move up, and the learned gate weights are
        untouched.
        """
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if indices.numel() == 0:
            return torch.zeros_like(self.expert_bias)
        if indices.dtype != torch.long:
            raise ValueError("indices must have dtype torch.long")
        if int(indices.min()) < 0 or int(indices.max()) >= self.n_experts:
            raise ValueError("indices contain an invalid expert ID")

        counts = torch.bincount(
            indices.reshape(-1),
            minlength=self.n_experts,
        ).to(
            device=self.expert_bias.device,
            dtype=self.expert_bias.dtype,
        )
        target = counts.sum() / self.n_experts
        adjustment = torch.sign(target - counts)
        self.expert_bias.add_(gamma * adjustment)
        return counts
