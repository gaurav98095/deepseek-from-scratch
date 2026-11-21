"""Independent SwiGLU expert networks used by DeepSeekMoE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """A single bias-free SwiGLU feed-forward expert."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected the last dimension to be {self.dim}, "
                f"got {x.shape[-1]}"
            )
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ExpertBank(nn.Module):
    """Independent routed experts and the always-active shared expert."""

    def __init__(self, config) -> None:
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.dim = config.dim

        self.routed_experts = nn.ModuleList(
            [
                Expert(
                    dim=config.dim,
                    hidden_dim=config.moe_inter_dim,
                )
                for _ in range(self.n_routed)
            ]
        )
        self.shared_expert = Expert(
            dim=config.dim,
            hidden_dim=config.n_shared_experts * config.moe_inter_dim,
        )

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared expert for every token."""
        return self.shared_expert(x)

    def forward_expert(self, expert_id: int, x: torch.Tensor) -> torch.Tensor:
        """Run one routed expert by integer ID."""
        if not 0 <= expert_id < self.n_routed:
            raise IndexError("Invalid expert ID")
        return self.routed_experts[expert_id](x)
