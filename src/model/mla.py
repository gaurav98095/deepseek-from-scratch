"""Educational low-rank query and KV projections for MLA."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekConfig
from .rmsnorm import RMSNorm


class QueryCompression(nn.Module):
    """Compress hidden states, then create content and position queries."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_content = config.qk_nope_head_dim
        self.d_rope = config.qk_rope_head_dim
        self.head_dim = self.d_content + self.d_rope

        self.q_down = nn.Linear(config.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.q_up = nn.Linear(
            config.q_lora_rank,
            config.n_heads * self.head_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        batch, time, _ = x.shape
        compressed = self.q_norm(self.q_down(x))
        expanded = self.q_up(compressed)
        expanded = expanded.reshape(batch, time, self.n_heads, self.head_dim)
        q_content, q_rope = torch.split(
            expanded,
            [self.d_content, self.d_rope],
            dim=-1,
        )
        return q_content, q_rope


class KVCompression(nn.Module):
    """Create a shared KV latent, content keys, values, and a position key."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.kv_rank = config.kv_lora_rank
        self.d_content = config.qk_nope_head_dim
        self.d_rope = config.qk_rope_head_dim
        self.d_value = config.v_head_dim

        self.kv_down = nn.Linear(
            config.dim,
            self.kv_rank + self.d_rope,
            bias=False,
        )
        self.kv_norm = RMSNorm(self.kv_rank)
        self.kv_up = nn.Linear(
            self.kv_rank,
            self.n_heads * (self.d_content + self.d_value),
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        batch, time, _ = x.shape
        compressed = self.kv_down(x)
        kv_latent, k_rope = torch.split(
            compressed,
            [self.kv_rank, self.d_rope],
            dim=-1,
        )
        kv_latent = self.kv_norm(kv_latent)

        expanded = self.kv_up(kv_latent)
        expanded = expanded.reshape(
            batch,
            time,
            self.n_heads,
            self.d_content + self.d_value,
        )
        k_content, values = torch.split(
            expanded,
            [self.d_content, self.d_value],
            dim=-1,
        )
        return kv_latent, k_rope, k_content, values


class ContentOnlyMLA(nn.Module):
    """Educational MLA path without positional attention yet."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.q_branch = QueryCompression(config)
        self.kv_branch = KVCompression(config)
        self.n_heads = config.n_heads
        self.d_content = config.qk_nope_head_dim
        self.d_value = config.v_head_dim
        self.out_proj = nn.Linear(
            config.n_heads * self.d_value,
            config.dim,
            bias=False,
        )
        future = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("future_mask", future, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        batch, time, _ = x.shape
        if time > self.future_mask.size(0):
            raise ValueError("sequence is longer than max_seq_len")

        q_content, _ = self.q_branch(x)
        _, _, k_content, values = self.kv_branch(x)

        scores = torch.einsum("bihd,bjhd->bhij", q_content, k_content)
        scores = scores / math.sqrt(self.d_content)
        scores = scores.masked_fill(
            self.future_mask[:time, :time],
            float("-inf"),
        )
        weights = F.softmax(scores.float(), dim=-1).to(values.dtype)
        attended = torch.einsum("bhij,bjhd->bihd", weights, values)
        attended = attended.reshape(batch, time, self.n_heads * self.d_value)
        return self.out_proj(attended)
