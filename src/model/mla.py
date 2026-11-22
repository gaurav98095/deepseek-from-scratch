"""Educational low-rank query and KV projections for MLA."""

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekConfig
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding


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


class MLA(nn.Module):
    """Naive DeepSeek-style MLA without optimized KV caching."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_content = config.qk_nope_head_dim
        self.d_rope = config.qk_rope_head_dim
        self.d_value = config.v_head_dim
        self.qk_dim = self.d_content + self.d_rope
        self.scale = self.qk_dim**-0.5

        self.q_branch = QueryCompression(config)
        self.kv_branch = KVCompression(config)
        self.rope = RotaryEmbedding(
            dim=self.d_rope,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.d_value,
            config.dim,
            bias=False,
        )
        future = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("future_mask", future, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        batch, time, _ = x.shape
        if time > self.future_mask.size(0):
            raise ValueError("sequence is longer than max_seq_len")

        q_content, q_position = self.q_branch(x)
        _, k_position, k_content, values = self.kv_branch(x)

        q_content = q_content.transpose(1, 2)
        q_position = q_position.transpose(1, 2)
        k_content = k_content.transpose(1, 2)
        values = values.transpose(1, 2)

        k_position = k_position.unsqueeze(1)
        k_position = k_position.expand(-1, self.n_heads, -1, -1)

        q_position = self.rope(q_position, start_pos=start_pos)
        k_position = self.rope(k_position, start_pos=start_pos)

        q = torch.cat([q_content, q_position], dim=-1)
        k = torch.cat([k_content, k_position], dim=-1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            self.future_mask[:time, :time],
            float("-inf"),
        )
        weights = F.softmax(scores.float(), dim=-1).to(values.dtype)
        attended = weights @ values
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(batch, time, self.n_heads * self.d_value)
        return self.out_proj(attended)
