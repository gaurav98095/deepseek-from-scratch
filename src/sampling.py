"""Sampling strategies for autoregressive language-model generation."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Select one token per batch row from ``[batch, vocabulary]`` logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if logits.shape[-1] == 0:
        raise ValueError("logits must contain at least one vocabulary entry")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise FloatingPointError("every batch row needs a finite logit")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scores = logits / temperature
    if top_k is not None:
        limit = min(top_k, scores.shape[-1])
        values, _ = torch.topk(scores, limit, dim=-1)
        scores = scores.masked_fill(scores < values[:, [-1]], -torch.inf)

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = (cumulative - sorted_probs) >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf)
        scores = scores.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_scores,
        )

    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("invalid sampling probabilities")
    return torch.multinomial(probabilities, num_samples=1)
