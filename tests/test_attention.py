import pytest
import torch

from src.model.attention import CausalSelfAttention, LearnedPositionalEmbedding, MultiHeadAttention, SelfAttention


def test_multi_head_attention_preserves_shape() -> None:
    # A 12-wide model split across 3 heads should return the original width.
    attention = MultiHeadAttention(embed_dim=12, num_heads=3)

    output, weights = attention(torch.randn(2, 5, 12), return_attention=True)

    assert output.shape == (2, 5, 12)
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_self_attention_is_one_head() -> None:
    # SelfAttention is the single-head specialization of the general module.
    attention = SelfAttention(embed_dim=8)

    _, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert weights.shape == (2, 1, 4, 4)


def test_causal_attention_cannot_use_future_tokens() -> None:
    # Changing future tokens must not change outputs for earlier positions.
    torch.manual_seed(0)
    attention = CausalSelfAttention(embed_dim=8, num_heads=2).eval()
    original = torch.randn(1, 4, 8)
    changed_future = original.clone()
    changed_future[:, 2:] += 1000

    first_two_original = attention(original)[:, :2]
    first_two_changed = attention(changed_future)[:, :2]

    assert torch.allclose(first_two_original, first_two_changed, atol=1e-5)


def test_learned_positional_embedding_shape_and_limit() -> None:
    # Position vectors broadcast across the batch and enforce the configured limit.
    positions = LearnedPositionalEmbedding(max_sequence_length=16, embed_dim=8)

    assert positions(5).shape == (1, 5, 8)
    with pytest.raises(ValueError, match="exceeds"):
        positions(17)


def test_attention_rejects_invalid_head_configuration() -> None:
    # The hidden width must divide evenly into the requested number of heads.
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embed_dim=10, num_heads=3)
