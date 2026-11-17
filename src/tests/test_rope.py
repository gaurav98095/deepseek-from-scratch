import pytest
import torch

from src.model.attention import CausalSelfAttention
from src.model.rope import RotaryEmbedding


def test_rope_preserves_shape() -> None:
    rope = RotaryEmbedding(dim=32, max_seq_len=16)
    x = torch.randn(2, 4, 8, 32)

    assert rope(x).shape == x.shape


def test_rope_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(dim=31, max_seq_len=16)


def test_rotation_preserves_vector_length() -> None:
    rope = RotaryEmbedding(dim=32, max_seq_len=16)
    x = torch.randn(2, 4, 8, 32)

    before = x.square().sum(dim=-1)
    after = rope(x).square().sum(dim=-1)

    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_relative_position_is_preserved() -> None:
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    first = (rope(q, start_pos=8) * rope(k, start_pos=13)).sum()
    second = (rope(q, start_pos=21) * rope(k, start_pos=26)).sum()

    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_rope_backward_has_finite_gradients() -> None:
    rope = RotaryEmbedding(dim=32, max_seq_len=16)
    x = torch.randn(2, 4, 8, 32, requires_grad=True)

    rope(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_causality_with_rope() -> None:
    torch.manual_seed(42)
    attention = CausalSelfAttention(dim=64, n_heads=4, max_seq_len=32)
    x = torch.randn(1, 8, 64)
    original = attention(x)

    changed_x = x.clone()
    changed_x[:, 5:, :] = 100 * torch.randn_like(changed_x[:, 5:, :])
    changed = attention(changed_x)

    torch.testing.assert_close(original[:, :5], changed[:, :5])
