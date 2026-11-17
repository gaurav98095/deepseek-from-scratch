import pytest
import torch

from src.model.attention import CausalSelfAttention
from src.model.config import DeepSeekConfig
from src.model.deepseek import DeepSeekSkeleton


def test_attention_output_shape() -> None:
    attention = CausalSelfAttention(dim=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 8, 32)

    assert attention(x).shape == x.shape


def test_attention_rejects_invalid_head_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(dim=30, n_heads=4, max_seq_len=16)


def test_future_tokens_do_not_change_earlier_outputs() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(dim=32, n_heads=4, max_seq_len=16)
    prefix = torch.randn(1, 4, 32)
    first = attention(torch.cat([prefix, torch.zeros(1, 2, 32)], dim=1))

    changed = torch.cat([prefix, torch.randn(1, 2, 32)], dim=1)
    second = attention(changed)

    torch.testing.assert_close(first[:, :4], second[:, :4])


def test_attention_backward_has_finite_gradients() -> None:
    attention = CausalSelfAttention(dim=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 8, 32, requires_grad=True)

    attention(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_projection_parameter_count() -> None:
    dim = 32
    attention = CausalSelfAttention(dim=dim, n_heads=4, max_seq_len=16)

    assert sum(parameter.numel() for parameter in attention.parameters()) == 4 * dim * dim


def test_attention_model_forward_shape() -> None:
    config = DeepSeekConfig(vocab_size=128, dim=32, n_heads=4, n_layers=2, max_seq_len=16)
    model = DeepSeekSkeleton(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    assert model(input_ids).shape == (2, 8, config.vocab_size)
