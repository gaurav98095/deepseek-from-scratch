import pytest
import torch

from src.model import DeepSeekConfig, DeepSeekSkeleton


def test_forward_shape() -> None:
    config = DeepSeekConfig()
    model = DeepSeekSkeleton(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    logits = model(input_ids)

    assert logits.shape == (2, 16, config.vocab_size)


def test_parameter_count() -> None:
    config = DeepSeekConfig()
    model = DeepSeekSkeleton(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    attention_per_block = 4 * config.dim * config.dim + config.dim
    expected = (
        2 * config.vocab_size * config.dim
        + attention_per_block * config.n_layers
        + config.dim
    )

    assert actual == expected


def test_backward() -> None:
    config = DeepSeekConfig()
    model = DeepSeekSkeleton(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    model(input_ids).square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sequence_limit() -> None:
    config = DeepSeekConfig(max_seq_len=8)
    model = DeepSeekSkeleton(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 9))

    with pytest.raises(ValueError):
        model(input_ids)
