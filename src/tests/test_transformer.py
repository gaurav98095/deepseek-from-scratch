import torch

from src.model import DeepSeekConfig, DeepSeekSkeleton
from src.model.ffn import SwiGLUFFN
from src.model.transformer import TransformerBlock


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        inter_dim=128,
        max_seq_len=32,
    )


def test_swiglu_shape() -> None:
    ffn = SwiGLUFFN(64, 128)
    x = torch.randn(2, 8, 64)

    assert ffn(x).shape == x.shape


def test_block_shape() -> None:
    block = TransformerBlock(small_config())
    x = torch.randn(2, 8, 64)

    assert block(x).shape == x.shape


def test_model_shape_and_backward() -> None:
    config = small_config()
    model = DeepSeekSkeleton(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)
    logits.square().mean().backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert all(parameter.grad is not None for parameter in model.parameters())
