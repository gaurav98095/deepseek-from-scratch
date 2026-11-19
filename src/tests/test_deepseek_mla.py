import torch

from src.model import DeepSeekConfig, DeepSeekMLA, DeepSeekSkeleton
from src.model.transformer import DeepSeekBlock


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        inter_dim=128,
        max_seq_len=32,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_deepseek_block_shape() -> None:
    config = small_config()
    block = DeepSeekBlock(config)
    x = torch.randn(2, 8, config.dim)

    assert block(x).shape == x.shape


def test_full_model_shape_and_backward() -> None:
    config = small_config()
    model = DeepSeekMLA(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)
    logits.square().mean().backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert model.blocks[0].attention.q_branch.q_down.weight.grad is not None
    assert model.blocks[-1].attention.kv_branch.kv_down.weight.grad is not None


def test_full_model_is_causal() -> None:
    torch.manual_seed(42)
    config = small_config()
    model = DeepSeekMLA(config).eval()
    prefix = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    changed = torch.tensor([[10, 20, 30, 40, 50, 91, 92, 93]])

    with torch.no_grad():
        original_logits = model(prefix)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :5],
        changed_logits[:, :5],
    )


def test_mla_and_dense_baseline_have_expected_counts() -> None:
    config = small_config()
    dense = DeepSeekSkeleton(config)
    mla = DeepSeekMLA(config)

    assert count_parameters(mla) < count_parameters(dense)
