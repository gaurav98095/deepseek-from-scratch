import torch

from src.model import DeepSeekConfig
from src.model.mla import ContentOnlyMLA, KVCompression, QueryCompression


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        inter_dim=128,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_seq_len=32,
    )


def test_query_shapes() -> None:
    config = small_config()
    q_content, q_rope = QueryCompression(config)(torch.randn(2, 16, 64))

    assert q_content.shape == (2, 16, 4, 8)
    assert q_rope.shape == (2, 16, 4, 4)


def test_kv_shapes() -> None:
    config = small_config()
    latent, k_rope, k_content, values = KVCompression(config)(
        torch.randn(2, 16, 64)
    )

    assert latent.shape == (2, 16, 8)
    assert k_rope.shape == (2, 16, 4)
    assert k_content.shape == (2, 16, 4, 8)
    assert values.shape == (2, 16, 4, 8)


def test_content_only_mla_shape_and_backward() -> None:
    config = small_config()
    model = ContentOnlyMLA(config)
    output = model(torch.randn(2, 16, 64, requires_grad=True))
    output.square().mean().backward()

    assert output.shape == (2, 16, 64)
    assert model.q_branch.q_down.weight.grad is not None
    assert model.kv_branch.kv_down.weight.grad is not None


def test_content_only_mla_is_causal() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = ContentOnlyMLA(config)
    prefix = torch.randn(1, 4, 64)
    first = model(torch.cat([prefix, torch.zeros(1, 2, 64)], dim=1))

    changed = torch.cat([prefix, torch.randn(1, 2, 64)], dim=1)
    second = model(changed)

    torch.testing.assert_close(first[:, :4], second[:, :4])
