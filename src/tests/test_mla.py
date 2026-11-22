import torch

from src.model import DeepSeekConfig
from src.model.mla import KVCompression, MLA, QueryCompression


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


def test_mla_shape_and_backward() -> None:
    config = small_config()
    model = MLA(config)
    output = model(torch.randn(2, 16, 64, requires_grad=True))
    output.square().mean().backward()

    assert output.shape == (2, 16, 64)
    assert model.q_branch.q_down.weight.grad is not None
    assert model.kv_branch.kv_down.weight.grad is not None
    assert model.out_proj.weight.grad is not None


def test_mla_is_causal() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = MLA(config)
    prefix = torch.randn(1, 4, 64)
    first = model(torch.cat([prefix, torch.zeros(1, 2, 64)], dim=1))

    changed = torch.cat([prefix, torch.randn(1, 2, 64)], dim=1)
    second = model(changed)

    torch.testing.assert_close(first[:, :4], second[:, :4])


def test_concatenated_scores_equal_separate_scores() -> None:
    torch.manual_seed(0)
    content_q = torch.randn(2, 4, 8, 32)
    content_k = torch.randn(2, 4, 8, 32)
    position_q = torch.randn(2, 4, 8, 16)
    position_k = torch.randn(2, 4, 8, 16)

    concatenated = torch.cat([content_q, position_q], dim=-1)
    concatenated = concatenated @ torch.cat(
        [content_k, position_k], dim=-1
    ).transpose(-2, -1)
    separate = (
        content_q @ content_k.transpose(-2, -1)
        + position_q @ position_k.transpose(-2, -1)
    )

    torch.testing.assert_close(concatenated, separate)
