import pytest
import torch

from src.model import DeepSeekConfig, TopKRouter


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        dim=64,
        n_heads=4,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=32,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=128,
        max_seq_len=16,
    )


def test_bias_is_a_non_trainable_checkpointed_buffer() -> None:
    router = TopKRouter(small_config())

    assert "expert_bias" not in dict(router.named_parameters())
    assert "expert_bias" in dict(router.named_buffers())
    assert not router.expert_bias.requires_grad
    assert "expert_bias" in router.state_dict()


def test_bias_update_moves_overloaded_down_and_underloaded_up() -> None:
    router = TopKRouter(small_config())
    indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 2],
            [0, 3],
        ],
        dtype=torch.long,
    )

    counts = router.update_bias(indices, gamma=0.05)

    torch.testing.assert_close(counts, torch.tensor([6.0, 4.0, 1.0, 1.0]))
    torch.testing.assert_close(
        router.expert_bias,
        torch.tensor([-0.05, -0.05, 0.05, 0.05]),
    )


def test_bias_changes_selection_but_not_output_weights() -> None:
    config = small_config()
    router = TopKRouter(config)
    original_scores = torch.tensor([0.9, 0.8, 0.6, 0.5])
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[:, 0] = torch.logit(original_scores)
        router.expert_bias.copy_(torch.tensor([-0.30, -0.25, 0.15, 0.20]))

    x = torch.zeros(1, 1, config.dim)
    x[..., 0] = 1.0
    indices, weights = router(x)

    assert indices.tolist() == [[[2, 3]]]
    expected_weights = torch.tensor([0.6 / 1.1, 0.5 / 1.1])
    torch.testing.assert_close(weights[0, 0], expected_weights, atol=1e-6, rtol=1e-6)


def test_router_gradients_remain_separate_from_bias_updates() -> None:
    router = TopKRouter(small_config())
    x = torch.randn(2, 4, 64, requires_grad=True)
    _, weights = router(x)
    weights[..., 0].sum().backward()

    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()
    assert router.expert_bias.grad is None


def test_bias_update_can_be_loaded_from_checkpoint() -> None:
    config = small_config()
    router = TopKRouter(config)
    router.update_bias(torch.tensor([[0, 1], [0, 1]]), gamma=0.1)

    restored = TopKRouter(config)
    restored.load_state_dict(router.state_dict())

    torch.testing.assert_close(restored.expert_bias, router.expert_bias)


def test_bias_update_validates_inputs() -> None:
    router = TopKRouter(small_config())

    with pytest.raises(ValueError, match="gamma"):
        router.update_bias(torch.tensor([[0, 1]]), gamma=-0.1)
    with pytest.raises(ValueError, match="dtype"):
        router.update_bias(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="invalid expert"):
        router.update_bias(torch.tensor([[0, 4]]))
