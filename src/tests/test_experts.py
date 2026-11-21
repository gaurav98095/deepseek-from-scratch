import pytest
import torch

from src.model import DeepSeekConfig, Expert, ExpertBank


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        dim=64,
        n_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        max_seq_len=16,
    )


def test_single_expert_shape_and_parameter_count() -> None:
    expert = Expert(dim=64, hidden_dim=32)
    x = torch.randn(2, 8, 64)

    output = expert(x)

    assert output.shape == x.shape
    assert sum(parameter.numel() for parameter in expert.parameters()) == 3 * 64 * 32


def test_experts_are_independent() -> None:
    torch.manual_seed(42)
    bank = ExpertBank(small_config())
    expert_0 = bank.routed_experts[0]
    expert_1 = bank.routed_experts[1]

    assert expert_0.gate_proj.weight is not expert_1.gate_proj.weight
    assert not torch.equal(
        expert_0.gate_proj.weight,
        expert_1.gate_proj.weight,
    )


def test_shared_and_routed_paths_have_matching_shapes() -> None:
    bank = ExpertBank(small_config())
    x = torch.randn(2, 5, 64)

    shared = bank.forward_shared(x)
    expert_0 = bank.forward_expert(0, x)
    expert_3 = bank.forward_expert(3, x)

    assert shared.shape == x.shape
    assert expert_0.shape == x.shape
    assert expert_3.shape == x.shape
    assert not torch.allclose(expert_0, expert_3)


def test_manually_weighted_experts_preserve_shape() -> None:
    bank = ExpertBank(small_config())
    x = torch.randn(2, 5, 64)

    output = (
        bank.forward_shared(x)
        + 0.6 * bank.forward_expert(1, x)
        + 0.4 * bank.forward_expert(2, x)
    )

    assert output.shape == x.shape


def test_only_selected_routed_experts_receive_gradients() -> None:
    torch.manual_seed(42)
    bank = ExpertBank(small_config())
    x = torch.randn(2, 5, 64, requires_grad=True)

    output = (
        bank.forward_shared(x)
        + 0.6 * bank.forward_expert(1, x)
        + 0.4 * bank.forward_expert(2, x)
    )
    output.square().mean().backward()

    assert bank.shared_expert.gate_proj.weight.grad is not None
    for expert_id, expert in enumerate(bank.routed_experts):
        has_gradient = expert.gate_proj.weight.grad is not None
        assert has_gradient is (expert_id in {1, 2})


def test_expert_id_validation() -> None:
    bank = ExpertBank(small_config())
    x = torch.randn(1, 64)

    with pytest.raises(IndexError, match="Invalid expert ID"):
        bank.forward_expert(-1, x)
    with pytest.raises(IndexError, match="Invalid expert ID"):
        bank.forward_expert(4, x)


def test_default_bank_parameter_counts() -> None:
    config = DeepSeekConfig()
    bank = ExpertBank(config)
    expert_params = 3 * config.dim * config.moe_inter_dim
    total = sum(parameter.numel() for parameter in bank.parameters())
    active = (
        sum(parameter.numel() for parameter in bank.shared_expert.parameters())
        + 2 * expert_params
    )

    assert total == (config.n_routed_experts + config.n_shared_experts) * expert_params
    assert active == (config.n_activated_experts + config.n_shared_experts) * expert_params
