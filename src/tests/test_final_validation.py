from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model import (
    CachedMLA,
    DeepSeekMoE,
    DeepSeekWithMTP,
    FinalTinyDeepSeek,
    SwiGLUFFN,
    TopKRouter,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_model_structure_and_forward_shapes(tiny_config) -> None:
    model = FinalTinyDeepSeek(tiny_config)
    assert len(model.layers) == 3
    assert all(isinstance(layer.attention, CachedMLA) for layer in model.layers)
    assert isinstance(model.layers[0].ffn, SwiGLUFFN)
    assert all(isinstance(layer.ffn, DeepSeekMoE) for layer in model.layers[1:])

    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, hidden = model(ids, return_hidden=True)
    assert hidden.shape == (2, 8, tiny_config.dim)
    assert logits.shape == (2, 8, tiny_config.vocab_size)
    assert torch.isfinite(logits).all()


def test_full_model_causality(tiny_config) -> None:
    torch.manual_seed(42)
    model = FinalTinyDeepSeek(tiny_config).eval()
    first = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    second = torch.tensor([[10, 20, 30, 40, 50, 91, 92, 93]])

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    torch.testing.assert_close(
        first_logits[:, :5],
        second_logits[:, :5],
        atol=1e-5,
        rtol=1e-5,
    )


def test_sparse_moe_matches_reference_dispatch(tiny_config) -> None:
    torch.manual_seed(42)
    moe = DeepSeekMoE(tiny_config).eval()
    hidden = torch.randn(2, 5, tiny_config.dim)
    sparse = moe(hidden)

    flat = hidden.reshape(-1, tiny_config.dim)
    indices, weights = moe.router(flat)
    reference = moe.experts.forward_shared(flat)
    for expert_id, expert in enumerate(moe.experts.routed_experts):
        expert_output = expert(flat)
        selected = (indices == expert_id)
        contribution_weight = (weights * selected).sum(dim=-1, keepdim=True)
        reference = reference + expert_output * contribution_weight
    reference = reference.reshape_as(sparse)

    torch.testing.assert_close(sparse, reference, atol=1e-6, rtol=1e-5)


def test_main_loss_reaches_attention_router_and_experts(tiny_config) -> None:
    torch.manual_seed(42)
    model = FinalTinyDeepSeek(tiny_config)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 9))
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(
        logits.float().reshape(-1, tiny_config.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()

    assert model.embed_tokens.embedding.weight.grad is not None
    assert all(
        any(parameter.grad is not None for parameter in layer.attention.parameters())
        for layer in model.layers
    )
    moe_layers = [layer for layer in model.layers if isinstance(layer.ffn, DeepSeekMoE)]
    assert all(layer.ffn.router.gate.weight.grad is not None for layer in moe_layers)
    assert all(layer.ffn.experts.shared_expert.gate_proj.weight.grad is not None for layer in moe_layers)
    assert any(
        expert.gate_proj.weight.grad is not None
        for layer in moe_layers
        for expert in layer.ffn.experts.routed_experts
    )


def test_balancing_bias_updates_are_selection_only(tiny_config) -> None:
    router = TopKRouter(tiny_config)
    indices = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 2], [0, 3]],
        dtype=torch.long,
    )
    counts = router.update_bias(indices, gamma=0.05)
    torch.testing.assert_close(counts, torch.tensor([6.0, 4.0, 1.0, 1.0]))
    torch.testing.assert_close(
        router.expert_bias,
        torch.tensor([-0.05, -0.05, 0.05, 0.05]),
    )

    with torch.no_grad():
        router.gate.weight.zero_()
        router.expert_bias.copy_(torch.tensor([-0.3, -0.2, 0.2, 0.3]))
    inputs = torch.randn(2, 4, tiny_config.dim)
    selected, weights = router(inputs)
    assert set(selected[0, 0].tolist()) == {2, 3}
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))


def test_mtp_alignment_and_loss_shapes(tiny_config) -> None:
    tokens = torch.tensor([[10, 20, 30, 40, 50, 60]])
    hidden = torch.arange(5).view(1, 5, 1)
    assert hidden[:, :-1].flatten().tolist() == [0, 1, 2, 3]
    assert tokens[:, 1:-1].flatten().tolist() == [20, 30, 40, 50]
    assert tokens[:, 2:].flatten().tolist() == [30, 40, 50, 60]

    model = DeepSeekWithMTP(tiny_config, mtp_weight=0.3)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 9))
    result = model(tokens)
    assert result["main_logits"].shape == (2, 8, tiny_config.vocab_size)
    assert result["mtp_logits"].shape == (2, 7, tiny_config.vocab_size)
    torch.testing.assert_close(
        result["loss"],
        result["main_loss"] + 0.3 * result["mtp_loss"],
    )


def test_mtp_future_token_isolation_and_gradient_path(tiny_config) -> None:
    torch.manual_seed(42)
    model = DeepSeekWithMTP(tiny_config).eval()
    original = torch.tensor([[10, 20, 30, 40, 50, 60]])
    changed = original.clone()
    changed[:, 2] = 99
    with torch.no_grad():
        original_result = model(original)
        changed_result = model(changed)
    torch.testing.assert_close(
        original_result["mtp_logits"][:, 0],
        changed_result["mtp_logits"][:, 0],
        atol=1e-5,
        rtol=1e-5,
    )

    model.train()
    result = model(torch.randint(0, tiny_config.vocab_size, (2, 9)))
    model.zero_grad(set_to_none=True)
    result["mtp_loss"].backward()
    assert model.main_model.layers[0].attention.q_branch.q_down.weight.grad is not None
    assert model.main_model.embed_tokens.embedding.weight.grad is not None
    assert model.mtp_module.project.weight.grad is not None


def test_cached_decode_matches_full_forward(tiny_config) -> None:
    torch.manual_seed(42)
    model = FinalTinyDeepSeek(tiny_config).eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (1, 12))

    with torch.no_grad():
        full = model(tokens)
        model.prefill(tokens[:, :-1])
        cached = model.decode_step(tokens[:, -1:])

    torch.testing.assert_close(
        cached,
        full[:, -1:],
        atol=2e-4,
        rtol=2e-4,
    )


def test_multiple_cached_steps_match_full_forward(tiny_config) -> None:
    torch.manual_seed(43)
    model = FinalTinyDeepSeek(tiny_config).eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (1, 10))

    with torch.no_grad():
        reference = model(tokens)
        model.prefill(tokens[:, :5])
        for position in range(5, 10):
            cached = model.decode_step(tokens[:, position : position + 1])
            torch.testing.assert_close(
                cached,
                reference[:, position : position + 1],
                atol=2e-4,
                rtol=2e-4,
            )


def test_checkpoint_round_trip_preserves_predictions_and_bias(
    tiny_config,
    tmp_path,
) -> None:
    torch.manual_seed(42)
    model = FinalTinyDeepSeek(tiny_config).eval()
    with torch.no_grad():
        model.layers[1].ffn.router.expert_bias.copy_(
            torch.tensor([0.1, -0.1, 0.2, -0.2])
        )
    tokens = torch.randint(0, tiny_config.vocab_size, (1, 8))
    expected = model(tokens)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=100)
    restored, checkpoint = load_checkpoint(path)

    torch.testing.assert_close(restored(tokens), expected)
    assert checkpoint["step"] == 100
    torch.testing.assert_close(
        restored.layers[1].ffn.router.expert_bias,
        model.layers[1].ffn.router.expert_bias,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="Requires a BF16-capable CUDA device",
)
def test_bf16_training_step_is_finite(tiny_config) -> None:
    model = FinalTinyDeepSeek(tiny_config).cuda()
    tokens = torch.randint(
        0,
        tiny_config.vocab_size,
        (2, 9),
        device="cuda",
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits.float().reshape(-1, tiny_config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
    assert torch.isfinite(loss)
    loss.backward()
