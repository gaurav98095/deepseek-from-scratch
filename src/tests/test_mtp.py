import torch
import pytest

from src.model import DeepSeekConfig, DeepSeekWithMTP, MTPModule, FinalTinyDeepSeek


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=128,
        dim=64,
        n_layers=3,
        n_dense_layers=1,
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
        max_seq_len=32,
    )


def test_mtp_module_returns_two_ahead_logits() -> None:
    config = small_config()
    main_model = FinalTinyDeepSeek(config)
    mtp = MTPModule(config)
    hidden = torch.randn(2, 5, config.dim)
    next_tokens = torch.randint(0, config.vocab_size, (2, 5))

    logits = mtp(
        hidden,
        next_tokens,
        shared_embedding=main_model.embed_tokens,
        shared_lm_head=main_model.lm_head,
    )

    assert logits.shape == (2, 5, config.vocab_size)


def test_wrapper_aligns_main_and_mtp_sequences() -> None:
    config = small_config()
    model = DeepSeekWithMTP(config)
    tokens = torch.randint(0, config.vocab_size, (2, 9))

    result = model(tokens)

    assert result["main_logits"].shape == (2, 8, config.vocab_size)
    assert result["main_hidden"].shape == (2, 8, config.dim)
    assert result["mtp_logits"].shape == (2, 7, config.vocab_size)
    torch.testing.assert_close(
        result["loss"],
        result["main_loss"] + 0.3 * result["mtp_loss"],
    )


def test_mtp_loss_backpropagates_to_main_model_and_shared_layers() -> None:
    torch.manual_seed(23)
    config = small_config()
    model = DeepSeekWithMTP(config)
    tokens = torch.randint(0, config.vocab_size, (2, 9))

    result = model(tokens)
    model.zero_grad(set_to_none=True)
    result["mtp_loss"].backward()

    assert model.main_model.layers[0].attention.q_branch.q_down.weight.grad is not None
    assert model.main_model.embed_tokens.embedding.weight.grad is not None
    assert model.main_model.lm_head.weight.grad is not None
    assert model.mtp_module.project.weight.grad is not None
    assert model.mtp_module.block.attn_norm.weight.grad is not None


def test_mtp_position_cannot_see_its_own_target() -> None:
    torch.manual_seed(24)
    config = small_config()
    model = DeepSeekWithMTP(config).eval()
    original = torch.tensor([[10, 20, 30, 40, 50, 60]])
    changed = original.clone()
    changed[:, 2] = 99

    with torch.no_grad():
        original_result = model(original)
        changed_result = model(changed)

    # Position zero receives h_0 and Emb(t_1), and predicts t_2. Changing t_2
    # must not change that prediction.
    torch.testing.assert_close(
        original_result["mtp_logits"][:, 0],
        changed_result["mtp_logits"][:, 0],
        rtol=2e-4,
        atol=2e-4,
    )
    assert not torch.allclose(
        original_result["mtp_logits"][:, -1],
        changed_result["mtp_logits"][:, -1],
    )


def test_mtp_rejects_short_sequences_and_negative_weight() -> None:
    config = small_config()
    with pytest.raises(ValueError, match="nonnegative"):
        DeepSeekWithMTP(config, mtp_weight=-0.1)

    model = DeepSeekWithMTP(config)
    with pytest.raises(ValueError, match="at least 3"):
        model(torch.randint(0, config.vocab_size, (1, 2)))
