from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from src.model import (
    CachedMLA,
    DeepSeekConfig,
    DeepSeekMoE,
    FinalDeepSeekBlock,
    FinalTinyDeepSeek,
    SwiGLUFFN,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint


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


def test_final_model_uses_cached_mla_and_dense_moe_split() -> None:
    model = FinalTinyDeepSeek(small_config())

    assert all(isinstance(layer, FinalDeepSeekBlock) for layer in model.layers)
    assert all(isinstance(layer.attention, CachedMLA) for layer in model.layers)
    assert isinstance(model.layers[0].ffn, SwiGLUFFN)
    assert all(isinstance(layer.ffn, DeepSeekMoE) for layer in model.layers[1:])


def test_forward_can_return_hidden_states() -> None:
    config = small_config()
    model = FinalTinyDeepSeek(config)
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    logits, hidden = model(tokens, return_hidden=True)

    assert logits.shape == (2, 7, config.vocab_size)
    assert hidden.shape == (2, 7, config.dim)


def test_prefill_and_decode_match_full_forward() -> None:
    torch.manual_seed(22)
    config = small_config()
    model = FinalTinyDeepSeek(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 9))

    with torch.no_grad():
        full_logits = model(tokens)
        prompt_logits = model.prefill(tokens[:, :-1])
        decoded_logits = model.decode_step(tokens[:, -1:])

    assert prompt_logits.shape == (1, 8, config.vocab_size)
    torch.testing.assert_close(
        decoded_logits,
        full_logits[:, -1:],
        rtol=3e-4,
        atol=3e-4,
    )
    assert model.cache_length == 9


def test_reset_cache_clears_every_block() -> None:
    config = small_config()
    model = FinalTinyDeepSeek(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 4))

    model.prefill(tokens)
    assert all(layer.attention.cache_length == 4 for layer in model.layers)
    model.reset_cache()
    assert all(layer.attention.cache_length == 0 for layer in model.layers)


def test_checkpoint_restores_predictions_and_router_bias() -> None:
    torch.manual_seed(23)
    config = small_config()
    model = FinalTinyDeepSeek(config).eval()
    model.layers[1].ffn.router.expert_bias.fill_(0.25)
    tokens = torch.randint(0, config.vocab_size, (1, 5))
    expected = model(tokens)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "final.pt"
        save_checkpoint(path, model, step=17, note="test")
        restored, checkpoint = load_checkpoint(path)
        actual = restored(tokens)

    torch.testing.assert_close(actual, expected)
    assert checkpoint["step"] == 17
    assert checkpoint["note"] == "test"
    torch.testing.assert_close(
        restored.layers[1].ffn.router.expert_bias,
        model.layers[1].ffn.router.expert_bias,
    )
