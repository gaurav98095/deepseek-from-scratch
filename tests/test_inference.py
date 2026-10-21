import pytest
import torch

from inference import generate
from src.data.byte_tokenizer import ByteTokenizer
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek


def test_generation_prefills_once_then_processes_one_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A forward hook records how many token positions reach the first Q projection.
    config = ModelConfig(
        vocab_size=258,
        max_position_embeddings=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        torch_dtype="float32",
    )
    model = DeepSeek(config).eval()

    # Equal zero logits make greedy decoding choose byte token 0, never EOS.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    projection_lengths: list[int] = []
    hook = model.layers[0].attention.q_proj.register_forward_pre_hook(
        lambda _module, inputs: projection_lengths.append(inputs[0].shape[1])
    )
    try:
        generated = generate(
            model,
            ByteTokenizer(),
            prompt="a",
            max_new_tokens=3,
            temperature=0,
            top_k=0,
        )
    finally:
        hook.remove()

    # BOS + "a" + EOS are prefetched together; later calls contain one token.
    assert projection_lengths == [3, 1, 1]
    assert generated.startswith("a")

    # The printed values come from real float32 K/V tensors in the one-layer model.
    cache_lines = capsys.readouterr().out.strip().splitlines()
    assert len(cache_lines) == 3
    assert "generation_step=1" in cache_lines[0]
    assert "kv_cache_tokens_per_layer=3" in cache_lines[0]
    assert "kv_cache_elements=48" in cache_lines[0]
    assert "kv_cache_bytes=192" in cache_lines[0]
    assert "generation_step=3" in cache_lines[2]
    assert "kv_cache_tokens_per_layer=5" in cache_lines[2]
    assert "kv_cache_elements=80" in cache_lines[2]
    assert "kv_cache_bytes=320" in cache_lines[2]
