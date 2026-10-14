import torch

from deepseek_scratch import CharacterTokenizer, DeepSeekForCausalLM, ModelConfig


def test_character_tokenizer_round_trip_and_unknown() -> None:
    tokenizer = CharacterTokenizer.from_text("ab ba")
    assert tokenizer.decode(tokenizer.encode("ab")) == "ab"
    assert tokenizer.encode("?") == [tokenizer.stoi[tokenizer.unk_token]]


def test_model_shapes_loss_and_causal_mask() -> None:
    config = ModelConfig(vocab_size=11, context_length=8, embedding_dim=12, num_layers=2, num_heads=3)
    model = DeepSeekForCausalLM(config)
    input_ids = torch.randint(config.vocab_size, (2, 5))
    logits, loss = model(input_ids, input_ids)
    assert logits.shape == (2, 5, config.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()

    model.eval()
    first = model(input_ids)[0]
    changed = input_ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % config.vocab_size
    second = model(changed)[0]
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)


def test_generation_preserves_prefix() -> None:
    config = ModelConfig(vocab_size=7, context_length=4, embedding_dim=8, num_layers=1, num_heads=2)
    model = DeepSeekForCausalLM(config)
    prompt = torch.tensor([[1, 2, 3]])
    result = model.generate(prompt, max_new_tokens=2)
    assert result.shape == (1, 5)
    assert torch.equal(result[:, :3], prompt)
