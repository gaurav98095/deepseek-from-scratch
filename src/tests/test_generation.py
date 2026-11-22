import torch

from src.generate import generate
from src.model import DeepSeekConfig, TinyDeepSeek
from src.sampling import sample_next_token


def test_greedy_selects_highest_logit() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    selected = sample_next_token(logits, greedy=True)
    assert selected.item() == 1


def test_top_k_one_keeps_only_the_best_token() -> None:
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    selected = sample_next_token(logits, top_k=1)
    assert selected.item() == 1


def test_top_p_keeps_the_first_probability_mass() -> None:
    torch.manual_seed(42)
    logits = torch.tensor([[4.0, 3.0, 0.0, -4.0]])
    selected = sample_next_token(logits, top_p=0.8)
    assert selected.item() in {0, 1}


def test_generate_preserves_prompt_and_adds_tokens() -> None:
    config = DeepSeekConfig(
        vocab_size=32,
        dim=16,
        n_layers=1,
        n_heads=2,
        inter_dim=32,
        max_seq_len=8,
    )
    model = TinyDeepSeek(config)
    prompt = torch.tensor([[1, 2, 3]])

    output = generate(
        model,
        prompt,
        max_new_tokens=2,
        temperature=1.0,
        top_p=0.9,
    )

    assert output.shape == (1, 5)
    torch.testing.assert_close(output[:, :3], prompt)
    assert model.cache_length == 0


def test_generate_rejects_context_overflow() -> None:
    config = DeepSeekConfig(
        vocab_size=32,
        dim=16,
        n_layers=1,
        n_heads=2,
        inter_dim=32,
        max_seq_len=4,
    )
    model = TinyDeepSeek(config)
    prompt = torch.tensor([[1, 2, 3]])

    try:
        generate(model, prompt, max_new_tokens=2)
    except ValueError as error:
        assert "maximum context length" in str(error)
    else:
        raise AssertionError("expected context overflow to be rejected")
