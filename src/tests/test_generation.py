import torch

from src.generate import generate
from src.model import DeepSeekConfig, DeepSeekSkeleton


def test_generate_preserves_prompt_and_adds_tokens() -> None:
    config = DeepSeekConfig(
        vocab_size=32,
        dim=16,
        n_layers=1,
        n_heads=2,
        inter_dim=32,
        max_seq_len=8,
    )
    model = DeepSeekSkeleton(config)
    prompt = torch.tensor([[1, 2, 3]])

    output = generate(
        model,
        prompt,
        max_new_tokens=2,
        temperature=1.0,
    )

    assert output.shape == (1, 5)
    torch.testing.assert_close(output[:, :3], prompt)
