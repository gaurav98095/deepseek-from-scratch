import torch

from model import DeepSeekConfig, TokenEmbedding


def test_token_embedding_shape_and_parameter_count() -> None:
    config = DeepSeekConfig(vocab_size=128, dim=32, n_heads=4)
    model = TokenEmbedding(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 5))

    output = model(token_ids)

    assert output.shape == (2, 5, config.dim)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        config.vocab_size * config.dim
    )
