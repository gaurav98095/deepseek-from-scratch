"""Small forward-pass check for the attention-based model baseline."""

import torch

try:
    from .model import DeepSeekConfig, DeepSeekSkeleton
except ImportError:
    # Supports: python smoke_test.py from inside the src/ directory.
    from model import DeepSeekConfig, DeepSeekSkeleton


def main() -> None:
    torch.manual_seed(42)
    config = DeepSeekConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeekSkeleton(config).to(device)

    tokens = torch.tensor(
        [
            [15, 82, 7, 104],
            [11, 24, 35, 45],
        ],
        dtype=torch.long,
        device=device,
    )
    logits = model(tokens)

    print("Device:", device)
    print("Input:", tokens.shape)
    print("Output:", logits.shape)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {num_params:,}")
    assert logits.shape == (2, 4, config.vocab_size)


if __name__ == "__main__":
    main()
