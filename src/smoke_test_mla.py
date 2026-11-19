"""Small forward-pass check for the dense MLA language model."""

import torch

try:
    from .model import DeepSeekConfig, DeepSeekMLA
except ImportError:
    # Supports running this file directly from the src/ directory.
    from model import DeepSeekConfig, DeepSeekMLA


def main() -> None:
    torch.manual_seed(42)
    config = DeepSeekConfig(
        vocab_size=512,
        dim=64,
        n_layers=2,
        n_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        inter_dim=128,
        max_seq_len=32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeekMLA(config).to(device)

    tokens = torch.tensor(
        [
            [15, 82, 7, 104, 23, 41],
            [11, 24, 35, 45, 67, 89],
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
    assert logits.shape == (2, 6, config.vocab_size)


if __name__ == "__main__":
    main()
