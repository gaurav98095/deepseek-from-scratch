"""Smoke test for prompt prefill and cache-based token decoding."""

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
    model = DeepSeekMLA(config).to(device).eval()

    prompt = torch.tensor(
        [[15, 82, 7, 104, 23]],
        dtype=torch.long,
        device=device,
    )
    generated = []

    with torch.no_grad():
        model.reset_cache()
        prompt_logits = model.prefill(prompt)
        next_token = prompt_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        print("Device:", device)
        print("Prompt:", prompt.tolist())
        print("Prefill logits:", prompt_logits.shape)
        print("Cache lengths after prefill:", [
            block.attention.cache_length for block in model.blocks
        ])

        for step in range(4):
            step_logits = model.decode_step(next_token)
            generated.append(int(next_token.item()))
            print(
                f"Decode step {step + 1}: token={int(next_token.item())}, "
                f"logits={tuple(step_logits.shape)}"
            )
            next_token = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)

    print("Generated token IDs:", generated)
    print("Final cache lengths:", [
        block.attention.cache_length for block in model.blocks
    ])
    assert all(
        block.attention.cache_length == prompt.shape[1] + len(generated)
        for block in model.blocks
    )


if __name__ == "__main__":
    main()
