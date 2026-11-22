"""Smoke test for final-model prompt prefill and cached token decoding."""

import torch

from src.model import DeepSeekConfig, TinyDeepSeek


def main() -> None:
    torch.manual_seed(42)
    config = DeepSeekConfig(
        vocab_size=512,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDeepSeek(config).to(device).eval()
    prompt = torch.tensor([[15, 82, 7, 104, 23]], dtype=torch.long, device=device)
    generated: list[int] = []

    with torch.no_grad():
        prompt_logits = model.prefill(prompt)
        next_token = prompt_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        print("Device:", device)
        print("Prompt:", prompt.tolist())
        print("Prefill logits:", tuple(prompt_logits.shape))
        print("Cache lengths after prefill:", [
            layer.attention.cache_length for layer in model.layers
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
        layer.attention.cache_length for layer in model.layers
    ])
    assert all(
        layer.attention.cache_length == prompt.shape[1] + len(generated)
        for layer in model.layers
    )


if __name__ == "__main__":
    main()
