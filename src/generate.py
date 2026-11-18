"""Autoregressive text generation for a saved dense-baseline checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.tokenizer import BPETokenizer
from src.model import DeepSeekConfig, DeepSeekSkeleton


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DeepSeekSkeleton, dict]:
    """Load a model and checkpoint metadata onto a device."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = DeepSeekConfig(**checkpoint["config"])
    model = DeepSeekSkeleton(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate(
    model: DeepSeekSkeleton,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens by repeatedly sampling the final-position logits."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    tokens = input_ids
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.max_seq_len :]
        logits = model(context)[:, -1, :] / temperature

        if top_k is not None:
            limited_top_k = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, limited_top_k, dim=-1)
            logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_id is not None and torch.all(next_token == eos_id):
            break

    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/gpt_baseline.pt")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    tokenizer_path = args.tokenizer or checkpoint.get(
        "tokenizer_path", "data/processed/tokenizer.json"
    )
    tokenizer = BPETokenizer.load(tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt produced no tokens")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
