"""Autoregressive generation for a consolidated DeepSeek checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.tokenizer import BPETokenizer
from src.model.final_deepseek import TinyDeepSeek
from src.sampling import sample_next_token
from src.training.checkpoint import load_inference_checkpoint


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TinyDeepSeek, dict]:
    """Load the final model and checkpoint metadata."""
    model, checkpoint = load_inference_checkpoint(checkpoint_path, device=device)
    return model, checkpoint


@torch.no_grad()
def generate(
    model: TinyDeepSeek,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    greedy: bool = False,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens with prompt prefill and one-token cached decoding."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.shape[1] > model.config.max_seq_len:
        raise ValueError("prompt exceeds maximum context length")
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generated tokens exceed maximum context length")

    tokens = input_ids.clone()
    if max_new_tokens == 0:
        return tokens

    try:
        logits = model.prefill(tokens)
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            )
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if step + 1 == max_new_tokens:
                break
            logits = model.decode_step(next_token)
        return tokens
    finally:
        model.reset_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/final_deepseek.pt")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
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
        top_p=args.top_p,
        greedy=args.greedy,
        eos_token_id=tokenizer.eos_id,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
