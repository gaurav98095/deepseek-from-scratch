"""Inference utilities and CLI for checkpoints produced by ``main.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepseek_scratch import CharacterTokenizer, DeepSeekForCausalLM, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/deepseek-pre-mla/last.pt"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"), help="Fallback corpus for old checkpoints")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Sample one token per batch item using temperature, top-k, and nucleus filtering."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probability = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probability - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


@torch.inference_mode()
def generate(
    model: DeepSeekForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Generate tokens through the stable inference interface.

    The context crop is deliberately isolated here. A KV-cache can replace this
    full-prefix forward pass later without changing the CLI or sampling code.
    """
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def load_checkpoint(checkpoint_path: Path, device: torch.device, fallback_data: Path) -> tuple[DeepSeekForCausalLM, CharacterTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_values = checkpoint.get("config")
    if config_values is None:
        training_args = checkpoint.get("args", {})
        config_values = {
            "vocab_size": checkpoint["model"]["token_embedding.weight"].shape[0],
            "context_length": training_args.get("context_length", 128),
            "embedding_dim": training_args.get("embedding_dim", 128),
            "num_layers": training_args.get("num_layers", 4),
            "num_heads": training_args.get("num_heads", 4),
            "dropout": training_args.get("dropout", 0.0),
        }
    config = ModelConfig(**config_values)
    model = DeepSeekForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    vocabulary = checkpoint.get("tokenizer_stoi")
    if vocabulary is None:
        vocabulary = CharacterTokenizer.from_text(fallback_data.read_text(encoding="utf-8")).stoi
    tokenizer = CharacterTokenizer(stoi=vocabulary, itos={int(index): token for token, index in vocabulary.items()})
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    device = choose_device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device, args.data)
    prompt_ids = tokenizer.encode_tensor(args.prompt, device=device).unsqueeze(0)
    generated_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(generated_ids[0]))


if __name__ == "__main__":
    main()
