"""Generate text from the newest or explicitly selected model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.byte_tokenizer import ByteTokenizer
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek, ModelKVCache


def _build_parser() -> argparse.ArgumentParser:
    # Keep inference independent from the training loop so a saved model can be
    # used without recreating a dataset or optimizer.
    parser = argparse.ArgumentParser(description=__doc__)
    # The architecture must match the configuration used to create the checkpoint.
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    # If omitted, the newest checkpoint in this directory is selected by mtime.
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    # The prompt is converted to byte IDs by the current educational tokenizer.
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    # This controls how many new tokens are appended after the prompt.
    parser.add_argument("--max-new-tokens", type=int, default=50)
    # Temperature 0 uses greedy decoding; higher values produce more variation.
    parser.add_argument("--temperature", type=float, default=1.0)
    # Top-k limits sampling to the most likely candidates when it is positive.
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _choose_device(requested: str) -> torch.device:
    # Fail clearly instead of silently moving a requested CUDA run to the CPU.
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _latest_checkpoint(directory: Path) -> Path:
    # Modification time makes the default behavior select the most recent save.
    checkpoints = list(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints found in {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> DeepSeek:
    # Construct the same module layout before loading the saved parameter tensors.
    config = ModelConfig.from_json(config_path)
    model = DeepSeek(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Training checkpoints contain optimizer state as well as the model weights.
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def generate(
    model: DeepSeek,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Append tokens to ``prompt`` and return the complete decoded text."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    device = next(model.parameters()).device
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    past_key_values: ModelKVCache | None = None

    for _ in range(max_new_tokens):
        # The first call processes the complete prompt and builds every layer's
        # cache. Later calls normally process only the newest generated token.
        cached_length = 0
        if past_key_values is not None:
            cached_length = past_key_values[0][0].shape[2]

        if past_key_values is None or cached_length >= model.config.max_position_embeddings:
            # At the context limit, discard the old cache and rebuild it from the
            # newest window. This also restarts learned positions at zero.
            model_input = token_ids[:, -model.config.max_position_embeddings :]
            past_key_values = None
        else:
            model_input = token_ids[:, -1:]

        output = model(
            model_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("the model did not return a KV cache")
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                # Suppress all but the top-k scores before sampling.
                limit = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, limit)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        token_ids = torch.cat((token_ids, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids[0].tolist())


def main() -> None:
    # Parse options once so command-line use and future integrations share behavior.
    args = _build_parser().parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)
    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    model = _load_model(args.config, checkpoint, device)
    text = generate(
        model,
        ByteTokenizer(),
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    print(f"Checkpoint: {checkpoint}")
    print(text)


if __name__ == "__main__":
    main()
