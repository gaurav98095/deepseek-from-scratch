"""Train the pre-MLA DeepSeek model and write metrics for TensorBoard."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from deepseek_scratch import CharacterTokenizer, DeepSeekForCausalLM, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"), help="Training text file")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/deepseek-pre-mla"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(tokens: torch.Tensor, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, tokens.numel() - context_length - 1, (batch_size,))
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return inputs.to(device), targets.to(device)


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def parameter_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(parameter.detach().float().pow(2).sum().item() for parameter in model.parameters()))


@torch.no_grad()
def evaluate(
    model: DeepSeekForCausalLM,
    tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(args.eval_steps):
        inputs, targets = get_batch(tokens, args.batch_size, args.context_length, device)
        logits, loss = model(inputs, targets)
        assert loss is not None
        losses.append(loss.item())
        accuracies.append(token_accuracy(logits, targets))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": perplexity(mean_loss), "accuracy": sum(accuracies) / len(accuracies)}


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    path: Path,
    model: DeepSeekForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    tokenizer_stoi: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "config": asdict(model.config),
            "tokenizer_stoi": tokenizer_stoi,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.context_length < 1:
        raise ValueError("steps, batch_size, and context_length must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    encoded = tokenizer.encode_tensor(text)
    split = max(args.context_length + 2, int(encoded.numel() * 0.9))
    if split >= encoded.numel() - 1:
        raise ValueError("data file is too short; provide more than context_length tokens")
    train_tokens, validation_tokens = encoded[:split], encoded[split:]
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = DeepSeekForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    args.run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.run_dir)
    writer.add_text("config", str(vars(args)))
    writer.add_text("model", str(config))
    print(f"device={device} vocab_size={config.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}")
    tokens_seen = start_step * args.batch_size * args.context_length
    last_log_time = time.perf_counter()
    rolling_loss = 0.0
    rolling_accuracy = 0.0

    for step in range(start_step, args.steps):
        step_start = time.perf_counter()
        current_lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        inputs, targets = get_batch(train_tokens, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits, loss = model(inputs, targets)
        assert loss is not None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += inputs.numel()
        rolling_loss += loss.item()
        rolling_accuracy += token_accuracy(logits.detach(), targets)

        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == 1:
            elapsed = time.perf_counter() - last_log_time
            count = args.log_interval if completed_step % args.log_interval == 0 else 1
            mean_loss = rolling_loss / count
            mean_accuracy = rolling_accuracy / count
            tokens_per_second = count * args.batch_size * args.context_length / max(elapsed, 1e-9)
            writer.add_scalar("train/loss", mean_loss, completed_step)
            writer.add_scalar("train/perplexity", perplexity(mean_loss), completed_step)
            writer.add_scalar("train/token_accuracy", mean_accuracy, completed_step)
            writer.add_scalar("train/learning_rate", current_lr, completed_step)
            writer.add_scalar("train/gradient_norm", float(grad_norm), completed_step)
            writer.add_scalar("train/parameter_norm", parameter_norm(model), completed_step)
            writer.add_scalar("performance/tokens_per_second", tokens_per_second, completed_step)
            writer.add_scalar("performance/step_time_seconds", (time.perf_counter() - step_start), completed_step)
            writer.add_scalar("performance/tokens_seen", tokens_seen, completed_step)
            print(f"step {completed_step:>6} | loss {mean_loss:.4f} | ppl {perplexity(mean_loss):.2f} | tok/s {tokens_per_second:.0f}")
            rolling_loss = 0.0
            rolling_accuracy = 0.0
            last_log_time = time.perf_counter()

        if completed_step % args.eval_interval == 0 or completed_step == args.steps:
            metrics = evaluate(model, validation_tokens, args, device)
            for name, value in metrics.items():
                writer.add_scalar(f"validation/{name}", value, completed_step)
            save_checkpoint(args.run_dir / "last.pt", model, optimizer, completed_step, args, tokenizer.stoi)
            print(f"           validation loss {metrics['loss']:.4f} | ppl {metrics['perplexity']:.2f} | accuracy {metrics['accuracy']:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
