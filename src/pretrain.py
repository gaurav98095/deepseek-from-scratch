"""End-to-end resumable pretraining for the small DeepSeek-style model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.token_batches import TokenBatcher
from .data.tokenizer_identity import file_sha256
from .model import DeepSeekConfig, DeepSeekWithMTP
from .training import (
    FixedTokenWindows,
    MoELoadBalancer,
    create_optimizer,
    efficient_train_step,
    evaluate_fixed_windows,
    load_training_checkpoint,
    save_training_checkpoint,
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators used by the run."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    path = data_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "tokenizer_vocab_size" not in metadata or "dtype" not in metadata:
        raise ValueError("metadata must contain tokenizer_vocab_size and dtype")
    tokenizer_path = data_dir / metadata.get("tokenizer_path", "tokenizer.json")
    if tokenizer_path.is_file():
        metadata.setdefault("tokenizer_sha256", file_sha256(tokenizer_path))
    return metadata


def build_config(metadata: dict[str, Any], args: argparse.Namespace) -> DeepSeekConfig:
    """Build the model configuration from dataset metadata and CLI settings."""
    return DeepSeekConfig(
        vocab_size=int(metadata["tokenizer_vocab_size"]),
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
        n_heads=args.n_heads,
        inter_dim=args.inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
    )


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Return settings that must stay fixed when a run is resumed."""
    return {
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seq_len": args.seq_len,
        "warmup_steps": args.warmup_steps,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "max_grad_norm": args.max_grad_norm,
        "mtp_weight": args.mtp_weight,
        "gamma": args.gamma,
        "eval_interval": args.eval_interval,
        "eval_windows": args.eval_windows,
        "tokenizer_sha256": args.tokenizer_sha256,
        "dtype": args.dtype,
        "bf16": args.bf16,
        "activation_checkpointing": args.activation_checkpointing,
    }


def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "loss",
        "main_loss",
        "mtp_loss",
        "lr",
        "grad_norm",
        "mean_max_load_ratio",
        "val_loss",
        "perplexity",
        "tokens_per_second",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def run_pretraining(args: argparse.Namespace) -> dict[str, Any]:
    """Run training from scratch or resume from ``args.resume``."""
    if args.steps <= 0 or args.batch_size <= 0 or args.accum_steps <= 0:
        raise ValueError("steps, batch_size, and accum_steps must be positive")
    if (
        args.eval_interval <= 0
        or args.eval_windows <= 0
        or args.save_interval <= 0
    ):
        raise ValueError("evaluation and checkpoint intervals must be positive")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    metadata = load_metadata(data_dir)
    dtype = args.dtype or metadata["dtype"]
    args.dtype = dtype
    config = build_config(metadata, args)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = DeepSeekWithMTP(config, mtp_weight=args.mtp_weight).to(device)
    model.set_activation_checkpointing(args.activation_checkpointing)
    optimizer = create_optimizer(
        model,
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    train_batcher = TokenBatcher(
        data_dir / "train.bin",
        seq_len=args.seq_len,
        dtype=dtype,
        seed=args.seed,
    )
    val_batcher = TokenBatcher(
        data_dir / "val.bin",
        seq_len=args.seq_len,
        dtype=dtype,
        seed=args.seed + 1,
    )
    val_windows = FixedTokenWindows(
        data_dir / "val.bin",
        seq_len=args.seq_len,
        dtype=dtype,
    )
    args.tokenizer_sha256 = metadata.get("tokenizer_sha256")
    settings = run_settings(args)
    start_step = 0

    if args.resume:
        state = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            train_batcher=train_batcher,
            val_batcher=val_batcher,
            expected_run_settings=settings,
        )
        start_step = int(state["step"])
        print(f"Resumed from step {start_step}: {args.resume}")
        saved_best = state.get("best_val_loss")
        if saved_best is None:
            saved_best = state.get("metrics", {}).get("val_loss")
        best_val_loss = float(saved_best) if saved_best is not None else float("inf")
    else:
        best_val_loss = float("inf")

    log_path = Path(args.log_path)
    last_metrics: dict[str, Any] = {}
    with MoELoadBalancer(model, gamma=args.gamma) as balancer:
        for step in range(start_step + 1, args.steps + 1):
            microbatches = [
                train_batcher.next_batch(args.batch_size).to(
                    device,
                    non_blocking=True,
                )
                for _ in range(args.accum_steps)
            ]
            metrics = efficient_train_step(
                model,
                optimizer,
                balancer,
                microbatches,
                step=step,
                total_steps=args.steps,
                warmup_steps=args.warmup_steps,
                max_lr=args.max_lr,
                min_lr=args.min_lr,
                max_grad_norm=args.max_grad_norm,
                use_bf16=args.bf16,
            )
            last_metrics = dict(metrics)
            validation: dict[str, Any] = {}
            is_best = False
            if step % args.eval_interval == 0 or step == args.steps:
                validation = evaluate_fixed_windows(
                    model.main_model,
                    val_windows,
                    batch_size=args.batch_size,
                    max_windows=args.eval_windows,
                )
                # Evaluation also traverses MoE routers, but must not affect
                # the next training-step balancing update.
                balancer.clear()
                model.train()
                last_metrics.update(validation)
                if validation["val_loss"] < best_val_loss:
                    best_val_loss = float(validation["val_loss"])
                    is_best = True

            row = {"step": step, **last_metrics}
            _append_log(log_path, row)
            print(
                f"step {step:>6}/{args.steps} | "
                f"train {metrics['loss']:.4f} | "
                f"main {metrics['main_loss']:.4f} | "
                f"lr {metrics['lr']:.7f}"
                + (
                    f" | val {validation['val_loss']:.4f}"
                    f" | ppl {validation['perplexity']:.2f}"
                    if validation
                    else ""
                )
            )

            if step % args.save_interval == 0 or step == args.steps:
                save_training_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                    run_settings=settings,
                    metrics=row,
                    best_val_loss=best_val_loss,
                )
                print(f"saved checkpoint: {args.checkpoint}")
            if is_best:
                save_training_checkpoint(
                    args.best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                    run_settings=settings,
                    metrics=row,
                    best_val_loss=best_val_loss,
                )
                print(f"saved best checkpoint: {args.best_checkpoint}")

    return {"step": max(start_step, args.steps), **last_metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--best-checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-path", default="training_log.csv")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--eval-windows",
        "--eval-batches",
        dest="eval_windows",
        type=int,
        default=20,
    )
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mtp-weight", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--activation-checkpointing", action="store_true")

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-dense-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--inter-dim", type=int, default=768)
    parser.add_argument("--moe-inter-dim", type=int, default=256)
    parser.add_argument("--q-lora-rank", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=64)
    parser.add_argument("--qk-nope-head-dim", type=int, default=32)
    parser.add_argument("--qk-rope-head-dim", type=int, default=16)
    parser.add_argument("--v-head-dim", type=int, default=32)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    parser.add_argument("--n-shared-experts", type=int, default=1)
    parser.add_argument("--n-activated-experts", type=int, default=2)
    return parser


def main() -> None:
    run_pretraining(build_parser().parse_args())


if __name__ == "__main__":
    main()
