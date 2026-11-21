"""Benchmark dense SwiGLU and the educational sparse MoE FFN."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch import nn

from src.model import DeepSeekConfig, DeepSeekMoE, SwiGLUFFN


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    module: nn.Module,
    inputs: torch.Tensor,
    *,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float]:
    """Measure median and mean forward latency in milliseconds."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")

    module.eval()
    device = inputs.device
    with torch.inference_mode():
        for _ in range(warmup):
            module(inputs)
        _synchronize(device)

        timings: list[float] = []
        if device.type == "cuda":
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module(inputs)
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
        else:
            for _ in range(repeats):
                start = time.perf_counter()
                module(inputs)
                timings.append((time.perf_counter() - start) * 1_000)

    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
    }


def compare_ffn(
    config: DeepSeekConfig,
    inputs: torch.Tensor,
    *,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, object]:
    """Benchmark dense and MoE layers on identical hidden states."""
    dense = SwiGLUFFN(config.dim, config.inter_dim).to(inputs.device)
    moe = DeepSeekMoE(config).to(inputs.device)
    dense_result = benchmark(dense, inputs, warmup=warmup, repeats=repeats)
    moe_result = benchmark(moe, inputs, warmup=warmup, repeats=repeats)
    ratio = moe_result["median_ms"] / dense_result["median_ms"]
    return {
        "shape": list(inputs.shape),
        "device": str(inputs.device),
        "dense": dense_result,
        "moe": moe_result,
        "moe_to_dense_median_ratio": ratio,
    }


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("batch size and sequence length must be positive")
    config = DeepSeekConfig(
        dim=256,
        inter_dim=768,
        moe_inter_dim=256,
        n_routed_experts=8,
        n_activated_experts=args.top_k,
    )
    device = _device(args.device)
    inputs = torch.randn(args.batch_size, args.seq_len, config.dim, device=device)
    print(json.dumps(compare_ffn(config, inputs, warmup=args.warmup, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
