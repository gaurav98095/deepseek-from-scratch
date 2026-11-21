"""Print and validate the layer structure of the Tiny-DeepSeek model."""

from __future__ import annotations

import argparse

from torch import nn

from src.model import DeepSeekConfig, DeepSeekMoE, DeepSeekMoEBlock, MLA, SwiGLUFFN, TinyDeepSeek


def count_parameters(module: nn.Module) -> int:
    """Return the number of parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def inspect_architecture(
    model: TinyDeepSeek,
) -> list[dict[str, int | str]]:
    """Return one structural record for every Transformer block."""
    records: list[dict[str, int | str]] = []
    for layer_id, layer in enumerate(model.layers):
        if not isinstance(layer, DeepSeekMoEBlock):
            raise TypeError(f"layer {layer_id} is not a DeepSeekMoEBlock")
        if not isinstance(layer.attention, MLA):
            raise TypeError(f"layer {layer_id} does not use MLA")

        if layer_id < model.config.n_dense_layers:
            if not isinstance(layer.ffn, SwiGLUFFN):
                raise TypeError(f"layer {layer_id} should use dense SwiGLU")
            ffn_name = "Dense SwiGLU"
        else:
            if not isinstance(layer.ffn, DeepSeekMoE):
                raise TypeError(f"layer {layer_id} should use DeepSeekMoE")
            ffn_name = "MoE"

        records.append(
            {
                "layer": layer_id,
                "attention": "MLA",
                "ffn": ffn_name,
                "parameters": count_parameters(layer),
            }
        )
    return records


def format_report(model: TinyDeepSeek) -> str:
    """Format a compact human-readable architecture report."""
    lines = ["Complete Tiny-DeepSeek architecture", ""]
    for record in inspect_architecture(model):
        lines.append(
            f"Layer {record['layer']}: {record['attention']} + {record['ffn']}"
            f" | {record['parameters']:,} parameters"
        )
    lines.extend(
        [
            "",
            f"Total: {count_parameters(model):,}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dense-layers", type=int, default=1)
    args = parser.parse_args()

    config = DeepSeekConfig(
        vocab_size=args.vocab_size,
        n_layers=args.layers,
        n_dense_layers=args.dense_layers,
    )
    print(format_report(TinyDeepSeek(config)))


if __name__ == "__main__":
    main()
