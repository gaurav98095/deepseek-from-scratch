# DeepSeek from scratch

An educational, reproducible implementation of a small decoder-only language
model, developed step by step. The repository keeps runnable source code in
`src/` and records the exact workflow in `docs/` so each stage can be rebuilt
from a clean checkout.

## Current stage

Dataset setup is implemented for TinyStories. See
[`docs/01-data-preparation.md`](docs/01-data-preparation.md) for installation,
download, smoke-test, output-format, and commit instructions.

The attention foundation is implemented next. See
[`docs/02b-attention.md`](docs/02b-attention.md) for the tensor contracts, scope, and
focused tests.

The centralized architecture configuration is documented in
[`docs/02a-config.md`](docs/02a-config.md), with the source values in
[`configs/deepseek_v3.json`](configs/deepseek_v3.json).

The current decoder model integration is documented in
[`docs/02b-attention.md`](docs/02b-attention.md).

The runnable training pipeline is documented in
[`docs/02c-training.md`](docs/02c-training.md). It uses the small current-scaffold
configuration by default so the pipeline can be started locally.

See [`commands.md`](commands.md) for the complete command reference.

Run the offline tests with:

```bash
pytest
```

Large downloads and generated datasets are intentionally not committed.
