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

The runnable training and inference pipeline is documented in
[`docs/02c-training.md`](docs/02c-training.md). It uses the small
current-scaffold configuration by default so the pipeline can be started
locally and then used to generate text from a saved checkpoint.

KV-cached autoregressive inference is implemented in
[`docs/03-kv-cache.md`](docs/03-kv-cache.md). Prompt tokens are processed once;
later decoding steps normally process only the newest token.

For a tensor-by-tensor walkthrough of the full reference dimensions, one
decoder block, repeated layers, shapes, and parameter calculations, see
[`understanding/01-deepseek-v3-flow.md`](understanding/01-deepseek-v3-flow.md).

For the normal multi-head-attention KV-cache formulas, implementation flow,
context-length growth, memory tables, and chart, see
[`understanding/02-compute-kv-cache.md`](understanding/02-compute-kv-cache.md).

See [`commands.md`](commands.md) for the complete command reference.

The [Makefile](Makefile) provides repeatable `dev`, `prod`, numbered-step,
training, TensorBoard, and test targets.

Run the offline tests with:

```bash
pytest
```

Large downloads and generated datasets are intentionally not committed.
