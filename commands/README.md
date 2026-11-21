# Command Guide

Run these commands from the repository root unless a command says otherwise.

## One-time environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA machine, check the driver and GPU before using CUDA:

```bash
nvidia-smi
```

## Article 1 — Understanding DeepSeek

Open the executable notebook:

```bash
jupyter lab notebooks/01-understanding-deepseek.ipynb
```

The notebook currently demonstrates configuration values, token IDs, embedding lookup, output shapes, and parameter counts.

## Article 2 — Setting Up the Codebase

Run the model skeleton's forward-pass smoke test from the repository root:

```bash
python -m src.smoke_test
```

The same script can be run from inside `src/`:

```bash
cd src
python smoke_test.py
cd ..
```

Run only the model tests:

```bash
python -m pytest src/tests/test_model.py -v
```

These checks cover output shape, parameter count, gradient flow, and sequence-length validation.

## Article 3 — Building a Tokenizer

Train and inspect the toy byte-level BPE tokenizer:

```bash
python -m src.train_tokenizer
```

Run only the tokenizer tests:

```bash
python -m pytest src/tests/test_tokenizer.py -v
```

These checks cover pair merging, text round-tripping, Unicode, EOS handling, vocabulary size, and invalid configuration values.

## Article 4 — Preparing the Pretraining Dataset

Prepare a bounded FineWeb-Edu sample from the repository root:

```bash
python -m src.data.prepare_data \
  --max-documents 5000 \
  --tokenizer-documents 1000 \
  --tokenizer-vocab-size 350
```

Use a smaller smoke run first when checking the pipeline:

```bash
python -m src.data.prepare_data \
  --max-documents 20 \
  --tokenizer-documents 10 \
  --tokenizer-vocab-size 300
```

Inspect the generated files:

```bash
python -m src.data.inspect_data
```

The generated files will live under `data/processed/`:

```text
tokenizer.json
train.bin
val.bin
metadata.json
```

Do not commit generated datasets or binary token files.

## Article 5 — Building the PyTorch DataLoader

The data loader implementation lives under `src/data/dataset.py` and `src/data/dataloader.py`. Run the dataset checks with:

```bash
python -m pytest src/tests/test_dataset.py -v
python -m pytest src/tests/test_dataloader.py -v
```

The loader reads `data/processed/train.bin` and `val.bin` with memory mapping and returns `[batch, sequence]` input-target pairs.

## Article 6 — Token Embeddings and RMSNorm

The RMSNorm implementation lives under `src/model/rmsnorm.py`. Run:

```bash
python -m pytest src/tests/test_rmsnorm.py -v
```

The tests cover the numerical formula, zero inputs, gradient flow, scale behavior, and BF16-safe statistics.

## Article 7 — Causal Self-Attention

The attention layer belongs in `src/model/attention.py`, with transformer integration in `src/model/transformer.py` and focused checks in `src/tests/test_attention.py`. Run the attention checks with:

```bash
python -m pytest src/tests/test_attention.py -v
```

The important checks cover output shapes, causal future-token isolation, finite gradients, and projection parameter counts.

## Article 8 — Rotary Position Embeddings

RoPE is implemented in `src/model/rope.py` and integrated into `src/model/attention.py`. Run the rotation and causal-attention checks with:

```bash
python -m pytest src/tests/test_rope.py -v
```

The checks cover shape preservation, length preservation, relative-position behavior, gradients, and causal masking after adding RoPE.

The notebook adds the repository root to `sys.path` automatically, so its imports work when Jupyter is opened from either the repository root or the `notebooks/` directory.

## Article 9 — Complete GPT-Style Transformer

The dense baseline adds `SwiGLUFFN` in `src/model/ffn.py` and combines it with attention and RMSNorm in `src/model/transformer.py`. Run the focused checks with:

```bash
python -m pytest src/tests/test_transformer.py -v
```

These checks cover the FFN shape, complete block shape, model logits, and gradient flow. The companion notebook is `notebooks/09-first-complete-gpt.ipynb`.

Train the dense baseline after preparing `data/processed`:

```bash
python -m src.train --steps 300 --batch-size 8 --seq-len 128
```

Generate from its checkpoint:

```bash
python -m src.generate --checkpoint checkpoints/gpt_baseline.pt --prompt "The history of"
```

The training and generation scripts are educational baseline implementations. They do not use a KV cache yet.

## Article 10 — Why MLA Exists

The KV-cache sizing utility is in `src/inference/kv_cache_memory.py`. Run its checks with:

```bash
python -m pytest src/tests/test_kv_cache.py -v
```

The utility compares logical cache storage for MHA, GQA, MQA, and a toy MLA configuration. The companion notebook is `notebooks/10-why-mla-exists.ipynb`.

## Article 11 — Low-Rank Query and KV Compression

The experimental MLA projection branches are in `src/model/mla.py`. Run the focused checks with:

```bash
python -m pytest src/tests/test_mla.py -v
```

These checks cover query shapes, KV latent and reconstruction shapes, content-only causal attention, and gradient flow. The companion notebook is `notebooks/11-low-rank-q-kv-compression.ipynb`.

## Article 12 — Decoupled RoPE in MLA

The complete naive MLA layer is `MLA` in `src/model/mla.py`. Run the MLA checks with:

```bash
python -m pytest src/tests/test_mla.py -v
```

The tests cover the complete output shape, gradients, causal masking, and equivalence of concatenated versus separate content/position score calculations. The companion notebook is `notebooks/12-decoupled-rope.ipynb`.

## Article 13 — Complete MLA Integration

The integrated dense MLA model is `DeepSeekMLA` in `src/model/deepseek_mla.py`, with `DeepSeekBlock` in `src/model/transformer.py`. Run the full-network checks with:

```bash
python -m pytest src/tests/test_deepseek_mla.py -v
```

Run the short GPT-versus-MLA comparison after preparing `data/processed`:

```bash
python -m src.experiments.compare_gpt_mla --steps 50 --batch-size 4 --seq-len 128
```

Run the lightweight MLA forward-pass smoke test without prepared data:

```bash
python -m src.smoke_test_mla
```

## Article 14 — Compressed KV Cache and Matrix Absorption

Open the executable notebook:

```bash
jupyter lab notebooks/14-kv-cache-and-absorption.ipynb
```

Run the focused standalone-MLA and full-model cache tests:

```bash
python -m pytest src/tests/test_mla_cache.py -v
```

The stateful cache implementation is in `src/model/mla_cache.py`. Both
`CachedMLA` and `DeepSeekMLA` expose `reset_cache()`, `prefill()`, and
`decode_step()` for explicit inference-state management. The article is
`docs/14-kv-cache-and-absorption.md`.

## Article 15 — Why DeepSeek Uses Mixture of Experts

Open the executable notebook:

```bash
jupyter lab notebooks/15-mixture-of-expert.ipynb
```

This article is conceptual. The notebook inspects the existing MoE
configuration, calculates total versus active parameters, and demonstrates
illustrative top-k routing scores. Expert networks and production routing are
implemented in later articles.

## Article 16 — Implementing DeepSeek Expert Networks

Open the executable notebook:

```bash
jupyter lab notebooks/16-implementing-expert-networks.ipynb
```

Run the expert-bank tests:

```bash
python -m pytest src/tests/test_experts.py -v
```

The implementation is in `src/model/experts.py`. It covers individual
SwiGLU experts, independently registered routed experts, the shared expert,
manual weighted combination, parameter counts, and sparse gradient flow.
Top-k routing and token dispatch are intentionally not included yet.

## Article 17 — Top-k Routing and Token Dispatch

Open the executable notebook:

```bash
jupyter lab notebooks/17-topk-routing.ipynb
```

Run the routing and sparse-dispatch tests:

```bash
python -m pytest src/tests/test_moe.py -v
```

The router is in `src/model/router.py`, and the sparse MoE layer is in
`src/model/moe.py`. The implementation uses sigmoid scores, top-k selection,
`index_select` token dispatch, and `index_add` output aggregation.

## Article 18 — Auxiliary-Loss-Free Load Balancing

Open the executable notebook:

```bash
jupyter lab notebooks/18-load-balancing-router.ipynb
```

Run the balancing tests:

```bash
python -m pytest src/tests/test_balancing.py -v
```

Run the controlled bias-feedback experiment:

```bash
python -m src.experiments.moe_balancing
```

The router now keeps non-trainable `expert_bias` state. Biases affect expert
selection only; selected output weights still use the original sigmoid scores.

## Article 19 — Building the Complete DeepSeekMoE Transformer

Open the executable notebook:

```bash
jupyter lab notebooks/19-deepseek-moe.ipynb
```

Run the complete-model tests:

```bash
python -m pytest src/tests/test_deepseek_moe.py -v
```

Run a short training experiment after preparing `data/processed`:

```bash
python -m src.train_moe --steps 100 --batch-size 4 --seq-len 128
```

The model is in `src/model/deepseek_moe.py`. It uses dense SwiGLU in the
initial layers and MoE in later layers. `src/training/moe_balancer.py` records
actual router assignments and updates balancing biases after optimizer steps.

## Article 20 — DeepSeekMoE Experiments

Open the executable notebook:

```bash
jupyter lab notebooks/20-deepseekmoe-experiments.ipynb
```

Inspect routing decisions and load metrics from a Python session with
`src.experiments.analyze_routing.inspect_model_routing`. Run the isolated
dense-versus-MoE benchmark with:

```bash
python -m src.experiments.benchmark_ffn \
  --device cpu \
  --batch-size 4 \
  --seq-len 128 \
  --warmup 5 \
  --repeats 20
```

Use `--device cuda` on a CUDA machine. Run the experiment utility tests with:

```bash
python -m pytest src/tests/test_moe_experiments.py -v
```

The article covers dense-versus-MoE comparisons, routing balance, top-k
active-parameter trade-offs, shared-expert ablations, specialization, and
naive token-dispatch latency.

## Article 21 — Understanding the Complete Transformer Block

Open the executable notebook:

```bash
jupyter lab notebooks/21-understanding-tf-block.ipynb
```

Run the architecture inspection report:

```bash
python -m src.experiments.inspect_architecture
```

Run the focused architecture checks:

```bash
python -m pytest src/tests/test_architecture_inspection.py -v
```

The article traces pre-RMSNorm, both residual connections, dense-prefix/MoE-
suffix layer selection, tensor shapes, parameter accounting, and the place of
Multi-Token Prediction around the main Transformer stack.

## Run all current tests

```bash
python -m pytest src/tests -v
```

## Useful checks

Inspect the repository files without running the model:

```bash
find src -maxdepth 3 -type f | sort
git status --short
```

Check Python formatting at the Git level:

```bash
git diff --check
```

## Import and path note

Use the `python -m src...` form from the repository root. It keeps `src` on the import path and makes imports such as `from src.model import DeepSeekConfig` work consistently.

## Current scope

The repository currently contains the complete dense attention-and-RoPE baseline, tokenizer, dataset preparation, data-loading foundations, baseline training, autoregressive generation, MLA integration, compressed KV-cache inference, independent MoE expert networks, single-device top-k routing, and expert-bias load balancing. Distributed training and other scaling commands will be added as those components are implemented.
