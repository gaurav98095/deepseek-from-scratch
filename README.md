# DeepSeek From Scratch

A clean PyTorch workspace for building and understanding a small DeepSeek-style language model from the ground up.

Start with [Article 1 — Understanding DeepSeek](docs/01-understanding-deepseek.md), continue with [Setting Up the DeepSeek Codebase](docs/02-setting-up-codebase.md), then read [Building a Tokenizer From Scratch](docs/03-build-tokenizer-from-scratch.md), [Preparing the Pretraining Dataset](docs/04-prepare-pretraining-dataset.md), [Building the PyTorch DataLoader](docs/05-building-data-loader.md), [Token Embeddings and RMSNorm](docs/06-embeddings-and-rms-norm.md), [Causal Self-Attention](docs/07-causal-self-attention.md), [Rotary Position Embeddings](docs/08-rope-from-scratch.md), [Building a Complete GPT-Style Transformer](docs/09-first-complete-gpt.md), [Why DeepSeek Uses MLA](docs/10-why-mla-exists.md), [Low-Rank Query and KV Compression](docs/11-low-rank-q-kv-compression.md), [Decoupled RoPE in MLA](docs/12-decoupled-rope.md), [Complete MLA Integration](docs/13-complete-mla-integration.md), [Compressed KV Cache and Matrix Absorption](docs/14-kv-cache-and-absorption.md), [Why DeepSeek Uses Mixture of Experts](docs/15-mixture-of-expert.md), [Implementing DeepSeek Expert Networks](docs/16-implementing-expert-networks.md), [Top-k Routing and Token Dispatch](docs/17-topk-routing.md), [Auxiliary-Loss-Free Load Balancing](docs/18-load-balancing-router.md), [Building the Complete DeepSeekMoE Transformer](docs/19-deepseek-moe.md), [DeepSeekMoE Experiments](docs/20-deepseekmoe-experiments.md), [Understanding the Complete Transformer Block](docs/21-understanding-tf-block.md), [Assembling the Final DeepSeek Model](docs/22-assembling-deepseek.md), [Implementing Multi-Token Prediction](docs/23-multi-token-prediction.md), [Complete Architecture Validation](docs/24-complete-deepseek-archi-and-validation.md), [Training DeepSeek with AdamW and Learning-Rate Scheduling](docs/25-training-deepseek-loss-lr.md), [BF16, Gradient Accumulation, and GPU Training](docs/26-bf16-gradaccum-gputrain.md), [End-to-End DeepSeek Pretraining](docs/27-complete-pretraining.md), and [Evaluating DeepSeek](docs/28-eval-and-analysis.md) alongside the implementation and notebooks.

## Repository layout

- `src/model/` — model configuration, embeddings, RMSNorm, RoPE, causal attention, SwiGLU, MLA projections, and DeepSeek components.
- `src/model/mla_cache.py` — stateful compressed-cache MLA inference.
- `src/model/experts.py` — independent SwiGLU experts and the shared expert bank.
- `src/model/router.py` and `src/model/moe.py` — sigmoid top-k routing and sparse token dispatch.
- `src/model/final_deepseek.py` — consolidated full-sequence and cached-inference model.
- `src/training/moe_balancer.py` — post-step routing-bias updates.
- `src/training/checkpoint.py` — final-model checkpoint save/load helpers.
- `src/model/mtp.py` and `src/model/deepseek_with_mtp.py` — one-stage MTP training path.
- `src/tests/test_final_validation.py` — end-to-end architecture and numerical validation.
- `src/training/optimizer.py`, `scheduler.py`, and `train_step.py` — FP32 optimization utilities.
- `src/training/efficient_train_step.py` and `profiling.py` — BF16, accumulation, and GPU profiling utilities.
- `src/data/token_batches.py` — reproducible random windows from memory-mapped token files.
- `src/training/evaluate.py` and `full_checkpoint.py` — validation and resumable training state.
- `src/training/fixed_evaluate.py` and `routing_diagnostics.py` — deterministic scoring and MoE utilization reports.
- `src/evaluate_checkpoint.py` and `src/plot_training.py` — checkpoint scorecards and training-curve plots.
- `src/pretrain.py` — complete pretraining entry point.
- `src/data/` — tokenizer, dataset preparation, memory-mapped dataset, and DataLoader tools.
- `src/tests/` — shape, numerical, gradient, and integration tests.
- `commands/` — commands for notebooks, smoke tests, and each article's checks.
- `src/smoke_deepseekmoe.py`, `smoke_inference.py`, and `smoke_pretrain.py` — final-model smoke checks.
- `src/train_tokenizer.py` — tokenizer training demonstration.
- `src/train_moe.py` — compact single-device MoE training entry point.
- `src/generate.py` — checkpoint loading and autoregressive generation.
- `docs/` — written explanations and architecture diagrams.
- `notebooks/` — executable experiments and tensor-shape demonstrations.
- `requirements.txt` — Python dependencies.
- `Makefile` — dataset, test, smoke, training, and resume commands.

The implementation will stay readable and modular: each component should be understandable on its own, tested, and then integrated into the growing model.

Generated datasets, checkpoints, logs, and Python cache files are kept out of version control.

## Common workflow

```bash
make prepare-data
make smoke-pretrain
make test
make run-pretraining
make evaluate-checkpoint CHECKPOINT=checkpoints/latest.pt
make routing-report CHECKPOINT=checkpoints/latest.pt
make plot-training LOG_PATH=training_log.csv
```

Use `make resume-pretraining` to continue from `checkpoints/latest.pt`. The
training loop also keeps `checkpoints/best.pt`, selected using fixed validation
windows. The documentation includes architecture diagrams, tensor-flow
visuals, training curves, routing comparisons, and checkpoint workflows.
