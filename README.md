# DeepSeek from scratch — pre-MLA stage

This repository implements the part of the DeepSeek-style decoder covered before Multi-Head Latent Attention:

1. character-level tokenization with an unknown token;
2. token and learned positional embeddings;
3. pre-norm transformer blocks;
4. causal standard multi-head self-attention;
5. GELU feed-forward layers and tied language-model output weights;
6. next-token loss and greedy generation.

RoPE, Multi-Head Latent Attention, MoE, and other later DeepSeek-specific components are intentionally not included yet.

## Run tests

```bash
python -m pytest -q
```

The dimensions are controlled by [`ModelConfig`](deepseek_scratch/config.py). The defaults are a compact teaching configuration; pass the vocabulary size from the tokenizer and override dimensions to match the upstream series configuration when available.
