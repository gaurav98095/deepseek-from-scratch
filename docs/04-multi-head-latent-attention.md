# Step 04 — Implement Multi-Head Latent Attention

Step 04 replaces the decoder's ordinary multi-head attention with an
educational multi-head latent attention (MLA) module. The implementation keeps
the learned absolute position embeddings already used by this project and
does not apply rotary position embeddings (RoPE).

## Projections

For input hidden states `x` with shape `[B, T, D]`, MLA first forms a
low-dimensional query representation and expands it into query heads:

```text
q_latent = LayerNorm(x W_DQ)                       [B, T, q_lora_rank]
Q        = reshape(q_latent W_UQ)                  [B, H, T, Dh]
```

Keys and values share a compressed per-token representation:

```text
kv_latent = LayerNorm(x W_DKV)                     [B, T, kv_lora_rank]
(K, V)    = reshape(kv_latent W_UKV)               each [B, H, T, Dh]
```

Queries attend to the expanded keys, then combine the values and project the
merged heads back to the model's hidden size. The attention calculation uses
the usual scaled dot product and causal mask. There is no RoPE key/query split
or rotary transformation; token positions come from the model's learned
absolute position embeddings.

## Compressed cache

Each decoder layer stores only `kv_latent`, not the expanded keys and values:

```text
one layer cache = [B, cached_tokens, kv_lora_rank]
all layer elements = B x L x T x kv_lora_rank
cache bytes = elements x bytes_per_element
```

For comparison, ordinary MHA stores `2 x B x L x T x H x Dh` elements.
The reduction depends on the configured latent rank. In this implementation,
the cached latent is expanded to K and V for attention on each call; optimizing
that expansion by absorbing weights is outside this educational step.

During cached decoding, a new token's compressed representation is appended
to the latent cache. The causal mask uses the cached prefix length, while the
learned absolute position embedding continues from that same offset.

## Configuration

The configuration provides:

```text
num_attention_heads  H
head_dim             Dh
q_lora_rank          query compression width
kv_lora_rank         shared key/value compression width and cache width
```

The reference config uses query rank `1536` and KV rank `512`; the tiny
training config uses ranks `128` and `64`. Hidden size no longer needs to equal
`num_attention_heads x head_dim`, because the final attention projection maps
the merged head output back to the residual-stream width.

Changing attention changes parameter names and cache representation. Existing
checkpoints trained with standard attention are not compatible and must be
retrained with the matching config.

## Implementation and verification

The decoder uses `CausalMultiHeadLatentAttention` from
`src/model/attention.py`. `DeepSeekOutput.past_key_values` contains one latent
tensor per layer, and inference reports its measured element and byte counts.
The previous `MultiHeadAttention` and `CausalSelfAttention` classes and their
expanded K/V cache implementation were removed; the model API retains the MLA
implementation and its compressed cache.

Run the focused tests with:

```bash
pytest tests/test_attention.py tests/test_inference.py
```

They verify causal masking, equivalence between full and token-by-token
decoding, the compressed cache shape, and inference's reported cache size.
