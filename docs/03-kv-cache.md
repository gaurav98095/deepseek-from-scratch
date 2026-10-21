# Step 03 — Add KV-cached inference

This step adds a normal multi-head-attention KV cache to inference. Training is
unchanged: it still processes complete token sequences and does not retain
cache tensors between batches.

The complete current source listings are maintained in
[02b-attention.md](./02b-attention.md) and
[02c-training.md](./02c-training.md). This document records what Step 03 adds,
where each change belongs, and how to verify it.

## Goal

Without caching, generating one token repeatedly projects K and V for every
token in the context. With caching:

```text
first call:  full prompt -> logits + K/V cache for every layer
later call:  newest token + old cache -> logits + extended cache
```

For normal multi-head attention, one layer stores:

```text
K = [batch, heads, cached_tokens, head_dim]
V = [batch, heads, cached_tokens, head_dim]
```

## Files changed

```text
src/model/attention.py
src/model/deepseek.py
src/model/__init__.py
inference.py
tests/test_attention.py
tests/test_inference.py
understanding/02-compute-kv-cache.md
docs/02b-attention.md
docs/02c-training.md
docs/03-kv-cache.md
README.md
commands.md
```

## 1. Cache types

In `src/model/attention.py`, add the one-layer cache alias after the imports:

```python
# One decoder layer caches its projected keys and values as [B, H, T, Dh].
KVCache = tuple[Tensor, Tensor]
```

In `src/model/deepseek.py`, import `KVCache` and add the whole-model cache:

```python
from .attention import KVCache, CausalSelfAttention, LearnedPositionalEmbedding

# The model keeps one independent (K, V) pair for every decoder layer.
ModelKVCache = tuple[KVCache, ...]
```

The outer tuple has exactly `num_hidden_layers` entries.

## 2. Attention cache flow

Extend `MultiHeadAttention.forward` with:

```python
past_key_value: KVCache | None = None
use_cache: bool = False
```

After projecting the current keys and values, append them to the layer cache:

```python
past_length = 0
if past_key_value is not None:
    past_key, past_value = past_key_value
    past_length = past_key.shape[2]
    key = torch.cat((past_key, key), dim=2)
    value = torch.cat((past_value, value), dim=2)

present_key_value = (key, value)
```

The implementation validates rank, shape, batch size, head count, head
dimension, device, and dtype before concatenation. When `use_cache=True`, the
method returns:

```python
return output, present_key_value
```

## 3. Offset the causal mask

With cached tokens, query length and key length are different. Replace the
square mask with:

```python
visible = torch.ones(
    sequence,
    key_length,
    dtype=torch.bool,
    device=hidden_states.device,
).tril(diagonal=past_length)
```

The diagonal offset is required because the first current query is absolute
position `past_length`, not position zero.

## 4. Continue learned positions

Extend `LearnedPositionalEmbedding.forward` with `start_position` and construct
positions as:

```python
positions = torch.arange(
    start_position,
    start_position + sequence_length,
    device=device,
)
```

The model computes `start_position` from the cache length. A token generated
after a 20-token prompt therefore receives position 20 rather than position 0.

## 5. Carry one cache through every decoder layer

`DecoderBlock.forward` accepts one layer's old cache and optionally returns the
updated cache. `DeepSeek.forward` accepts the outer `ModelKVCache`, routes entry
`i` to decoder layer `i`, and collects each returned pair in the same order.

`DeepSeekOutput` now contains:

```python
@dataclass
class DeepSeekOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: ModelKVCache | None = None
```

Cache use is rejected when labels are supplied because this cache is an
inference optimization, not a mechanism for carrying training graphs between
batches.

## 6. Use the cache in inference

In the root-level `inference.py`, initialize:

```python
past_key_values: ModelKVCache | None = None
```

The first generation iteration sends the full prompt. Each normal later
iteration sends only:

```python
model_input = token_ids[:, -1:]
```

The model call is:

```python
output = model(
    model_input,
    past_key_values=past_key_values,
    use_cache=True,
)
past_key_values = output.past_key_values
```

After every model call, inference measures the actual returned tensors and
prints a line such as:

```text
generation_step=1 kv_cache_tokens_per_layer=18 kv_cache_elements=36864 kv_cache_bytes=147456 kv_cache_mib=0.1406
```

The fields mean:

```text
generation_step             token-generation iteration
kv_cache_tokens_per_layer   context tokens already projected into K and V
kv_cache_elements           total K and V scalar count across all layers
kv_cache_bytes              actual storage based on each tensor's dtype
kv_cache_mib                bytes divided by 1024 x 1024
```

This is the cache used to predict the token at that generation step. The newly
sampled token joins the cache on the following model call, if another token is
requested.

When the cache reaches `max_position_embeddings`, inference discards it and
rebuilds a cache from the newest full context window. This keeps the learned
absolute position indices valid.

## 7. Verify numerical equivalence

The tests compare:

```text
cached attention outputs == full causal-attention outputs
cached token-by-token model logits == full-sequence model logits
```

The first catches K/V concatenation and causal-mask errors. The second also
catches incorrect layer routing and positional offsets.

`tests/test_inference.py` records the sequence length that reaches the first
query projection. For a three-token prompt and three generated tokens, it must
observe:

```text
[3, 1, 1]
```

The first value is prompt prefill. The following values prove that inference
does not send the entire growing context through attention again.

Run:

```bash
pytest tests/test_attention.py
pytest tests/test_inference.py
pytest tests/test_prepare_tinystories.py
```

The cache test also checks that a five-token example produces:

```text
K shape = [2, 3, 5, 4]
V shape = [2, 3, 5, 4]
```

## 8. Run cached inference

No new command-line option is needed; inference uses the cache automatically
and prints its size at every token-generation step:

```bash
make infer INFERENCE_PROMPT="Once upon a time"
```

The KV cache is runtime memory. It is not saved in the model checkpoint.

## 9. Commit this step

```bash
git add src/model/attention.py src/model/deepseek.py src/model/__init__.py inference.py tests/test_attention.py tests/test_inference.py understanding/02-compute-kv-cache.md docs/02b-attention.md docs/02c-training.md docs/03-kv-cache.md README.md commands.md
git commit -m "feat(inference): add KV-cached generation"
```
