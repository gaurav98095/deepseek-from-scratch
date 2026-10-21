# Computing KV-cache size for normal multi-head attention

This document computes the key-value cache for the current educational
DeepSeek flow. It assumes ordinary multi-head self-attention where every
attention head has its own keys and values.

The checked-in reference configuration is:

```text
number of decoder layers L = 61
number of attention heads H = 128
head dimension Dh = 56
hidden size D = 7168
maximum context length P = 163840
```

The relationship between the hidden size and attention dimensions is:

```text
D = H x Dh
7168 = 128 x 56
```

The current root-level [inference.py](../inference.py) now reuses a KV cache.
It performs one full prompt pass, called the prefill, and then normally sends
only one newly generated token through the model at each decoding step. This
document explains both the cache memory and the exact implementation flow.

## 1. What is stored in the cache?

At every decoder layer, attention creates:

```text
K = hidden_states @ WK
V = hidden_states @ WV
```

For a sequence of `T` tokens, the key and value tensors have shapes:

```text
K shape = [B, H, T, Dh]
V shape = [B, H, T, Dh]
```

The cache stores these K and V tensors for already processed tokens so the
model does not need to calculate them again when a new token is generated.

The cache does not store Q for previous tokens. During next-token generation,
the new token still needs a new query, while its key and value are appended to
the cache.

## 2. KV-cache shape for one layer

For one layer, the cache is often represented as:

```text
key_cache   = [B, H, T, Dh]
value_cache = [B, H, T, Dh]
```

Together:

```text
total elements per layer
= elements in K + elements in V
= B x H x T x Dh + B x H x T x Dh
= 2 x B x H x T x Dh
```

Because `H x Dh = D`:

```text
total elements per layer
= 2 x B x T x D
```

This is the generic formula for normal multi-head attention.

## 3. KV-cache size for all decoder layers

The model has `L` decoder layers, and every layer has its own K and V cache:

```text
total KV elements
= 2 x B x L x H x T x Dh
```

Using `D = H x Dh`, this can also be written as:

```text
total KV elements
= 2 x B x L x T x D
```

The number of bytes depends on the datatype:

```text
KV-cache bytes
= 2 x B x L x T x H x Dh x bytes_per_element
```

Common storage sizes are:

| Datatype | Bytes per element |
| --- | ---: |
| FP32 | 4 |
| BF16 | 2 |
| FP16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

The current reference configuration uses BF16 as its configured model dtype,
so the main examples below use 2 bytes per element.

## 4. Current reference model calculation

For the repository reference configuration with batch size `B = 1`:

```text
L = 61
H = 128
Dh = 56
D = 7168
```

### 4.1 One token

The cache added by one new token across all layers is:

```text
K and V elements per token
= 2 x B x L x H x Dh
= 2 x 1 x 61 x 128 x 56
= 874496 elements
```

At BF16 or FP16:

```text
874496 x 2 bytes
= 1748992 bytes per token
≈ 1.75 MB per token
```

So every additional generated token increases the cache by approximately
1.75 MB for one sequence in this reference configuration.

### 4.2 One layer at one token

For comparison, one layer contributes:

```text
2 x B x H x Dh
= 2 x 1 x 128 x 56
= 14336 elements
```

At BF16 or FP16:

```text
14336 x 2 bytes
= 28672 bytes
≈ 28.67 KB per layer per token
```

Multiplying by 61 layers gives the 1.75 MB per-token total.

## 5. Cache growth as context length increases

Because `T` is multiplied directly in the formula, KV-cache size grows
linearly with context length:

```text
KV bytes = 2 x B x L x H x T x Dh x bytes_per_element
```

For `B = 1` and BF16/FP16, the current reference model gives:

| Context `T` | KV elements | BF16/FP16 bytes | Approximate decimal size | Approximate binary size |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 874496 | 1748992 | 1.75 MB | 1.67 MiB |
| 1024 | 895483904 | 1790967808 | 1.79 GB | 1.67 GiB |
| 2048 | 1790967808 | 3581935616 | 3.58 GB | 3.34 GiB |
| 4096 | 3581935616 | 7163871232 | 7.16 GB | 6.67 GiB |
| 8192 | 7163871232 | 14327742464 | 14.33 GB | 13.35 GiB |
| 16384 | 14327742464 | 28655484928 | 28.66 GB | 26.69 GiB |
| 32768 | 28655484928 | 57310969856 | 57.31 GB | 53.38 GiB |
| 65536 | 57310969856 | 114621939712 | 114.62 GB | 106.76 GiB |
| 163840 | 143277424640 | 286554849280 | 286.55 GB | 266.88 GiB |

The maximum configured context is `T = 163840`. At that length, one sequence
requires approximately 286.55 GB of BF16 or FP16 KV-cache memory under normal
multi-head attention.

## 6. KV-cache growth chart

The following chart shows decimal GB for one sequence at BF16 or FP16:

```mermaid
xychart-beta
    title "Normal MHA KV-cache growth for the reference model"
    x-axis "Context tokens T" [1024, 2048, 4096, 8192, 16384, 32768, 65536, 163840]
    y-axis "Cache size in decimal GB" 0 --> 300
    line [1.79, 3.58, 7.16, 14.33, 28.66, 57.31, 114.62, 286.55]
```

The straight-line behavior is the important idea: doubling `T` doubles the
KV-cache memory.

## 7. Batch size multiplies the cache

All values above use `B = 1`. If the batch contains multiple independent
sequences, each sequence needs its own keys and values:

```text
KV bytes for batch B
= KV bytes for B=1 x B
```

For example, at `T = 4096` and BF16/FP16:

| Batch size | KV-cache size |
| ---: | ---: |
| 1 | 7.16 GB |
| 2 | 14.33 GB |
| 4 | 28.66 GB |
| 8 | 57.31 GB |
| 16 | 114.62 GB |

This is why long-context serving often uses a small batch size or dynamically
allocates cache space as sequences grow.

## 8. Datatype changes the memory requirement

For `B = 1` and `T = 163840`, the number of cache elements is fixed:

```text
total elements = 143277424640
```

Only the bytes per element change:

| Datatype | Bytes per element | Approximate cache size |
| --- | ---: | ---: |
| FP32 | 4 | 573.11 GB |
| BF16 | 2 | 286.55 GB |
| FP16 | 2 | 286.55 GB |
| INT8 | 1 | 143.28 GB |
| INT4 | 0.5 | 71.64 GB |

Quantization reduces storage, but it also requires a compatible cache
implementation and appropriate numerical handling. It is not automatically
provided by the current educational inference script.

## 9. What happens when one new token is generated?

Assume the model has already cached `T` tokens and receives one new token.
For each decoder layer:

```text
1. Compute Q for the new token.
2. Compute K for the new token.
3. Compute V for the new token.
4. Append the new K to key_cache.
5. Append the new V to value_cache.
6. Let the new query attend to all cached positions.
7. Produce the new hidden representation.
```

The cache changes from:

```text
K, V = [B, H, T, Dh]
```

to:

```text
K, V = [B, H, T + 1, Dh]
```

The new query itself has shape:

```text
Q_new = [B, H, 1, Dh]
```

It compares against all cached keys:

```text
Q_new             [B, H, 1, Dh]
cached K.T        [B, H, Dh, T + 1]
Q_new @ cached K.T [B, H, 1, T + 1]
```

Only one new row of attention scores is needed for the new token. The cache
avoids recomputing K and V for the previous `T` tokens.

## 10. Why normal multi-head attention is expensive

In normal multi-head attention, the cache stores one K vector and one V vector
for every head:

```text
K and V per token per layer = 2 x H x Dh
                             = 2 x D
```

For this reference model:

```text
2 x 7168 = 14336 elements per token per layer
```

Across 61 layers:

```text
14336 x 61 = 874496 elements per token
```

That is the source of the linear growth shown above. The cache size is not
determined only by the context window. It also depends on batch size, number
of layers, number of heads, head dimension, and datatype.

## 11. Generic calculator

For any normal multi-head-attention model, use this calculation:

```text
elements = 2 x B x L x H x T x Dh
bytes = elements x bytes_per_element
decimal_GB = bytes / 1000000000
binary_GiB = bytes / 1073741824
```

If the model configuration gives hidden size instead of head dimension, use:

```text
Dh = D / H
```

Then:

```text
elements = 2 x B x L x T x D
```

### Example: current reference model

```text
B = 1
L = 61
T = 4096
D = 7168
bytes_per_element = 2

elements = 2 x 1 x 61 x 4096 x 7168
         = 3581935616

bytes = 3581935616 x 2
      = 7163871232

decimal_GB = 7163871232 / 1000000000
           ≈ 7.16 GB
```

## 12. How the repository implementation works

The cache is deliberately implemented only for inference. Ordinary training
still calls the model without cache arguments and follows exactly the same
full-sequence path as before.

### 12.1 One cache per decoder layer

In [attention.py](../src/model/attention.py), one layer cache is represented as:

```python
KVCache = tuple[Tensor, Tensor]
```

The first tensor is K and the second is V. In
[deepseek.py](../src/model/deepseek.py), the whole-model cache is:

```python
ModelKVCache = tuple[KVCache, ...]
```

Its tuple length equals `num_hidden_layers`. Layer 0 never shares cached values
with layer 1 because every layer has different learned WK and WV projections.

### 12.2 Prompt prefill

At the first generation step, [inference.py](../inference.py) passes the full
prompt into the model:

```text
model input shape = [B, prompt_length]
past_key_values = None
use_cache = True
```

Every decoder layer computes Q, K, and V for the prompt. It returns its K and V
as `[B, H, prompt_length, Dh]`. The output object therefore contains one cache
pair per layer.

```mermaid
flowchart LR
    Prompt[Full prompt tokens] --> Prefill[One full model pass]
    Prefill --> Logits[Last prompt-token logits]
    Prefill --> L0[Layer 0 K and V]
    Prefill --> L1[Layer 1 K and V]
    Prefill --> LN[Layer L-1 K and V]
```

### 12.3 One-token decode

After sampling the first new token, the next call passes only that token plus
the previous cache:

```text
model input shape = [B, 1]
past key/value shape per layer = [B, H, T, Dh]
new key/value shape per layer = [B, H, 1, Dh]
updated key/value shape = [B, H, T + 1, Dh]
```

Inside each attention layer:

```text
K_all = concatenate(K_past, K_new, sequence dimension)
V_all = concatenate(V_past, V_new, sequence dimension)
scores = Q_new x transpose(K_all) / sqrt(Dh)
output = softmax(scores) x V_all
```

The model returns the updated cache, and inference feeds it into the following
step. The old tokens do not pass through WK or WV again.

The inference command also measures the returned tensors after every step:

```text
cache elements = sum(K.numel + V.numel for every layer)
cache bytes = sum(K.numel x K.element_size + V.numel x V.element_size)
cache MiB = cache bytes / 1048576
```

It prints the generation-step number, tokens cached per layer, total elements,
bytes, and MiB. These are runtime measurements of the real cache dtype and
shapes, not an estimate from the configuration.

```mermaid
flowchart LR
    New[Newest token only] --> QKV[Compute Q new, K new, V new]
    Old[Cached K and V for T tokens] --> Append[Append K new and V new]
    QKV --> Append
    QKV --> Scores[Q new attends to all T+1 keys]
    Append --> Scores
    Scores --> Next[Next-token logits and updated cache]
```

### 12.4 Why the causal-mask diagonal is offset

Without a cache, a sequence of length `Q` uses a square lower-triangular mask:

```text
query 0 sees key 0
query 1 sees keys 0..1
query 2 sees keys 0..2
```

With `T_past` cached tokens, current query 0 is actually absolute position
`T_past`. It must see all cached keys and its own new key. Therefore the code
constructs `[query_length, key_length]` and uses:

```python
visible = torch.ones(query_length, key_length).tril(diagonal=past_length)
```

For a one-token decode with five cached tokens, the mask is:

```text
query shape = [1]
key shape   = [6]
visible     = [True, True, True, True, True, True]
```

Using an ordinary `tril()` with diagonal zero here would expose only the first
key and silently produce incorrect logits.

### 12.5 Why positional embeddings also need an offset

The newest token is not position zero. If the cache already holds `T` tokens,
the next position is `T`. The positional embedding call therefore uses:

```text
start_position = cached sequence length
positions = [T, T + 1, ..., T + current_length - 1]
```

Restarting at zero for every one-token call would give every generated token
the same positional vector. The cache could have correct K/V shapes while the
model still produced different logits from full-sequence inference.

### 12.6 Reaching the configured context limit

This educational model uses learned absolute positions and has a finite
`max_position_embeddings`. Once the cache reaches that length, inference:

```text
1. Drops the existing cache.
2. Keeps only the newest max_position_embeddings token IDs.
3. Recomputes that window as a new prefill with positions starting at zero.
4. Continues one-token cached decoding.
```

Rebuilding at the boundary costs one full-window forward pass, but it keeps the
implementation correct and prevents the cache from growing beyond the
configured position table.

### 12.7 What the correctness tests prove

[test_attention.py](../tests/test_attention.py) checks two equivalences:

```text
cached causal attention output == full causal attention output
cached token-by-token model logits == full-sequence model logits
```

The first catches K/V concatenation and causal-mask errors. The second also
catches incorrect layer routing and positional offsets.

## 13. Practical conclusion

For the current DeepSeek reference dimensions and normal multi-head attention:

```text
each new token adds ≈1.75 MB of BF16/FP16 KV cache for B = 1
```

Therefore:

```text
4096 tokens  -> ≈7.16 GB
163840 tokens -> ≈286.55 GB
```

The current repository uses a small configuration for local execution, so its
real cache requirements are much smaller. The large numbers above are a direct
consequence of applying the reference dimensions to normal multi-head
attention. KV caching reduces repeated computation; it does not make the cache
memory free, and that memory still grows linearly until the context limit.
