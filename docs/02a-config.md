# Model configuration

The model dimensions are defined in one checked-in file:

```text
configs/deepseek_v3.json
```

The Python model code reads that file through:

```text
src/model/config.py
```

This prevents dimensions from being duplicated across attention, embedding,
feed-forward, and decoder-layer implementations.

## Configuration values

The default numeric values follow the published DeepSeek-V3 configuration:

- vocabulary size: `129280`
- context length: `163840`
- hidden size: `7168`
- decoder layers: `61`
- attention heads: `128`
- key/value heads: `128`
- dimension per head in this implementation: `56`
- dense feed-forward size: `18432`
- activation: `silu`
- normalization epsilon: `1e-6`
- parameter dtype: `bfloat16`

The repository's current implementation profile uses the ordinary attention and
positional-embedding components requested for this educational build. The
configuration includes only dimensions and options that belong to the current
decoder and attention foundation. Additional architecture components will be
added to the configuration in the same commit as their implementation.

## Recreate this step

Create `configs/deepseek_v3.json` with the exact JSON values in the file at the
repository root. Then create `src/model/config.py` with the typed dataclass and
JSON loader present in the repository. Finally, export `ModelConfig`,
`default_config_path`, and `load_default_config` from `src/model/__init__.py`.

The intended usage from later model code is:

```python
from src.model.config import load_default_config

config = load_default_config()
attention = CausalSelfAttention(
    embed_dim=config.hidden_size,
    num_heads=config.num_attention_heads,
    dropout=config.attention_dropout,
    bias=config.attention_bias,
)
```

## Historical commit

Suggested commit:

```text
feat(config): add centralized DeepSeek model configuration
```

Source references:

- [DeepSeek-V3 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)
