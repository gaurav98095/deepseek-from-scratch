# Step 02c — Run the complete training and inference pipeline

This step connects the prepared TinyStories JSONL file to the current
attention-based decoder and optimizer. It intentionally uses a small byte
tokenizer and a small model configuration so the pipeline can run locally.

The flow is:

```text
TinyStories JSONL
    → UTF-8 byte tokenizer
    → fixed-length token sequences
    → token and position embeddings
    → causal attention decoder blocks
    → vocabulary logits
    → shifted cross-entropy loss
    → AdamW, warmup/cosine schedule, gradient clipping
    → checkpoint
```

Inference uses the same model weights but enables a per-layer KV cache. The
prompt is processed once, and later decoding calls normally process only the
newest token.

## Run it

First prepare the dataset using [01-data-preparation.md](./01-data-preparation.md).
Then install dependencies and run:

```bash
python -m pip install -r requirements.txt
python -m src.train \
  --data data/processed/tinystories-train.jsonl \
  --config configs/tiny_train.json \
  --steps 1000 \
  --batch-size 8 \
  --block-size 128 \
  --device auto \
  --tensorboard-logdir runs/tensorboard \
  --checkpoint checkpoints/deepseek-tiny.pt
```

In another terminal, start TensorBoard:

```bash
tensorboard --logdir runs/tensorboard
```

The run records training loss and perplexity, optimizer learning rate and
gradient norm, batch/sequence sizes, step time, tokens per second, total token
count, logits mean and standard deviation, parameter counts, and CUDA memory
metrics when training on CUDA.

For a short smoke run:

```bash
python -m src.train \
  --data data/processed/tinystories-smoke.jsonl \
  --config configs/tiny_train.json \
  --steps 10 \
  --warmup-steps 0 \
  --batch-size 2 \
  --block-size 128 \
  --device cpu
```

The command prints periodic loss values and saves model, optimizer, scheduler,
and step state to the checkpoint path.

## Inference from a checkpoint

Inference happens only after a checkpoint has been created by training. The
checkpoint is a local generated artifact and is not downloaded from GitHub or
stored in the repository.

Run the development pipeline first:

```bash
make step-2-dev
```

This creates:

```text
checkpoints/deepseek-tiny.pt
```

The root-level `inference.py` script loads the newest `.pt` file in the
`checkpoints/` directory when no explicit checkpoint is supplied. It rebuilds
the model from the matching JSON configuration, loads the saved model weights,
encodes the prompt with the byte tokenizer, builds a KV cache during prompt
prefill, and then generates one token at a time without recomputing old K/V
projections.

### `inference.py`

Create this file at the repository root, outside `src/`:

```python
"""Generate text from the newest or explicitly selected model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.byte_tokenizer import ByteTokenizer
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek, ModelKVCache


def _build_parser() -> argparse.ArgumentParser:
    # Keep inference independent from the training loop so a saved model can be
    # used without recreating a dataset or optimizer.
    parser = argparse.ArgumentParser(description=__doc__)
    # The architecture must match the configuration used to create the checkpoint.
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    # If omitted, the newest checkpoint in this directory is selected by mtime.
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    # The prompt is converted to byte IDs by the current educational tokenizer.
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    # This controls how many new tokens are appended after the prompt.
    parser.add_argument("--max-new-tokens", type=int, default=50)
    # Temperature 0 uses greedy decoding; higher values produce more variation.
    parser.add_argument("--temperature", type=float, default=1.0)
    # Top-k limits sampling to the most likely candidates when it is positive.
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _choose_device(requested: str) -> torch.device:
    # Fail clearly instead of silently moving a requested CUDA run to the CPU.
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _latest_checkpoint(directory: Path) -> Path:
    # Modification time makes the default behavior select the most recent save.
    checkpoints = list(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints found in {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> DeepSeek:
    # Construct the same module layout before loading the saved parameter tensors.
    config = ModelConfig.from_json(config_path)
    model = DeepSeek(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Training checkpoints contain optimizer state as well as the model weights.
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def generate(
    model: DeepSeek,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Append tokens to ``prompt`` and return the complete decoded text."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    device = next(model.parameters()).device
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    past_key_values: ModelKVCache | None = None

    for generation_step in range(1, max_new_tokens + 1):
        # The first call processes the complete prompt and builds every layer's
        # cache. Later calls normally process only the newest generated token.
        cached_length = 0
        if past_key_values is not None:
            cached_length = past_key_values[0][0].shape[2]

        if past_key_values is None or cached_length >= model.config.max_position_embeddings:
            # At the context limit, discard the old cache and rebuild it from the
            # newest window. This also restarts learned positions at zero.
            model_input = token_ids[:, -model.config.max_position_embeddings :]
            past_key_values = None
        else:
            model_input = token_ids[:, -1:]

        output = model(
            model_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("the model did not return a KV cache")
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        # Measure the actual allocated K/V tensors rather than estimating from
        # configuration values. This includes every decoder layer's K and V.
        cache_elements = sum(
            key_cache.numel() + value_cache.numel()
            for key_cache, value_cache in past_key_values
        )
        cache_bytes = sum(
            key_cache.numel() * key_cache.element_size()
            + value_cache.numel() * value_cache.element_size()
            for key_cache, value_cache in past_key_values
        )
        cached_tokens_per_layer = past_key_values[0][0].shape[2]
        print(
            f"generation_step={generation_step} "
            f"kv_cache_tokens_per_layer={cached_tokens_per_layer} "
            f"kv_cache_elements={cache_elements} "
            f"kv_cache_bytes={cache_bytes} "
            f"kv_cache_mib={cache_bytes / 2**20:.4f}"
        )

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                # Suppress all but the top-k scores before sampling.
                limit = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, limit)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        token_ids = torch.cat((token_ids, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids[0].tolist())


def main() -> None:
    # Parse options once so command-line use and future integrations share behavior.
    args = _build_parser().parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)
    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    model = _load_model(args.config, checkpoint, device)
    text = generate(
        model,
        ByteTokenizer(),
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    print(f"Checkpoint: {checkpoint}")
    print(text)


if __name__ == "__main__":
    main()
```

Run inference through the Makefile:

```bash
make infer INFERENCE_PROMPT="Once upon a time"
```

For a specific checkpoint and longer generation:

```bash
make infer \
  INFERENCE_CONFIG=configs/tiny_train.json \
  CHECKPOINT=checkpoints/deepseek-tiny.pt \
  INFERENCE_PROMPT="The little girl went" \
  INFERENCE_TOKENS=100 \
  INFERENCE_TEMPERATURE=0.7
```

The checkpoint must be loaded with the same architecture configuration used
during training. If the checkpoint is deleted, run `make step-2-dev` again.

## Exact files in this step

### `configs/tiny_train.json`

Create this file exactly:

```json
{
  "model_name": "DeepSeek-Tiny-CurrentScaffold",
  "model_type": "decoder_only",
  "implementation_profile": "standard_attention_educational",
  "vocab_size": 258,
  "max_position_embeddings": 129,
  "hidden_size": 256,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "num_key_value_heads": 8,
  "head_dim": 32,
  "intermediate_size": 1024,
  "hidden_activation": "silu",
  "attention_dropout": 0.0,
  "attention_bias": false,
  "position_embedding_type": "learned_absolute",
  "initializer_range": 0.02,
  "rms_norm_eps": 0.000001,
  "torch_dtype": "float32",
  "tie_word_embeddings": false
}
```

### `src/data/byte_tokenizer.py`

This first tokenizer maps UTF-8 bytes directly to token IDs. It has 256 byte
tokens plus beginning-of-sequence ID 256 and end-of-sequence ID 257.

```python
"""A dependency-free byte tokenizer for the first runnable training pipeline."""

from __future__ import annotations


class ByteTokenizer:
    """Encode UTF-8 bytes plus BOS/EOS tokens for a small demo vocabulary."""

    # IDs 0-255 are reserved for raw UTF-8 byte values.
    bos_token_id = 256
    eos_token_id = 257
    # Two special tokens are added to the 256 possible byte values.
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        # UTF-8 makes this tokenizer work for stories containing non-ASCII text.
        tokens = list(text.encode("utf-8"))
        if add_bos:
            # BOS tells the model where a story starts.
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            # EOS tells the model where a story ends.
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        # Special tokens are ignored because only byte IDs can form text.
        byte_tokens = [token for token in tokens if 0 <= token < 256]
        # Replacement decoding keeps generation readable if bytes are incomplete.
        return bytes(byte_tokens).decode("utf-8", errors="replace")
```

### `src/data/tinystories.py`

This dataset iterator reads prepared JSONL records, tokenizes each story, and
returns only complete sequences of `block_size + 1` tokens. The extra token is
needed because the model shifts inputs and labels by one position.

```python
"""Streaming TinyStories records into fixed-length language-model batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .byte_tokenizer import ByteTokenizer


class TinyStoriesDataset(IterableDataset[Tensor]):
    """Yield fixed-length token sequences from prepared TinyStories JSONL."""

    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, block_size: int) -> None:
        super().__init__()
        # At least two positions are needed to create shifted input/target pairs.
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self) -> Iterator[Tensor]:
        # Streaming the file avoids loading the complete prepared dataset into RAM.
        with self.path.open(encoding="utf-8") as data_file:
            for line in data_file:
                # Each line is one normalized story written by the preparation step.
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str):
                    continue
                # Tokenization happens lazily so training can start from large files.
                tokens = self.tokenizer.encode(text)
                # Move by block_size so each yielded sample is a non-overlapping chunk.
                for start in range(0, len(tokens) - self.block_size, self.block_size):
                    # The extra token becomes the final next-token target.
                    sequence = tokens[start : start + self.block_size + 1]
                    if len(sequence) == self.block_size + 1:
                        # Long tensors are required by nn.Embedding input IDs.
                        yield torch.tensor(sequence, dtype=torch.long)
```

### `src/model/deepseek.py`

The model file follows the forward pass from top to bottom: one decoder block,
the output container, the complete model, and one compact training-update
helper. `DeepSeek` and `DeepSeekTrainer` both require explicit configuration,
so a caller can never silently instantiate the enormous reference model.

Create `src/model/deepseek.py` with exactly this content:

```python
"""The current educational DeepSeek decoder and its training utilities.

The file is intentionally organized in the same order as the forward pass:

1. One decoder block.
2. The model's output container.
3. The complete decoder-only model.
4. The optimizer and one-step training helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import KVCache, CausalSelfAttention, LearnedPositionalEmbedding
from .config import ModelConfig

# The model keeps one independent (K, V) pair for every decoder layer.
ModelKVCache = tuple[KVCache, ...]


class DecoderBlock(nn.Module):
    """One pre-normalized attention-plus-MLP decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # D is reused by normalization, attention, and both MLP projections.
        hidden_size = config.hidden_size

        # This is a pre-norm decoder: normalize before each transformation, then
        # add that transformation's output back to the residual stream.
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)

        # Causal attention mixes information only from the current and earlier tokens.
        self.attention = CausalSelfAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=config.attention_bias,
        )

        # The second normalization prepares attention output for the token-wise MLP.
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)

        # The MLP expands D -> intermediate_size, applies SiLU independently at
        # every token position, and contracts intermediate_size -> D.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size, bias=config.attention_bias),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, hidden_size, bias=config.attention_bias),
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: KVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """Apply attention, then the feed-forward network, with residuals."""

        # First sublayer: x = x + Attention(LayerNorm(x)).
        attention_input = self.input_layernorm(hidden_states)
        attention_result = self.attention(
            attention_input,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            # `use_cache=True` makes attention return both output and updated K/V.
            attention_output, present_key_value = cast(
                tuple[Tensor, KVCache],
                attention_result,
            )
        else:
            attention_output = cast(Tensor, attention_result)
            present_key_value = None
        hidden_states = hidden_states + attention_output

        # Second sublayer: x = x + MLP(LayerNorm(x)). Both residual additions
        # preserve the [B, T, D] shape expected by the next decoder block.
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)

        # Training keeps the original Tensor-only path. Incremental inference
        # additionally carries this layer's cache to the next generated token.
        if use_cache:
            return hidden_states, cast(KVCache, present_key_value)
        return hidden_states


@dataclass
class DeepSeekOutput:
    """The logits and optional next-token training loss returned by the model."""

    # Every token position has one unnormalized score per vocabulary item.
    logits: Tensor

    # Inference leaves loss as None; training supplies labels to compute it.
    loss: Tensor | None = None

    # Incremental inference receives one updated K/V pair per decoder layer.
    past_key_values: ModelKVCache | None = None


class DeepSeek(nn.Module):
    """Decoder-only language model assembled from :class:`ModelConfig`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Requiring an explicit config prevents accidental construction of the
        # full reference-size model during a small local experiment.
        self.config = config

        # Token lookup: integer IDs [B, T] -> learned vectors [B, T, D].
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        # Position lookup: positions [T] -> vectors [1, T, D], which broadcast
        # across B when added to token embeddings.
        self.position_embeddings = LearnedPositionalEmbedding(
            max_sequence_length=self.config.max_position_embeddings,
            embed_dim=self.config.hidden_size,
        )

        # Build exactly the configured number of independent decoder blocks.
        # Keep the name `layers` so existing checkpoint parameter keys remain loadable.
        self.layers = nn.ModuleList(
            DecoderBlock(self.config) for _ in range(self.config.num_hidden_layers)
        )

        # Normalize the completed residual stream before predicting tokens.
        self.final_layernorm = nn.LayerNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # Vocabulary projection: [B, T, D] -> [B, T, vocab_size].
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Weight tying makes input embedding rows and output classifier rows share
        # the same Parameter object when the configuration requests it.
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize every nested Linear and Embedding module after construction.
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        """Initialize learned linear and embedding weights."""

        # LayerNorm keeps PyTorch's defaults; learned matrix-like modules use the
        # normal distribution and standard deviation recorded in the model config.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

            # Embeddings have no bias. Linear biases, when enabled, start at zero.
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> DeepSeekOutput:
        """Run embeddings, all decoder blocks, and the vocabulary head."""

        # The model accepts a rectangular batch of integer token IDs: [B, T].
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")

        # Empty sequences cannot produce attention scores or next-token targets.
        if input_ids.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")

        # Learned position embeddings have only max_position_embeddings rows.
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        # Each input position needs a corresponding next-token training label.
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        # KV caching is an inference optimization. Keeping it out of the labeled
        # training path avoids retaining old computation graphs across batches.
        if labels is not None and (past_key_values is not None or use_cache):
            raise ValueError("KV caching is supported only when labels are omitted")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one cache per decoder layer")

        # Every layer must describe the same already-processed prefix length.
        past_length = 0
        if past_key_values is not None:
            cache_lengths: set[int] = set()
            for key_cache, value_cache in past_key_values:
                if key_cache.ndim != 4 or value_cache.ndim != 4:
                    raise ValueError("each layer cache must have shape [B, H, T, Dh]")
                cache_lengths.add(key_cache.shape[2])
                cache_lengths.add(value_cache.shape[2])
            if len(cache_lengths) != 1:
                raise ValueError("all key/value caches must have the same sequence length")
            past_length = cache_lengths.pop()

        # The new tokens continue after the cached prefix in the learned position table.
        sequence_length = input_ids.shape[1]
        total_sequence_length = past_length + sequence_length
        if total_sequence_length > self.config.max_position_embeddings:
            raise ValueError("cached and current tokens exceed max_position_embeddings")

        # Convert IDs and positions into one initial hidden vector per token.
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(
            sequence_length,
            start_position=past_length,
            device=input_ids.device,
        )

        # Every block consumes and returns [B, T, D], so the same variable can
        # carry the residual stream through the complete decoder stack.
        present_key_values: list[KVCache] = []
        for layer_index, block in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            block_result = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, layer_cache = cast(tuple[Tensor, KVCache], block_result)
                present_key_values.append(layer_cache)
            else:
                hidden_states = cast(Tensor, block_result)

        # Produce unnormalized next-token scores for every position.
        logits = self.lm_head(self.final_layernorm(hidden_states))
        loss = None
        if labels is not None:
            # Position t predicts label t+1. The final logit has no later target,
            # and the first label has no earlier prediction, so both are omitted.
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )

        # A structured return keeps inference and training on the same model API.
        return DeepSeekOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(present_key_values) if use_cache else None,
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for one optimizer and scheduler."""

    # AdamW optimizer hyperparameters.
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

    # The schedule warms up linearly, then decays with a cosine to max_steps.
    warmup_steps: int = 100
    max_steps: int = 1000

    # Global gradient clipping protects an update from an unusually large norm.
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        """Reject settings that would make the training schedule invalid."""

        # Validate once during construction so the hot training path stays simple.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not all(0 < beta < 1 for beta in self.betas):
            raise ValueError("betas must be between zero and one")
        if self.warmup_steps < 0 or self.max_steps <= self.warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


class DeepSeekTrainer:
    """Own the optimizer and perform one complete training update."""

    def __init__(self, model: DeepSeek, config: TrainingConfig) -> None:
        # The trainer owns optimization state but does not own data iteration,
        # logging, or checkpoint policy; those remain visible in `src/train.py`.
        self.model = model
        self.config = config

        # These values are updated after each successful optimizer step.
        self.global_step = 0
        self.last_grad_norm = 0.0

        # AdamW applies decoupled weight decay while maintaining first- and
        # second-moment estimates for every trainable parameter.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        # LambdaLR multiplies the base learning rate by the value returned below.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._learning_rate_multiplier,
        )

    def _learning_rate_multiplier(self, step: int) -> float:
        """Return linear warmup followed by cosine decay."""

        if step < self.config.warmup_steps:
            # Increase from roughly 1/warmup_steps to the full base learning rate.
            return max(step, 1) / max(self.config.warmup_steps, 1)

        # Map post-warmup steps to progress in [0, 1], then cosine-decay the
        # multiplier from 1 to 0 by the final configured optimizer step.
        progress = (step - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def train_batch(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> DeepSeekOutput:
        """Perform one forward, backward, optimizer, and scheduler update."""

        # Enable training behavior such as attention-probability dropout.
        self.model.train()

        # Remove gradients from the previous update. Setting them to None avoids
        # unnecessary zero-filled allocations and is supported by AdamW.
        self.optimizer.zero_grad(set_to_none=True)

        # When separate labels are absent, use input_ids for next-token training;
        # DeepSeek.forward performs the one-position shift internally.
        output = self.model(
            input_ids,
            labels=input_ids if labels is None else labels,
            attention_mask=attention_mask,
        )
        if output.loss is None:
            raise RuntimeError("training requires a language-model loss")

        # Backpropagation fills parameter gradients from the scalar LM loss.
        output.loss.backward()

        # Clip the combined gradient norm before AdamW reads the gradients. The
        # returned pre-clipping norm is saved for TensorBoard diagnostics.
        self.last_grad_norm = float(
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        )

        # Update parameters first, then advance the learning-rate schedule and
        # the public count of completed training updates.
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return output
```

The trainer intentionally contains only the optimizer, scheduler, and one
training update. Dataset iteration, metrics, TensorBoard, and checkpoint
writing stay in the command-line training file where their execution order is
easy to follow.

### `src/train.py`

The command-line entry point selects a device, loads an explicit configuration,
builds the model and trainer, records TensorBoard metrics, loops until
`--steps`, and directly writes the final checkpoint.

Create `src/train.py` with exactly this content:

```python
"""Train the current educational decoder on prepared TinyStories data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.byte_tokenizer import ByteTokenizer
from .data.tinystories import TinyStoriesDataset
from .model.config import ModelConfig
from .model.deepseek import DeepSeek, DeepSeekTrainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for one training run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/tinystories-train.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_train.json"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepseek-tiny.pt"))
    parser.add_argument("--tensorboard-logdir", type=Path, default=Path("runs/tensorboard"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def choose_device(requested: str) -> torch.device:
    """Choose CUDA when requested or available, otherwise use CPU."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(args: argparse.Namespace) -> None:
    """Build the data/model pipeline and train until ``args.steps``."""

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.log_interval < 1:
        raise ValueError("log_interval must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = ModelConfig.from_json(args.config)

    if args.block_size + 1 > config.max_position_embeddings:
        raise ValueError("block_size + 1 must not exceed max_position_embeddings in the config")

    tokenizer = ByteTokenizer()
    if config.vocab_size < tokenizer.vocab_size:
        raise ValueError(
            f"config vocab_size ({config.vocab_size}) must be at least the "
            f"tokenizer vocab_size ({tokenizer.vocab_size})"
        )

    model = DeepSeek(config).to(device)
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = min(100, args.steps // 10)
    trainer = DeepSeekTrainer(
        model,
        TrainingConfig(
            learning_rate=args.learning_rate,
            warmup_steps=warmup_steps,
            max_steps=args.steps,
        ),
    )

    dataset = TinyStoriesDataset(args.data, tokenizer, args.block_size)
    batches = DataLoader(dataset, batch_size=args.batch_size)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir))
    writer.add_text("run/model_config", json.dumps(config.to_dict(), indent=2), 0)
    writer.add_text("run/arguments", str(vars(args)), 0)
    writer.add_scalar("model/parameters_total", sum(p.numel() for p in model.parameters()), 0)
    writer.add_scalar(
        "model/parameters_trainable",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        0,
    )

    try:
        while trainer.global_step < args.steps:
            made_progress = False

            for input_ids in batches:
                made_progress = True
                started = time.perf_counter()
                input_ids = input_ids.to(device)
                output = trainer.train_batch(input_ids)
                seconds = time.perf_counter() - started
                step = trainer.global_step
                loss = output.loss.item()
                tokens = input_ids.numel()
                tokens_per_second = tokens / seconds

                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/perplexity", math.exp(min(loss, 20.0)), step)
                writer.add_scalar("optimizer/learning_rate", trainer.optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("optimizer/gradient_norm", trainer.last_grad_norm, step)
                writer.add_scalar("data/batch_size", input_ids.shape[0], step)
                writer.add_scalar("data/sequence_length", input_ids.shape[1], step)
                writer.add_scalar("performance/step_seconds", seconds, step)
                writer.add_scalar("performance/tokens_per_second", tokens_per_second, step)
                writer.add_scalar("performance/total_tokens", step * tokens, step)
                writer.add_scalar("model/logits_mean", output.logits.detach().mean().item(), step)
                writer.add_scalar("model/logits_std", output.logits.detach().std().item(), step)

                if device.type == "cuda":
                    writer.add_scalar(
                        "system/cuda_memory_allocated_mb",
                        torch.cuda.memory_allocated() / 2**20,
                        step,
                    )
                    writer.add_scalar(
                        "system/cuda_memory_reserved_mb",
                        torch.cuda.memory_reserved() / 2**20,
                        step,
                    )
                    writer.add_scalar(
                        "system/cuda_max_memory_allocated_mb",
                        torch.cuda.max_memory_allocated() / 2**20,
                        step,
                    )

                if step % args.log_interval == 0:
                    print(f"step={step} loss={loss:.4f} tokens_per_second={tokens_per_second:.1f}")
                    writer.flush()
                if trainer.global_step >= args.steps:
                    break

            if not made_progress:
                raise RuntimeError("the dataset produced no complete training sequences")

        writer.add_text("run/status", "completed", trainer.global_step)
        writer.flush()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "global_step": trainer.global_step,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint to {args.checkpoint}")
    finally:
        writer.close()


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
```

## What is and is not included

Included in this runnable stage:

- TinyStories JSONL reading
- UTF-8 byte tokenization
- Fixed-length batches
- Token and positional embeddings
- Repeated causal attention decoder blocks
- Feed-forward layers and residual connections
- Vocabulary logits and shifted language-model loss
- AdamW, warmup/cosine scheduling, gradient clipping
- Final checkpoint saving with model, optimizer, scheduler, and step state
- Model-weight loading for inference
- Per-layer KV-cache reuse during autoregressive inference
- TensorBoard metric logging

The large `configs/deepseek_v3.json` remains a dimension reference. Use
`configs/tiny_train.json` for local execution until the later scaling and data
engineering stages are implemented.

## Commit this step

```text
feat(train): add runnable attention-based training pipeline
```

```bash
git add configs/tiny_train.json src/data/byte_tokenizer.py src/data/tinystories.py src/model/deepseek.py src/train.py inference.py src/data/__init__.py README.md docs/02c-training.md
git commit -m "feat(train): add runnable attention-based training pipeline"
```
