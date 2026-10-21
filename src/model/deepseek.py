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

from .attention import (
    CausalMultiHeadLatentAttention,
    LatentKVCache,
    LearnedPositionalEmbedding,
)
from .config import ModelConfig

# The model keeps one compressed KV latent per decoder layer.
ModelKVCache = tuple[LatentKVCache, ...]


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
        self.attention = CausalMultiHeadLatentAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            dropout=config.attention_dropout,
            bias=config.attention_bias,
            norm_eps=config.rms_norm_eps,
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
        past_key_value: LatentKVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, LatentKVCache]:
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
                tuple[Tensor, LatentKVCache],
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
            return hidden_states, cast(LatentKVCache, present_key_value)
        return hidden_states


@dataclass
class DeepSeekOutput:
    """The logits and optional next-token training loss returned by the model."""

    # Every token position has one unnormalized score per vocabulary item.
    logits: Tensor

    # Inference leaves loss as None; training supplies labels to compute it.
    loss: Tensor | None = None

    # Incremental inference receives one compressed KV latent per decoder layer.
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
            for latent_cache in past_key_values:
                if latent_cache.ndim != 3:
                    raise ValueError("each layer cache must have shape [B, T, kv_lora_rank]")
                if latent_cache.shape[0] != input_ids.shape[0]:
                    raise ValueError("cache batch size must match input_ids")
                if latent_cache.shape[2] != self.config.kv_lora_rank:
                    raise ValueError("cache latent rank must match kv_lora_rank")
                cache_lengths.add(latent_cache.shape[1])
            if len(cache_lengths) != 1:
                raise ValueError("all latent caches must have the same sequence length")
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
        present_key_values: list[LatentKVCache] = []
        for layer_index, block in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            block_result = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, layer_cache = cast(
                    tuple[Tensor, LatentKVCache],
                    block_result,
                )
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
