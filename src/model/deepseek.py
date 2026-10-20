"""Complete decoder-only model assembled from the current building blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import CausalSelfAttention, LearnedPositionalEmbedding
from .config import ModelConfig, load_default_config


class DecoderBlock(nn.Module):
    """Pre-normalized decoder block with causal attention and an MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Pre-normalization stabilizes the hidden-state scale before attention.
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Each block uses the causal attention primitive built in the previous step.
        self.attention = CausalSelfAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=config.attention_bias,
        )
        # A second normalization prepares the attention output for the feed-forward path.
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The MLP expands features, applies the configured activation, then projects back.
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias),
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Residual addition preserves the input signal while attention adds context.
        hidden_states = hidden_states + self.attention(
            self.input_layernorm(hidden_states), attention_mask=attention_mask
        )
        # The second residual path adds token-wise nonlinear processing.
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


@dataclass
class DeepSeekOutput:
    """Outputs returned by :class:`DeepSeek`."""

    # Logits provide one vocabulary score vector for every input position.
    logits: Tensor
    # Loss is populated only when labels are passed during training.
    loss: Tensor | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and loop settings for pretraining."""

    # AdamW peak learning rate after warmup.
    learning_rate: float = 3e-4
    # These values control the optimizer's moving averages.
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    # Warmup prevents large updates before the model has useful gradients.
    warmup_steps: int = 100
    # The scheduler needs a final step to define its decay range.
    max_steps: int = 1000
    # Clipping limits a single batch's influence on the parameters.
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        # Validate training hyperparameters before constructing optimizer state.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.betas[0] < 1 or not 0 < self.betas[1] < 1:
            raise ValueError("betas must be between zero and one")
        if self.warmup_steps < 0 or self.max_steps <= self.warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


class DeepSeekTrainer:
    """Own the optimizer and one complete language-model training loop."""

    def __init__(self, model: DeepSeek, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        # AdamW decouples weight decay from the gradient update.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        # LambdaLR applies warmup and cosine decay after each optimizer step.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_multiplier)
        # This counter determines when the loop and checkpoint are complete.
        self.global_step = 0

    def _lr_multiplier(self, step: int) -> float:
        # Increase linearly during warmup.
        if step < self.config.warmup_steps:
            return max(step, 1) / max(self.config.warmup_steps, 1)
        # After warmup, decay smoothly toward zero with a cosine schedule.
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
        """Run forward, loss, backward, clipping, optimizer, and scheduler steps."""

        # Training mode enables training-time module behavior such as dropout.
        self.model.train()
        # Clear old gradients because PyTorch accumulates them by default.
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            input_ids,
            labels=labels if labels is not None else input_ids,
            attention_mask=attention_mask,
        )
        if output.loss is None:
            raise RuntimeError("training requires a language-model loss")
        # Backpropagate the language-model loss through the full decoder.
        output.loss.backward()
        # Clip before stepping so extreme gradients cannot destabilize the update.
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        # Apply parameters first, then advance the learning-rate schedule.
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return output

    def train_epoch(self, batches: Iterable[Tensor | tuple[Tensor, Tensor]]) -> float:
        """Train over batches and return the mean loss."""

        # Keep scalar copies so the returned average does not retain computation graphs.
        losses: list[float] = []
        for batch in batches:
            if isinstance(batch, tuple):
                input_ids, labels = batch
            else:
                input_ids, labels = batch, None
            output = self.train_batch(input_ids, labels=labels)
            losses.append(float(output.loss.detach()))
        if not losses:
            raise ValueError("batches must contain at least one batch")
        return sum(losses) / len(losses)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, scheduler, and progress state."""

        # Save every state needed to resume optimization, not only model weights.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, and progress state."""

        # Load tensors on CPU so restoration works even if the original device differs.
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint["global_step"])


class DeepSeek(nn.Module):
    """Decoder-only language model assembled from the repository config.

    Pass a smaller ``ModelConfig`` while developing locally. Calling the class
    without a config uses the full checked-in DeepSeek-V3-scale dimensions.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        # A supplied small config makes local experiments practical; None loads the reference config.
        self.config = config or load_default_config()
        # Token embeddings convert integer IDs into model-width vectors.
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # Absolute positions are added once before the first decoder block.
        self.position_embeddings = LearnedPositionalEmbedding(
            max_sequence_length=self.config.max_position_embeddings,
            embed_dim=self.config.hidden_size,
        )
        # ModuleList registers every repeated block so its parameters are trained and saved.
        self.layers = nn.ModuleList(DecoderBlock(self.config) for _ in range(self.config.num_hidden_layers))
        # Final normalization prepares the hidden states for vocabulary scoring.
        self.final_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        # The language-model head maps each hidden vector back to vocabulary logits.
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            # Weight tying reduces parameters by sharing input and output embeddings.
            self.lm_head.weight = self.token_embeddings.weight

        # Apply the configured initialization to every linear and embedding layer.
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        # Small normal initialization is applied consistently across the model.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> DeepSeekOutput:
        # Integer token IDs must be a two-dimensional batch/sequence matrix.
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        # Convert IDs to vectors and add a position vector shared across the batch.
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(
            input_ids.shape[1], device=input_ids.device
        )
        # Every block updates the sequence using context and a feed-forward path.
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        # Produce one vocabulary distribution per input position.
        logits = self.lm_head(self.final_layernorm(hidden_states))

        loss = None
        if labels is not None:
            # Shift by one so position t predicts the token at position t+1.
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )
        return DeepSeekOutput(logits=logits, loss=loss)
