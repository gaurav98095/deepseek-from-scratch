"""Configuration for the small DeepSeek-style model."""

from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    """Dimensions and routing settings shared by model components."""

    # Tokenizer and sequence length
    vocab_size: int = 8192
    max_seq_len: int = 1024

    # Model dimensions
    dim: int = 256
    n_layers: int = 6
    n_dense_layers: int = 1
    n_heads: int = 8

    # Feed-forward dimensions
    inter_dim: int = 768
    moe_inter_dim: int = 256

    # MLA dimensions
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32

    # MoE and routing configuration
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    route_scale: float = 1.0

    # Rotary position encoding
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if not 0 <= self.n_dense_layers <= self.n_layers:
            raise ValueError("n_dense_layers must be between 0 and n_layers")
        if self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be nonnegative")
        if not 1 <= self.n_activated_experts <= self.n_routed_experts:
            raise ValueError("invalid number of activated experts")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if self.q_lora_rank <= 0 or self.kv_lora_rank <= 0:
            raise ValueError("LoRA ranks must be positive")
        if self.n_expert_groups <= 0:
            raise ValueError("n_expert_groups must be positive")
        if self.n_routed_experts % self.n_expert_groups != 0:
            raise ValueError("experts must divide evenly across groups")
        if not 1 <= self.n_limited_groups <= self.n_expert_groups:
            raise ValueError("invalid number of limited groups")
        if self.route_scale <= 0:
            raise ValueError("route_scale must be positive")
