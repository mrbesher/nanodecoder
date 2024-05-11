from dataclasses import dataclass


@dataclass
class NanoDecoderConfig:
    hidden_size: int = 128
    intermediate_size: int = 512
    kv_lora_rank: int = 32
    max_position_embeddings: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    vocab_size: int = 64
    attention_dropout: float = 0.1
