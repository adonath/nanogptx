"""Here are some thoughts on how to structure the code"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT configuration"""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout_rate: float = 0.0
    use_bias: bool = True
    seed: int = 0
    device: str = "cpu"
