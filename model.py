"""Here are some thoughts on how to structure the code"""

from dataclasses import dataclass

from utils import Config


@dataclass
class GPTConfig(Config):
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

    @property
    def n_embd_mlp(self) -> int:
        """Hidden embedding size for the MLP"""
        return 4 * self.n_embd

    @property
    def n_embd_attn(self) -> int:
        """Hidden embedding size for the attention"""
        return 3 * self.n_embd
