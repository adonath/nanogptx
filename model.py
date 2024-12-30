"""Here are some thoughts on how to structure the code"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

import jax
from jax import numpy as jnp

from utils import Config, register_dataclass_jax


class Axis(int, Enum):
    """Axis order"""

    batch = 0
    sequence = 1
    feature = 2


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
    _key = None

    @property
    def n_embd_mlp(self) -> int:
        """Hidden embedding size for the MLP"""
        return 4 * self.n_embd

    @property
    def n_embd_attn(self) -> int:
        """Hidden embedding size for the attention"""
        return 3 * self.n_embd

    def generate_key(self) -> jax.Array:
        """Generate random key for initialization"""
        if self._key is None:
            self._key = jax.random.PRNGKey(self.seed)

        self._key, subkey = jax.random.split(self._key)
        return subkey


@register_dataclass_jax(data_fields=["weight"])
@dataclass(frozen=True)
class Embedding:
    """Embedding layer"""

    weight: jax.Array

    @classmethod
    def from_n_features(cls, vocab_size: int, n_embd: int, key: jax.Array):
        """Create an embedding layer from number of features"""
        return cls(weight=jax.random.normal(key, (vocab_size, n_embd)))

    @classmethod
    def from_config(cls, config):
        """Create an embedding layer from configuration"""
        return cls.from_n_features(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            key=config.generate_key(),
        )

    def __call__(self, x):
        return jnp.take(self.weight, x, axis=0)


@register_dataclass_jax(data_fields=["weight", "bias"])
@dataclass(frozen=True)
class LayerNorm:
    """Layer normalization"""

    weight: jax.Array
    bias: Optional[jax.Array] = None
    eps: ClassVar[float] = 1e-5

    def __call__(self, x):
        mean = jnp.mean(x, axis=Axis.feature, keepdims=True)
        var = jnp.var(x, axis=Axis.feature, keepdims=True)

        x = (x - mean) / jnp.sqrt((var + self.eps)) * self.weight

        if self.bias is not None:
            x = x + self.bias

        return x

    @classmethod
    def from_config(cls, config: GPTConfig) -> LayerNorm:
        """Create a layer normalization layer from configuration"""
        weight = jnp.ones((config.n_embd,))
        bias = jnp.zeros((config.n_embd,)) if config.use_bias else None
        return cls(weight=weight, bias=bias)
