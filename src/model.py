"""Here are some thoughts on how to structure the code"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import ClassVar, Optional

import jax
from jax import numpy as jnp
from jax import tree_util
from safetensors import safe_open
from safetensors.flax import save_file
from utils import Config, join_path, register_dataclass_jax

log = logging.getLogger(__name__)

PATH = Path(__file__).parent


class PretrainedModels(str, Enum):
    """Pretrained models"""

    resume = "resume"
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


class Axis(int, Enum):
    """Axis order"""

    batch = 0
    sequence = 1
    feature = 2


class EmbeddingAxis(int, Enum):
    """Axis order for embeddings"""

    vocab = 0
    embd = 1


@dataclass(kw_only=True)
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
        """Embedding size for the stacked qkv attention"""
        return 3 * self.n_embd

    def generate_rng_key(self) -> jax.Array:
        """Generate random key for initialization"""
        if self._key is None:
            self._key = jax.random.PRNGKey(self.seed)

        # in general state based key generation is not a good idea in Jax!
        # however the config class never(!) crosses any jit and function transform
        # boundaries. So it is safe to use it here.
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @classmethod
    def dummy(cls, n_layer: int = 12, n_head: int = 12):
        """Dummy configuration to create a model with minimal parameters but with equivalent PyTree structure"""
        return cls(
            block_size=0,
            vocab_size=0,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=0,
        )


@register_dataclass_jax(data_fields=["weight"])
@dataclass(frozen=True)
class Embedding:
    """Embedding layer"""

    weight: jax.Array

    @property
    def vocab_size(self):
        """Vocabulary size"""
        return self.weight.shape[EmbeddingAxis.vocab]

    @property
    def n_embd(self):
        """Number of embeddings"""
        return self.weight.shape[EmbeddingAxis.embd]

    @classmethod
    def from_n_features(cls, vocab_size: int, n_embd: int, rng_key: jax.Array):
        """Create an embedding layer from number of features"""
        return cls(weight=jax.random.normal(rng_key, (vocab_size, n_embd)))

    @classmethod
    def from_config(cls, config):
        """Create an embedding layer from configuration"""
        return cls.from_n_features(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            rng_key=config.generate_rng_key(),
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

    @classmethod
    def from_n_dim(cls, n_dim: int, use_bias: bool = True):
        """Create a layer normalization layer from number of features"""
        weight = jnp.ones((n_dim,))
        bias = jnp.zeros((n_dim,)) if use_bias else None
        return cls(weight=weight, bias=bias)

    @classmethod
    def from_config(cls, config: GPTConfig) -> LayerNorm:
        """Create a layer normalization layer from configuration"""
        return cls.from_n_dim(config.n_embd, use_bias=config.use_bias)

    def __call__(self, x):
        mean = jnp.mean(x, axis=Axis.feature, keepdims=True)
        var = jnp.var(x, axis=Axis.feature, keepdims=True)

        x = (x - mean) / jnp.sqrt((var + self.eps)) * self.weight

        if self.bias is not None:
            x = x + self.bias

        return x


@register_dataclass_jax(meta_fields=["rate"])
@dataclass
class Dropout:
    """Dropout layer"""

    rate: float = 0.1

    def __call__(self, x, rng_key, is_training):
        if is_training:
            # taken from https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_dropout.py#L95C13-L97C45
            q = 1 - jax.lax.stop_gradient(self.rate)
            mask = jax.random.bernoulli(rng_key, q, x.shape)
            return jnp.where(mask, x / q, 0)

        return x

    @classmethod
    def from_config(cls, config):
        """Create a dropout layer from configuration"""
        return cls(rate=config.dropout_rate)


@register_dataclass_jax(meta_fields=["approximate"])
@dataclass
class Gelu:
    """Gaussian Error Linear Unit"""

    approximate: bool = True

    def __call__(self, x):
        return jax.nn.gelu(x, approximate=self.approximate)


@register_dataclass_jax(data_fields=["weight", "bias"])
@dataclass
class Linear:
    """Linear layer"""

    weight: jax.Array
    bias: Optional[jax.Array] = None

    @property
    def n_in(self):
        """Number of input features"""
        return self.weight.shape[Axis.feature]

    @property
    def n_out(self):
        """Number of output features"""
        return self.weight.shape[Axis.sequence]

    @classmethod
    def from_n_features(
        cls, n_in: int, n_out: int, rng_key: jax.Array, use_bias: bool = True
    ):
        """Create a linear layer from number of features"""
        weight = jax.random.normal(rng_key, (n_out, n_in))
        bias = jnp.zeros(n_out) if use_bias else None
        return cls(weight=weight, bias=bias)

    @classmethod
    def from_config(cls, config):
        """Create a linear layer from configuration"""
        return cls.from_n_features(
            config.n_embd,
            config.n_embd_mlp,
            rng_key=config.generate_rng_key(),
            use_bias=config.use_bias,
        )

    def __call__(self, x):
        x = jnp.matmul(x, self.weight.mT)

        if self.bias is not None:
            x = x + self.bias

        return x


@register_dataclass_jax(data_fields=["c_fc", "gelu", "c_proj", "dropout"])
@dataclass(frozen=True)
class MLP:
    """Multi-layer perceptron"""

    c_fc: Linear
    gelu: Gelu
    c_proj: Linear
    dropout: Dropout

    @classmethod
    def from_config(cls, config):
        """Create an MLP layer from configuration"""
        c_fc = Linear.from_n_features(
            config.n_embd,
            config.n_embd_mlp,
            use_bias=config.use_bias,
            rng_key=config.generate_rng_key(),
        )
        c_proj = Linear.from_n_features(
            config.n_embd_mlp,
            config.n_embd,
            use_bias=config.use_bias,
            rng_key=config.generate_rng_key(),
        )
        return cls(
            c_fc=c_fc, gelu=Gelu(), c_proj=c_proj, dropout=Dropout(config.dropout_rate)
        )

    def __call__(self, x, rng_key, is_training) -> jax.Array:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        rng_key, _ = jax.random.split(rng_key)
        x = self.dropout(x, rng_key=rng_key, is_training=is_training)
        return x


@register_dataclass_jax(
    data_fields=["c_attn", "c_proj", "attn_dropout", "resid_dropout"],
    meta_fields=["n_head"],
)
@dataclass
class CausalSelfAttention:
    """Causal self-attention layer"""

    c_attn: Linear
    c_proj: Linear
    attn_dropout: Dropout
    resid_dropout: Dropout
    n_head: int

    @classmethod
    def from_config(cls, config):
        """Create a causal self-attention layer from configuration"""
        return cls(
            c_attn=Linear.from_n_features(
                config.n_embd,
                config.n_embd_attn,
                rng_key=config.generate_rng_key(),
                use_bias=config.use_bias,
            ),
            c_proj=Linear.from_n_features(
                config.n_embd,
                config.n_embd,
                rng_key=config.generate_rng_key(),
                use_bias=config.use_bias,
            ),
            n_head=config.n_head,
            attn_dropout=Dropout(config.dropout_rate),
            resid_dropout=Dropout(config.dropout_rate),
        )

    def __call__(self, x, rng_key, is_training):
        query, key, value = jnp.split(self.c_attn(x), 3, axis=Axis.feature)

        shape = (
            x.shape[Axis.batch],
            x.shape[Axis.sequence],
            self.n_head,
            x.shape[Axis.feature] // self.n_head,
        )
        query = jnp.reshape(query, shape)
        key = jnp.reshape(key, shape)
        value = jnp.reshape(value, shape)

        x = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=True,
        )

        x = jnp.reshape(x, (x.shape[Axis.batch], x.shape[Axis.sequence], -1))
        x = self.c_proj(x)
        x = self.resid_dropout(x, rng_key=rng_key, is_training=is_training)
        return x


@register_dataclass_jax(data_fields=["ln_1", "attn", "ln_2", "mlp"])
@dataclass(frozen=True)
class Block:
    """Self-attention block"""

    ln_1: LayerNorm
    attn: CausalSelfAttention
    ln_2: LayerNorm
    mlp: MLP

    @classmethod
    def from_config(cls, config: GPTConfig) -> Block:
        """Create a block from configuration"""
        return cls(
            ln_1=LayerNorm.from_config(config),
            attn=CausalSelfAttention.from_config(config),
            ln_2=LayerNorm.from_config(config),
            mlp=MLP.from_config(config),
        )

    def __call__(self, x, rng_key, is_training) -> jax.Array:
        x = x + self.attn(self.ln_1(x), rng_key=rng_key, is_training=is_training)
        x = x + self.mlp(self.ln_2(x), rng_key=rng_key, is_training=is_training)
        return x


@register_dataclass_jax(data_fields=["wte", "wpe", "drop", "h", "ln_f", "lm_head"])
@dataclass(frozen=True)
class GPT:
    """GPT Transformer model"""

    wte: Embedding
    wpe: Embedding
    drop: Dropout
    h: list[Block]
    ln_f: LayerNorm
    lm_head: Linear

    def __post_init__(self):
        self.lm_head.weight = self.lm_head.weight.at[:].set(self.wte.weight)

    @partial(jax.jit, static_argnames=("is_training",))
    def __call__(self, idx, rng_key, is_training):
        pos = jnp.arange(idx.shape[Axis.sequence])

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        rng_key, sub_rng_key = jax.random.split(rng_key)
        x = self.drop(tok_emb + pos_emb, rng_key=sub_rng_key, is_training=is_training)

        for block in self.h:
            rng_key, sub_rng_key = jax.random.split(rng_key)
            x = block(x, rng_key=sub_rng_key, is_training=is_training)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @property
    def block_size(self):
        """Block size"""
        return self.wpe.vocab_size

    def to_config(self):
        """Return configuration for model"""
        return GPTConfig(
            block_size=self.block_size,
            vocab_size=self.wte.vocab_size,
            n_layer=len(self.h),
            n_head=self.h[0].attn.n_head,
            n_embd=self.wte.n_embd,
            dropout_rate=self.drop.rate,
            use_bias=self.ln_f.bias is not None,
        )

    @classmethod
    def from_config(cls, config):
        """Create a GPT model from configuration"""
        return cls(
            wte=Embedding.from_config(config),
            wpe=Embedding.from_n_features(
                config.block_size, config.n_embd, rng_key=config.generate_rng_key()
            ),
            drop=Dropout.from_config(config),
            h=[Block.from_config(config) for _ in range(config.n_layer)],
            ln_f=LayerNorm.from_config(config),
            lm_head=Linear.from_n_features(
                config.n_embd,
                config.vocab_size,
                rng_key=config.generate_rng_key(),
                use_bias=False,
            ),
        )

    def n_parameters(self, non_embedding=True):
        """Number of parameters"""
        n_parameters = sum(
            p.size if isinstance(p, jax.Array) else 0 for p in jax.tree_leaves(self)
        )

        if non_embedding:
            n_parameters -= self.wte.weight.size

        return n_parameters

    @classmethod
    def from_pretrained(cls, model_type, device=None) -> GPT:
        """From pretrained model"""
        model_type = PretrainedModels(model_type)
        path = PATH / f"data/models/{model_type.value}/model.safetensors"

        if not path.exists():
            raise FileNotFoundError(
                f"Model {model_type.value} not available. Download weights using 'download.py' first."
            )

        return cls.read(path, device=device)

    @classmethod
    def read(cls, path, transpose_weights=True, device=None) -> GPT:
        """Read model from safetensors file"""
        # create a dummy model to get the equivalent PyTree structure, this is
        # not nice, but jax does allow generate a PyTree from static definitions
        dummy_model = GPT.from_config(GPTConfig.dummy())

        paths, treedef = tree_util.tree_flatten_with_path(dummy_model)

        data_model = {join_path(path): value for path, value in paths}

        log.info(f"Reading model from {path}")

        data = {}
        with safe_open(path, framework="flax", device=device) as f:
            for k in f.keys():
                data[k] = f.get_tensor(k)

        # tied parameters are missing, just creat a reference as placeholder
        data["lm_head.weight"] = data["wte.weight"]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for key in data_model:
            data_model[key] = data[key]

            if any(key.endswith(_) for _ in transposed) and transpose_weights:
                data_model[key] = data_model[key].T

        return tree_util.tree_unflatten(treedef, data_model.values())

    def write(self, path):
        """Write model to safetensors file"""
        paths, _ = tree_util.tree_flatten_with_path(self)
        data = {join_path(path): value for path, value in paths}

        log.info(f"Writing model to {path}")
        save_file(data, path)

    def generate(self, idx, max_new_tokens, rng_key, temperature=1.0, top_k=None):
        """Generate new tokens"""
        top_k = min(top_k, self.wte.vocab_size) if top_k is not None else None

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if len(idx) <= self.block_size else idx[:, -self.block_size :]
            )
            logits = self(idx_cond, rng_key=rng_key, is_training=False)
            logits = logits[:, -1:, :] / temperature

            if top_k is not None:
                values, indices = jax.lax.top_k(logits, top_k)
            else:
                values, indices = logits, jnp.arange(self.wte.vocab_size)

            probs = jax.nn.softmax(values, axis=-1)

            rng_key, sub_rng_key = jax.random.split(rng_key)
            idx_next = jax.random.choice(
                sub_rng_key,
                indices.flatten(),
                p=probs.flatten(),
                shape=(1, 1),
            )
            idx = jnp.concat((idx, idx_next), axis=Axis.sequence)

        return idx
