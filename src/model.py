"""Here are some thoughts on how I structured the code

I think JAX and dataclasses are a great fit. I use the dataclasses
in the followwing way:

- "Modules" are defined as dataclasses (e.g. GPT, Embedding, LayerNorm, etc.)
- They define a namespace that combines the parameters with the forward pass
- Classes are registered with JAX and can be combined into a hierarchical PyTree structure, that JAX natively understands
- Configuration is kept separate, I never store configuration in the class, but just initialize its state using
alternative constructors such as .from_config()
- This keeps the model hierarchy clean and keeps serialization etc. simple.
- When I need additonal attributes I implement them as properties, which are derived from the parameter state (such as .vocab_size, .n_embd, etc.)
- This way the classes do not have to be frozen but their state can be modified after initialization without going "out of sync"

What I do like:
- The code turned out very clean and readable.
- The lazy initialization via ArrayInfo is very nice and useful. Especially also implementing the
the intiialization from safetensors as delayed initialization with post init hook.
- The use of Enum for axis and dtype is very nice and makes the code more readable.

What I currently do not like:

- The .from_config() methods introduce a lot of boilerplate code. I have experimented with injecting the default values
via a context manager and context vars, but the pattern felt a bit too unusual.

"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal, Optional

import jax
from dacite import Config as DaciteConfig
from dacite import from_dict
from jax import numpy as jnp
from jax.tree_util import (
    register_dataclass,
    tree_map_with_path,
)
from safetensors import safe_open
from safetensors.flax import save_file

from utils import PATH_DATA, InitFromEnum, PretrainedModels, sizeof_fmt
from utils_jax import (
    JaxDevicesEnum,
    JaxDtypesEnum,
    flatten_pytree_with_path,
    read_safetensors_header,
    update_leave_from_mapping,
)

log = logging.getLogger(__name__)

PATH = Path(__file__).parent
DEFAULT_INIT_STD = 0.02
DEFAULT_DTYPE = JaxDtypesEnum.float32
DEFAULT_RNG_KEY = jax.random.key(98238)
DEFAULT_DEVICE = tuple(JaxDevicesEnum)[0]
DOT_PRODUCT_ATTENTION = "cudnn" if "cuda" in str(DEFAULT_DEVICE) else "xla"


class Axis(int, Enum):
    """Axis order"""

    batch = 0
    sequence = 1
    feature = 2


class EmbeddingAxis(int, Enum):
    """Axis order for embeddings"""

    vocab = 0
    embd = 1


@register_dataclass
@dataclass
class GPTConfig:
    """Model configuration"""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout_rate: float = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
    use_bias: bool = True  # do we use bias inside LayerNorm and Linear layers?
    init_std: float = DEFAULT_INIT_STD

    @property
    def n_embd_mlp(self) -> int:
        """Hidden embedding size for the MLP"""
        return 4 * self.n_embd

    @property
    def n_embd_attn(self) -> int:
        """Embedding size for the stacked qkv attention"""
        return 3 * self.n_embd

    @property
    def init_std_c_proj(self):
        """Standard deviation init for c proj layers"""
        return self.init_std / math.sqrt(2 * self.n_layer)

    @classmethod
    def from_safetensors_meta(cls, data):
        """Create from safetensors meta"""
        data = {_.name: data[f"model.{_.name}"] for _ in fields(cls)}
        return from_dict(
            data_class=cls, data=data, config=DaciteConfig(cast=[int, float, bool])
        )

    @classmethod
    def read_json(cls, path):
        """Read huggingface JSON config file fopr GPT2"""
        with Path(path).open("r") as f:
            data = json.load(f)

        return cls(
            block_size=data["n_ctx"],
            vocab_size=data["vocab_size"],
            dropout_rate=data["attn_pdrop"],
            n_layer=data["n_layer"],
            n_embd=data["n_embd"],
            n_head=data["n_head"],
            init_std=data["initializer_range"],
            use_bias=True,
        )


# TODO: initializers have uniform API since 0.7.0 until then we use:
def initialize_normal(std):
    """Initialize with normal distribution"""

    def normal(key, shape, dtype, out_sharding):
        return std * jax.device_put(
            jax.random.normal(key, shape=shape, dtype=dtype), out_sharding
        )

    return normal


def initialize_ones(key, shape, dtype, out_sharding):
    """Initialize ones"""
    return jnp.ones(shape=shape, dtype=dtype, device=out_sharding)


def initialize_zeros(key, shape, dtype, out_sharding):
    """Initialize ones"""
    return jnp.zeros(shape=shape, dtype=dtype, device=out_sharding)


@dataclass
class initialize_from_safetensors:
    """Init from safetensors file"""

    filename: str
    name: str
    framework: Literal["numpy", "flax", "pt"] = "numpy"

    def __call__(self, key, shape, dtype, out_sharding):
        with safe_open(self.filename, framework=self.framework) as f:
            # TODO: use dlpack for device buffer donation
            array = f.get_tensor(self.name)
            array = jax.device_put(array.astype(dtype), out_sharding)

            if shape is not None and shape != array.shape:
                message = (
                    f"Actual shape of {self.name} {array.shape} "
                    f"does not agree with requested shape {shape}"
                )
                raise ValueError(message)

        return array


@dataclass
class ArrayInfo:
    """Array info, somewhat inspired from jax-llm-examples"""

    shape: tuple[int, ...]
    init: Callable | None = None
    dtype: JaxDtypesEnum = DEFAULT_DTYPE
    out_sharding: JaxDevicesEnum = DEFAULT_DEVICE
    post_init: Callable = lambda _: _

    @property
    def size(self):
        """Array size in number of entries"""
        return math.prod(self.shape)

    @property
    def nbytes(self):
        """Size in number of bytes"""
        return self.size * jnp.dtype(self.dtype).itemsize

    def to_value(self, rng_key, dtype=None, out_sharding=None):
        """Initialize to value"""
        result = self.init(
            key=rng_key,
            shape=self.shape,
            dtype=self.dtype.jax if dtype is None else dtype,
            out_sharding=self.out_sharding.jax
            if out_sharding is None
            else out_sharding,
        )
        return self.post_init(result)

    @classmethod
    def from_safetensors(cls, filename, name, meta):
        """Create array info object from safetensor file"""
        init = initialize_from_safetensors(
            filename=filename,
            name=name,
        )
        return cls(init=init, shape=meta["shape"], dtype=meta["dtype"])


def init_array_leaves(rng_key, dtype=None, out_sharding=None):
    """State based callable for initializing arrays"""

    def init(leave):
        nonlocal rng_key

        if isinstance(leave, ArrayInfo):
            if leave.init is None:
                return None
            rng_key, subkey = jax.random.split(rng_key)
            return leave.to_value(subkey, dtype=dtype, out_sharding=out_sharding)

        return leave

    return init


@register_dataclass
@dataclass
class Embedding:
    """Embedding layer"""

    weight: jax.Array | ArrayInfo

    @property
    def vocab_size(self):
        """Vocabulary size"""
        return self.weight.shape[EmbeddingAxis.vocab]

    @property
    def n_embd(self):
        """Number of embeddings"""
        return self.weight.shape[EmbeddingAxis.embd]

    @staticmethod
    def pad_to_multiple_of(x, multiple=128, value=jnp.nan):
        """Pad to multiple of a value"""
        vocab_size = x.shape[EmbeddingAxis.vocab]
        new_vocab_size = ((vocab_size + multiple - 1) // multiple) * multiple
        padding = new_vocab_size - vocab_size
        width, width[EmbeddingAxis.vocab] = [(0, 0)] * x.ndim, (0, padding)
        return jnp.pad(x, width, constant_values=value, mode="constant")

    @classmethod
    def from_n_features(
        cls,
        vocab_size: int,
        n_embd: int,
        init_std=DEFAULT_INIT_STD,
    ):
        """Create an embedding layer from number of features"""
        weight = ArrayInfo(
            shape=(vocab_size, n_embd),
            init=initialize_normal(init_std),
        )
        return cls(weight=weight)

    def __call__(self, x):
        return jnp.take(self.weight, x, axis=EmbeddingAxis.vocab)


@register_dataclass
@dataclass
class LayerNorm:
    """Layer normalization"""

    weight: jax.Array
    bias: Optional[jax.Array] = None
    eps: ClassVar[float] = 1e-5

    @classmethod
    def from_n_dim(
        cls,
        n_dim: int,
        use_bias: bool = True,
    ):
        """Create a layer normalization layer from number of features"""
        weight = ArrayInfo(
            shape=(n_dim,),
            init=initialize_ones,
        )

        bias = ArrayInfo(
            shape=(n_dim,),
            init=initialize_zeros,
        )

        bias = bias if use_bias else None
        return cls(weight=weight, bias=bias)

    def __call__(self, x):
        mean = jnp.mean(x, axis=Axis.feature, keepdims=True)
        var = jnp.var(x, axis=Axis.feature, keepdims=True)

        x = (x - mean) / jnp.sqrt((var + self.eps)) * self.weight

        if self.bias is not None:
            x = x + self.bias

        return x


@register_dataclass
@dataclass
class Dropout:
    """Dropout layer"""

    rate: float = field(default=0.1, metadata=dict(static=True))

    def __call__(self, x, rng_key, is_training):
        if is_training:
            # taken from https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_dropout.py#L95C13-L97C45
            q = 1 - jax.lax.stop_gradient(self.rate)
            mask = jax.random.bernoulli(rng_key, q, x.shape)
            return jnp.where(mask, x / q, 0)

        return x


@register_dataclass
@dataclass
class Gelu:
    """Gaussian Error Linear Unit"""

    approximate: bool = field(default=True, metadata=dict(static=True))

    @jax.checkpoint
    def __call__(self, x):
        return jax.nn.gelu(x, approximate=self.approximate)


@register_dataclass
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
        cls,
        n_in: int,
        n_out: int,
        use_bias: bool = True,
        init_std=DEFAULT_INIT_STD,
    ):
        """Create a linear layer from number of features"""

        weight = ArrayInfo(
            shape=(n_out, n_in),
            init=initialize_normal(init_std),
        )

        bias = ArrayInfo(
            shape=(n_out,),
            init=initialize_zeros,
        )

        bias = bias if use_bias else None
        return cls(weight=weight, bias=bias)

    def __call__(self, x):
        x = jnp.matmul(x, self.weight.mT)

        if self.bias is not None:
            x = x + self.bias

        return x


@register_dataclass
@dataclass
class MLP:
    """Multi-layer perceptron"""

    c_fc: Linear
    gelu: Gelu
    c_proj: Linear
    dropout: Dropout

    @classmethod
    def from_config(cls, config):
        """Create an MLP layer from configuration"""
        kwargs = {
            "use_bias": config.use_bias,
        }

        c_fc = Linear.from_n_features(
            config.n_embd,
            config.n_embd_mlp,
            init_std=config.init_std,
            **kwargs,
        )

        c_proj = Linear.from_n_features(
            config.n_embd_mlp,
            config.n_embd,
            init_std=config.init_std_c_proj,
            **kwargs,
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


@register_dataclass
@dataclass
class CausalSelfAttention:
    """Causal self-attention layer"""

    c_attn: Linear
    c_proj: Linear
    attn_dropout: Dropout
    resid_dropout: Dropout
    n_head: int = field(metadata=dict(static=True))

    @classmethod
    def from_config(cls, config):
        """Create a causal self-attention layer from configuration"""
        kwargs = {
            "use_bias": config.use_bias,
            "n_in": config.n_embd,
        }
        return cls(
            c_attn=Linear.from_n_features(
                n_out=config.n_embd_attn,
                init_std=config.init_std,
                **kwargs,
            ),
            c_proj=Linear.from_n_features(
                n_out=config.n_embd,
                init_std=config.init_std_c_proj,
                **kwargs,
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

        x_dpa = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=True,
            implementation=DOT_PRODUCT_ATTENTION,
        )

        x = jnp.reshape(x_dpa, x.shape)
        x = self.c_proj(x)
        x = self.resid_dropout(x, rng_key=rng_key, is_training=is_training)
        return x


@register_dataclass
@dataclass
class Block:
    """Self-attention block"""

    ln_1: LayerNorm
    attn: CausalSelfAttention
    ln_2: LayerNorm
    mlp: MLP

    @classmethod
    def from_config(cls, config) -> Block:
        """Create a block from configuration"""
        kwargs_norm = {
            "use_bias": config.use_bias,
            "n_dim": config.n_embd,
        }
        return cls(
            ln_1=LayerNorm.from_n_dim(**kwargs_norm),
            attn=CausalSelfAttention.from_config(config),
            ln_2=LayerNorm.from_n_dim(**kwargs_norm),
            mlp=MLP.from_config(config),
        )

    def __call__(self, x, rng_key, is_training) -> jax.Array:
        x = x + self.attn(self.ln_1(x), rng_key=rng_key, is_training=is_training)
        x = x + self.mlp(self.ln_2(x), rng_key=rng_key, is_training=is_training)
        return x


Flops = namedtuple("Flops", ["per_token", "per_iter", "per_fwdbwd", "tokens_per_iter"])


@dataclass
class GPTInfo:
    """GPT info"""

    n_parameters: int
    n_bytes: int

    def __str__(self):
        return f"GPT Model: {sizeof_fmt(self.n_parameters, system="decimal")} parameters | {sizeof_fmt(self.n_bytes)}"


@register_dataclass
@dataclass
class GPT:
    """GPT Transformer model"""

    wte: Embedding
    wpe: Embedding
    drop: Dropout
    h: list[Block]
    ln_f: LayerNorm

    @property
    def lm_head(self):
        """Weight shared lm head"""
        return Linear(weight=self.wte.weight, bias=None)

    @partial(jax.jit, static_argnames=("is_training", "inference"))
    def __call__(self, idx, rng_key, is_training, inference=False):
        pos = jnp.arange(idx.shape[Axis.sequence])

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        rng_key, sub_rng_key = jax.random.split(rng_key)
        x = self.drop(tok_emb + pos_emb, rng_key=sub_rng_key, is_training=is_training)

        for block in self.h:
            rng_key, sub_rng_key = jax.random.split(rng_key)
            x = block(x, rng_key=sub_rng_key, is_training=is_training)

        x = self.ln_f(x)

        if inference:
            return self.lm_head(x[:, [-1], :])

        logits = self.lm_head(x)
        return logits

    @property
    def config(self):
        """Return configuration for the current model"""
        return GPTConfig(
            block_size=self.wpe.vocab_size,
            vocab_size=self.wte.vocab_size,
            n_layer=len(self.h),
            n_head=self.h[0].attn.n_head,
            n_embd=self.wte.n_embd,
            dropout_rate=self.drop.rate,
            use_bias=self.ln_f.bias is not None,
        )

    def flops(self, batch_size=1, dtype=jnp.int32, sharding=None):
        """Estimate number of flops per iteration"""

        def f(x):
            return self(x, rng_key=DEFAULT_RNG_KEY, is_training=True)

        def get_flops(x):
            compiled = jax.jit(f).trace(x).lower().compile()
            return compiled.cost_analysis()["flops"]

        x = jax.ShapeDtypeStruct((1, 1), dtype=dtype)
        flops_per_token = get_flops(x)

        x = jax.ShapeDtypeStruct((1, self.config.block_size), dtype=dtype)
        flops_per_fwdbwd = get_flops(x)

        x = jax.ShapeDtypeStruct(
            (batch_size, self.config.block_size), dtype=dtype, sharding=sharding
        )
        flops_per_iter = get_flops(x)

        return Flops(
            per_token=flops_per_token,
            per_fwdbwd=flops_per_fwdbwd,
            per_iter=flops_per_iter,
            tokens_per_iter=self.config.block_size * batch_size,
        )

    def info(self, non_embedding=True):
        """Number of parameters"""

        def add_size(a, b, name="size"):
            return getattr(a, name, a) + getattr(b, name, b)

        n_parameters = jax.tree.reduce(add_size, self)
        n_bytes = jax.tree.reduce(partial(add_size, name="nbytes"), self)

        if non_embedding:
            n_parameters -= self.wpe.weight.size
            n_bytes -= self.wpe.weight.nbytes

        return GPTInfo(
            n_parameters=n_parameters,
            n_bytes=n_bytes,
        )

    @classmethod
    def from_config(cls, config):
        """Create a GPT model from configuration"""
        kwargs_emb = {
            "init_std": config.init_std,
        }
        return cls(
            wte=Embedding.from_n_features(
                vocab_size=config.vocab_size,
                n_embd=config.n_embd,
                **kwargs_emb,
            ),
            wpe=Embedding.from_n_features(
                vocab_size=config.block_size,
                n_embd=config.n_embd,
                **kwargs_emb,
            ),
            drop=Dropout(config.dropout_rate),
            h=[Block.from_config(config) for _ in range(config.n_layer)],
            ln_f=LayerNorm.from_n_dim(
                n_dim=config.n_embd,
                use_bias=config.use_bias,
            ),
        )

    def init(self, rng_key=DEFAULT_RNG_KEY, dtype=None, device=None):
        """Init arrays of the model"""
        # TODO: do an abstract evaluation of the shape, dtypes and shardings here?
        init_arrays = init_array_leaves(
            rng_key=rng_key, dtype=dtype, out_sharding=device
        )
        return jax.tree.map(
            init_arrays, self, is_leaf=lambda _: isinstance(_, ArrayInfo)
        )

    @classmethod
    def from_pretrained(cls, model_type, **kwargs) -> GPT:
        """From pretrained model"""
        model_type = PretrainedModels(model_type)
        path = PATH_DATA / f"models/{model_type.value}"

        filename_model = path / "model.safetensors"

        if not filename_model.exists():
            raise FileNotFoundError(
                f"Model {model_type.value} not available. Download weights using 'download.py' first."
            )

        return cls.read(filename_model, **kwargs)

    @classmethod
    def from_init(cls, init_from: InitFromEnum, config: GPTConfig = None):
        """Create from init enum"""
        if init_from == InitFromEnum.scratch:
            if config is None:
                raise ValueError("Init from `scratch` requires defining `config`")
            model = cls.from_config(config)
        elif init_from == InitFromEnum.resume:
            candidates = (PATH_DATA / "checkpoints").glob("**/*.safetensors")
            latest = max(candidates, key=os.path.getctime)
            model = cls.read(latest, transpose_weights=False)
        else:
            model = cls.from_pretrained(init_from)

        return model

    @classmethod
    def read(cls, path, transpose_weights=True, **kwargs) -> GPT:
        """Read model from safetensors file"""
        log.info(f"Reading model from {path}")

        header = read_safetensors_header(path)
        metadata = header.pop("__metadata__")

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        array_infos = {}

        for name, meta in header.items():
            array_infos[name] = ArrayInfo.from_safetensors(
                filename=path,
                name=name,
                meta=meta,
            )

            if any(name.endswith(_) for _ in transposed) and transpose_weights:
                array_infos[name].post_init = jnp.matrix_transpose

        filename_json = path.parent / "config.json"

        if filename_json.exists():
            config = GPTConfig.read_json(filename_json)
        else:
            config = GPTConfig.from_safetensors_meta(metadata)

        model = cls.from_config(replace(config, **kwargs))

        return tree_map_with_path(
            update_leave_from_mapping(array_infos, use_default_if_missing=False), model
        )

    def write(self, path, metadata=None):
        """Write model to safetensors file"""
        data = flatten_pytree_with_path(self)

        if metadata is None:
            metadata = flatten_pytree_with_path(self.config, parse_type=str)

        path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing model to {path}")
        save_file(data, path, metadata=metadata)


def abstract_call(model, x, is_training=True):
    """Call model evaluating shapes, dtypes and shardings"""

    def info_to_struct(info):
        return jax.ShapeDtypeStruct(info.shape, info.dtype, sharding=info.out_sharding)

    model = jax.tree.map(
        info_to_struct, model, is_leaf=lambda _: isinstance(_, ArrayInfo)
    )

    def f(x):
        return model(x, rng_key=DEFAULT_RNG_KEY, is_training=is_training)

    return jax.eval_shape(f, x)
