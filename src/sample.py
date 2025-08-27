import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import tiktoken
import tyro
from jax import tree_util
from safetensors import safe_open

from model import DEFAULT_DEVICE, GPT, Axis
from prepare import DTYPES, ENCODINGS
from train import InitFromEnum
from utils import (
    PATH_DATA,
    JaxDevicesEnum,
    JaxFloatDtypesEnum,
)

PREFIX = "FILE:"


log = logging.getLogger(__file__)


@tree_util.register_dataclass
@dataclass
class TokenSampler:
    """Token sampler"""

    num_samples: int = 10  # Number of samples to draw
    max_new_tokens: int = 500  # Number of tokens generated in each sample
    temperature: float = 0.8  # Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
    top_k: int = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability

    def generate(self, model, tokens, rng_key):
        """Generate new tokens"""
        top_k = (
            min(self.top_k, model.wte.vocab_size) if self.top_k is not None else None
        )

        n_tokens = tokens.shape[Axis.sequence]
        width, width[Axis.sequence] = [(0, 0)] * tokens.ndim, (0, self.max_new_tokens)
        tokens = jnp.pad(tokens, pad_width=width)

        def sample(context, idx):
            context_window = jax.lax.dynamic_slice_in_dim(
                context,
                idx - model.config.block_size,
                model.config.block_size,
                axis=Axis.sequence,
            )
            logits = model(context_window, rng_key=rng_key, is_training=False)
            logits = logits[:, -1:, :] / self.temperature

            if top_k is not None:
                values, indices = jax.lax.top_k(logits, top_k)
            else:
                values, indices = logits, jnp.arange(model.wte.vocab_size)

            probs = jax.nn.softmax(values, axis=Axis.feature)

            keys = jax.random.split(
                jax.random.fold_in(rng_key, idx), context.shape[Axis.batch]
            )
            next_token = jax.vmap(jax.random.choice)(
                keys,
                indices[:, 0, :],
                p=probs[:, 0, :],
            )

            context = context.at[:, idx].set(next_token)
            return context, next_token

        idxs = jnp.arange(n_tokens, n_tokens + self.max_new_tokens)
        _, next_tokens = jax.lax.scan(sample, tokens, idxs)
        return next_tokens.T


# fmt: off
@tree_util.register_dataclass
@dataclass
class SampleConfig:
    """Sampling configuration"""

    init_from: InitFromEnum = InitFromEnum.gpt2  # Initialization source
    start: str = "\n"  # Prompt string or file (e.g., '\n', '<|endoftext|>', or 'FILE:prompt.txt')
    device: Optional[JaxDevicesEnum] = DEFAULT_DEVICE # Overwrite default device
    dtype: Optional[JaxFloatDtypesEnum] = JaxFloatDtypesEnum.float32
    seed: int = 9283  # Random seed
    sampler: TokenSampler = field(default_factory=TokenSampler)

    @property
    def rng_key(self) -> jax.Array:
        """Generate random key for initialization"""
        return jax.random.key(self.seed)

    @property
    def prompt(self):
        """Prompt"""
        if self.start.startswith(PREFIX):
            with open(self.start[len(PREFIX) :], "r", encoding="utf-8") as f:
                return f.read()
        return self.start
# fmt: on


def sample(config):
    """Sample from a GPT style model"""
    if config.init_from == InitFromEnum.resume:
        candidates = (PATH_DATA / "checkpoints").glob("*.safetensors")
        latest = max(candidates, key=os.path.getctime)
        model = GPT.read(
            latest,
            transpose_weights=False,
        )

        with safe_open(latest, framework="numpy") as f:
            from train import Config

            config_all = Config.from_safetensors_meta(f.metadata())
            encoding = ENCODINGS[config_all.loading.index.encoding]
    else:
        encoding = tiktoken.get_encoding("gpt2")
        model = GPT.from_pretrained(config.init_from)

    x = jnp.asarray(
        encoding.encode(config.prompt, allowed_special={"<|endoftext|>"}),
        dtype=DTYPES[encoding.name],
    )[None, ...]

    x = jax.device_put(x, config.device.jax)

    model = model.init(
        device=config.device.jax,
        dtype=config.dtype.jax,
    )

    log.info(f"{model.info()}")
    for idx in range(config.sampler.num_samples):
        sample = config.sampler.generate(
            model=model,
            tokens=x,
            rng_key=jax.random.fold_in(config.rng_key, idx),
        )

    print(encoding.decode(sample[0].tolist()))
    print("---------------")


if __name__ == "__main__":
    config = tyro.cli(SampleConfig)

    start_time = time.time()
    sample(config=config)
    end_time = time.time()

    tokens_per_second = (config.sampler.num_samples * config.sampler.max_new_tokens) / (
        end_time - start_time
    )
    log.info(f"Sampled at {tokens_per_second:.2f} TPS")
