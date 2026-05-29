import logging
import os
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import tiktoken
import tyro
from safetensors import safe_open

from model import DEFAULT_DEVICE, GPT, Axis
from prepare import DTYPES, ENCODINGS
from train import InitFromEnum
from utils import PATH_DATA, EncodingEnum
from utils_jax import JaxDevicesEnum, JaxFloatDtypesEnum

PREFIX = "FILE:"


log = logging.getLogger(__name__)


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
        block_size = model.config.block_size
        # Left-pad so the first sliding-window slice starts at index 0 rather
        # than getting clamped from a negative start. The pad tokens sit at the
        # far left of the attention window where they have the least influence
        # and slide off after `pad_left` generation steps.
        pad_left = max(0, block_size - n_tokens)
        width = [(0, 0)] * tokens.ndim
        width[Axis.sequence] = (pad_left, self.max_new_tokens)
        tokens = jnp.pad(tokens, pad_width=width)

        def sample_step(context, idx):
            context_window = jax.lax.dynamic_slice_in_dim(
                context,
                idx - block_size,
                block_size,
                axis=Axis.sequence,
            )
            logits = model(
                context_window, rng_key=rng_key, is_training=False, inference=True
            )
            logits = logits / self.temperature

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

        start = pad_left + n_tokens
        idxs = jnp.arange(start, start + self.max_new_tokens)
        _, next_tokens = jax.lax.scan(sample_step, tokens, idxs)
        return next_tokens.T


@dataclass
class SampleConfig:
    """Sampling configuration"""

    init_from: InitFromEnum = InitFromEnum.gpt2  # Initialization source
    start: str = "\n"  # Prompt string or file (e.g., '\n', '<|endoftext|>', or 'FILE:prompt.txt')
    device: JaxDevicesEnum = DEFAULT_DEVICE  # Overwrite default device
    dtype: JaxFloatDtypesEnum = JaxFloatDtypesEnum.float32
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


def sample(config):
    """Sample from a GPT style model"""
    if config.init_from == InitFromEnum.scratch:
        raise ValueError("Init from `scratch` is not supported for sampling")

    if config.init_from == InitFromEnum.resume:
        candidates = (PATH_DATA / "checkpoints").glob("**/*.safetensors")
        latest = max(candidates, key=os.path.getctime)
        with safe_open(latest, framework="numpy") as f:
            encoding_name = f.metadata().get("loading.index.encoding", "gpt2")
        encoding = ENCODINGS[EncodingEnum(encoding_name)]
        model = GPT.read(latest, transpose_weights=False)
    else:
        encoding = tiktoken.get_encoding("gpt2")
        model = GPT.from_init(config.init_from)

    prompt_tokens = encoding.encode(config.prompt, allowed_special={"<|endoftext|>"})
    x = jnp.asarray(prompt_tokens, dtype=DTYPES[encoding.name])[None, ...]
    x = jax.device_put(x, config.device.jax)

    model = model.init(
        device=config.device.jax,
        dtype=config.dtype.jax,
    )

    log.info("%s", model.info())
    rng_key = config.rng_key
    for idx in range(config.sampler.num_samples):
        generated = config.sampler.generate(
            model=model,
            tokens=x,
            rng_key=jax.random.fold_in(rng_key, idx),
        )

        print(encoding.decode(prompt_tokens + generated[0].tolist()))
        print("---------------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = tyro.cli(SampleConfig)

    start_time = time.perf_counter()
    sample(config=config)
    end_time = time.perf_counter()

    tokens_per_second = (config.sampler.num_samples * config.sampler.max_new_tokens) / (
        end_time - start_time
    )
    log.info("Sampled at %.2f tok/s", tokens_per_second)
