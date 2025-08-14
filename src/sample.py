import logging
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tiktoken
import tyro
from jax import tree_util
from safetensors import safe_open

from model import GPT
from prepare import DTYPES, ENCODINGS
from train import InitFromEnum
from utils import (
    JAX_DEVICES,
    PATH_DATA,
    AvailableJaxDevices,
    JaxDtypesEnum,
)

PREFIX = "FILE:"


log = logging.getLogger(__file__)


@tree_util.register_dataclass
@dataclass
class SampleConfig:
    """Sampling configuration"""

    init_from: InitFromEnum = InitFromEnum.gpt2  # Initialization source
    start: str = (
        ""  # Prompt string or file (e.g., '\n', '<|endoftext|>', or 'FILE:prompt.txt')
    )
    num_samples: int = 10  # Number of samples to draw
    max_new_tokens: int = 500  # Number of tokens generated in each sample
    temperature: float = 0.8  # Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
    top_k: int = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 9283  # Random seed
    device: AvailableJaxDevices = list(JAX_DEVICES)[0]
    dtype: JaxDtypesEnum = JaxDtypesEnum.float32
    _key = None

    @property
    def device_jax(self):
        """Return actual device"""
        return JAX_DEVICES[self.device]

    @property
    def rng_key(self) -> jax.Array:
        """Generate random key for initialization"""
        if self._key is None:
            self._key = jax.random.PRNGKey(self.seed)

        # in general state based key generation is not a good idea in Jax!
        # however the config class never(!) crosses any jit and function transform
        # boundaries. So it is safe to use it here.
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @property
    def prompt(self):
        """Prompt"""
        if self.start.startswith(PREFIX):
            with open(self.start[len(PREFIX) :], "r", encoding="utf-8") as f:
                return f.read()
        return self.start


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
            encoding = ENCODINGS[config_all.data.encoding]
    else:
        encoding = tiktoken.get_encoding("gpt2")
        model = GPT.from_pretrained(config.init_from)

    x = jnp.asarray(
        encoding.encode(config.prompt, allowed_special={"<|endoftext|>"}),
        device=config.device_jax,
        dtype=DTYPES[encoding.name],
    )[None, ...]

    # use num_samples as batch size
    x = jnp.repeat(x, repeats=config.num_samples, axis=0)

    samples = model.init(
        device=config.device_jax,
        dtype=config.dtype.jax,
    ).generate(
        x,
        max_new_tokens=config.max_new_tokens,
        rng_key=config.rng_key,
        temperature=config.temperature,
        top_k=config.top_k,
    )

    for sample in samples:
        time.sleep(0.1)
        print(encoding.decode(sample.tolist()))
        print("---------------")


if __name__ == "__main__":
    config = tyro.cli(SampleConfig)

    start_time = time.time()
    sample(config=config)
    end_time = time.time()

    tokens_per_second = (config.num_samples * config.max_new_tokens) / (
        end_time - start_time
    )
    log.info(f"Sampled at {tokens_per_second:.2f} TPS")
