import logging
import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import tiktoken
import tyro
from model import GPT
from safetensors import safe_open
from train import InitFrom
from utils import (
    PATH_DATA,
    GlobalConfig,
)

from data import DTYPES, ENCODINGS

PREFIX = "FILE:"


log = logging.getLogger(__file__)


@dataclass(kw_only=True)
class SampleConfig(GlobalConfig):
    """Sampling configuration"""

    init_from: InitFrom = InitFrom.gpt2  # Initialization source
    start: str = (
        ""  # Prompt string or file (e.g., '\n', '<|endoftext|>', or 'FILE:prompt.txt')
    )
    num_samples: int = 10  # Number of samples to draw
    max_new_tokens: int = 500  # Number of tokens generated in each sample
    temperature: float = 0.8  # Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
    top_k: int = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability

    @property
    def prompt(self):
        """Prompt"""
        if self.start.startswith(PREFIX):
            with open(self.start[len(PREFIX) :], "r", encoding="utf-8") as f:
                return f.read()
        return self.start


def sample(config):
    """Sample from a GPT style model"""
    if config.init_from == InitFrom.resume:
        candidates = (PATH_DATA / "checkpoints").glob("*.safetensors")
        latest = max(candidates, key=os.path.getctime)
        model = GPT.read(
            latest,
            device=config.device_jax,
            dtype=config.dtype_jax,
            transpose_weights=False,
        )

        with safe_open(latest, framework="numpy") as f:
            metadata = f.metadata()
            encoding = ENCODINGS[metadata["encoding"]]
    else:
        encoding = tiktoken.get_encoding("gpt2")
        model = GPT.from_pretrained(
            config.init_from, device=config.device_jax, dtype=config.dtype_jax
        )

    x = jnp.asarray(
        encoding.encode(config.prompt, allowed_special={"<|endoftext|>"}),
        device=config.device_jax,
        dtype=DTYPES[encoding.name],
    )[None, ...]

    # use num_samples as batch size
    x = jnp.repeat(x, repeats=config.num_samples, axis=0)

    samples = model.generate(
        x,
        max_new_tokens=config.max_new_tokens,
        rng_key=config.rng_key,
        temperature=config.temperature,
        top_k=config.top_k,
    )

    for sample in samples:
        time.sleep(
            0.1
        )  # Sleep that it looks a bit more like the model generating output
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
