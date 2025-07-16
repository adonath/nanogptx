from dataclasses import dataclass, field

import jax
import tiktoken
from jax import numpy as jnp

from model import GPT, PretrainedModels
from utils import JAX_DEVICES, JAX_DTYPES, Config

PREFIX = "FILE:"


@dataclass
class SampleConfig(Config):
    """Evaluation configuration"""

    init_from: str = field(
        default="gpt2",
        metadata={
            "help": "Initialization source",
            "choices": [e.value for e in PretrainedModels],
        },
    )
    out_dir: str = field(
        default="out",
        metadata={"help": "Output directory (ignored if init_from is not 'resume')"},
    )
    start: str = field(
        default="\n",
        metadata={
            "help": f"Prompt string or file (e.g., '\n', '<|endoftext|>', or '{PREFIX}prompt.txt')"
        },
    )
    num_samples: int = field(default=10, metadata={"help": "Number of samples to draw"})
    max_new_tokens: int = field(
        default=500, metadata={"help": "Number of tokens generated in each sample"}
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)"
        },
    )
    top_k: int = field(
        default=200,
        metadata={
            "help": "Retain only the top_k most likely tokens, clamp others to have 0 probability"
        },
    )
    seed: int = field(default=1337, metadata={"help": "Random seed"})
    device: str = field(
        default=list(JAX_DEVICES)[0],
        metadata={"help": "Device to use", "choices": list(JAX_DEVICES)},
    )
    dtype: str = field(
        default=list(JAX_DTYPES)[0],
        metadata={"help": "Data type", "choices": list(JAX_DTYPES)},
    )

    @property
    def prompt(self):
        """Prompt"""
        if self.start.startswith(PREFIX):
            with open(self.start[len(PREFIX) :], "r", encoding="utf-8") as f:
                return f.read()

        return self.start


def sample(config):
    """Sample from a GPT style model"""
    enc = tiktoken.get_encoding("gpt2")

    def encode(str_):
        return enc.encode(str_, allowed_special={"<|endoftext|>"})

    def decode(str_):
        return enc.decode(str_)

    device = JAX_DEVICES[config.device]

    x = jnp.array(encode(config.prompt), dtype=jnp.int64, device=device)[None, ...]

    model = GPT.from_pretrained(config.init_from, device=device)

    for _ in range(config.num_samples):
        y = model.generate(
            x,
            max_new_tokens=config.max_new_tokens,
            rng_key=jax.random.key(config.seed),
            temperature=config.temperature,
            top_k=config.top_k,
        )
        print(decode(y[0].tolist()))
        print("---------------")


if __name__ == "__main__":
    config = SampleConfig.from_argparse()
    sample(config=config)
