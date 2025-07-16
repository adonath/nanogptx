from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tiktoken
import tyro
from model import GPT, PretrainedModels
from utils import JAX_DEVICES, Config, JaxDevicesEnum, JaxDtypesEnum

PREFIX = "FILE:"


@dataclass(kw_only=True)
class SampleConfig(Config):
    """Sampling configuration"""

    init_from: PretrainedModels = PretrainedModels.gpt2  # Initialization source
    out_dir: str = "out"  # Output directory (ignored if init_from is not 'resume')
    start: str = "\n"  # Prompt string or file (e.g., '\n', '<|endoftext|>', or 'FILE:prompt.txt')
    num_samples: int = 10  # Number of samples to draw
    max_new_tokens: int = 500  # Number of tokens generated in each sample
    temperature: float = 0.8  # Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
    top_k: int = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337  # Random seed
    device: JaxDevicesEnum = list(JaxDevicesEnum)[0]  # Device to use
    dtype: JaxDtypesEnum = JaxDtypesEnum.float32  # Data type

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

    x = jnp.array(encode(config.prompt), device=device)[None, ...]

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
    config = tyro.cli(SampleConfig)
    sample(config=config)
