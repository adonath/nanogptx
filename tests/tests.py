# TODO: add some unit tests...

import jax
from jax import numpy as jnp

from model import GPT, Embedding, GPTConfig


def test_embedding():
    key = jax.random.PRNGKey(0)
    embed = Embedding.from_n_features(n_embd=3, vocab_size=4, key=key)

    assert embed.weight.shape == (4, 3)

    idxs = jnp.asarray([1, 2, 3, 3, 2])
    y = embed(idxs)

    assert y.shape == (5, 3)


def test_gpt_from_config():
    config = GPTConfig()
    model = GPT.from_config(config)
    # See
    assert model.n_parameters() == 124439808
