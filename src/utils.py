import hashlib
import json
import logging
import random
import string
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

import jax
import requests
from jax import numpy as jnp
from jax import tree_util

log = logging.getLogger(__name__)

TAB_WIDTH = 4
PATH_BASE = Path(__file__).parent.parent
PATH_DATA = PATH_BASE / "data"


def asdict_str(data):
    """Return a dict with str values"""

    if is_dataclass(data):
        data = asdict(data)

    return {key: str(value) for key, value in data.items()}


def get_checksum(array):
    """Compute a checksum for an array"""
    # TODO: replace by a checksum implementation that can be computed on a GPU
    return hashlib.md5(array.tobytes()).hexdigest()


def get_jax_devices():
    """Get available devices"""
    available_devices = {}

    for backend in ["tpu", "gpu", "METAL", "cpu"]:
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            continue

        for device in devices:
            available_devices[str(device)] = device

    return available_devices


JAX_DEVICES = get_jax_devices()
AvailableJaxDevices = Literal[tuple(JAX_DEVICES)]


def get_jax_dtypes():
    """Get available dtypes"""
    dtypes = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "int32": jnp.int32}

    if jax.config.values.get("jax_enable_x64", False):
        dtypes["float64"] = jnp.float64
        dtypes["int64"] = jnp.int64

    return dtypes


JAX_DTYPES = get_jax_dtypes()
AvailableJaxDtypes = Literal[tuple(JAX_DTYPES)]


@dataclass(kw_only=True)
class GlobalConfig:
    """GLobal config"""

    seed: int = 9283  # Random seed
    device: AvailableJaxDevices = list(JAX_DEVICES)[0]
    dtype: AvailableJaxDtypes = "float32"
    _key = None

    @property
    def device_jax(self):
        """Return actual device"""
        return JAX_DEVICES[self.device]

    @property
    def dtype_jax(self):
        """Return actual device"""
        return JAX_DTYPES[self.dtype]

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


def dot_product_attention_simple(query, key, value, mask=None):
    """Simple scaled dot product attention, can be used with mps"""
    d_k = query.shape[-1]
    attn_logits = jnp.matmul(query, jnp.swapaxes(key, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)

    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, value)
    return values


def assert_shapes_equal(pytree, other_pytree):
    """Assert that the array shapes of two models are equal."""
    paths, treedef = tree_util.tree_flatten_with_path(pytree)
    paths_other, treedef_other = tree_util.tree_flatten_with_path(other_pytree)

    if not treedef == treedef_other:
        message = f"Tree definitions do not match, got {treedef} and {treedef_other}"
        raise ValueError(message)

    for (path, value), (_, value_other) in zip(paths, paths_other):
        if value.shape != value_other.shape:
            message = f"Shape mismatch at path {join_path(path)}, got {value.shape} and {value_other.shape}"
            raise ValueError(message)


def join_path(path):
    """Join path to Pytree leave"""
    values = [getattr(_, "name", str(getattr(_, "idx", None))) for _ in path]
    return ".".join(values)


def get_random_name():
    """Generate random adjective-animal name"""
    url = "https://raw.githubusercontent.com/fcrespo82/ubuntu-name-generator/refs/heads/master/src/app/names.ts"
    try:
        response = requests.get(url, timeout=1)
    except requests.exceptions.Timeout:
        return "lazy-lama"

    text = response.content.decode("utf-8").replace("export const ubuntu_names = ", "")
    data = json.loads(text)

    letter = random.choice(string.ascii_lowercase)
    animals = data[letter]["animals"]
    adjectives = data[letter]["adjectives"]

    return f"{random.choice(adjectives)}-{random.choice(animals)}".lower()
