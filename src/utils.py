import hashlib
import json
import logging
import random
import string
import struct
from enum import StrEnum
from functools import partial
from pathlib import Path

import jax
from jax import numpy as jnp
from jax import tree_util

log = logging.getLogger(__name__)

TAB_WIDTH = 4
PATH_BASE = Path(__file__).parent.parent
PATH_DATA = PATH_BASE / "data"
KEY_SEP = "."

FLOPS_UNIT = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS

SAFETENSOR_TO_JAX_DTYPE = {
    "BOOL": jnp.bool_,
    "U8": jnp.uint8,
    "U16": jnp.uint16,
    "U32": jnp.uint32,
    "I8": jnp.int8,
    "I16": jnp.int16,
    "I32": jnp.int32,
    "F16": jnp.float16,
    "F32": jnp.float32,
    "BF16": jnp.bfloat16,
}


if jax.config.values.get("jax_enable_x64", False):
    SAFETENSOR_TO_JAX_DTYPE.update(
        {
            "U64": jnp.uint64,
            "I64": jnp.int64,
            "F64": jnp.float64,
            "C64": jnp.complex64,
            "C128": jnp.complex128,
        }
    )


join_path = partial(tree_util.keystr, simple=True, separator=KEY_SEP)


def read_safetensors_header(file_path: str) -> dict[str, tuple]:
    """Parse safetensors header and return {tensor_name: (shape, jax_dtype)}"""
    with open(file_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    for name, meta in header.items():
        if name == "__metadata__":
            continue

        header[name] = {
            "shape": tuple(meta["shape"]),
            "dtype": SAFETENSOR_TO_JAX_DTYPE[meta["dtype"]],
        }

    return header


def flatten_pytree_with_path(data, parse_type=lambda _: _):
    """Flatten a dict"""
    values, _ = jax.tree.flatten_with_path(data)
    return {join_path(path): parse_type(value) for path, value in values}


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
JaxDevicesEnum = StrEnum("JaxDtypesEnum", list(JAX_DEVICES))
JaxDevicesEnum.jax = property(lambda self: JAX_DEVICES[self.name])


def get_jax_dtypes():
    """Get available dtypes"""
    dtypes = {str(jnp.dtype(_)): _ for _ in SAFETENSOR_TO_JAX_DTYPE.values()}
    return dtypes


JAX_DTYPES = get_jax_dtypes()
JaxDtypesEnum = StrEnum("JaxDtypesEnum", list(JAX_DTYPES))
JaxDtypesEnum.jax = property(lambda self: JAX_DTYPES[self.name])


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


def get_random_name():
    """Generate random adjective-animal name"""
    with (PATH_BASE / "assets/names.json").open("r") as f:
        data = json.load(f)

    letter = random.choice(string.ascii_lowercase)
    animals = data[letter]["animals"]
    adjectives = data[letter]["adjectives"]

    return f"{random.choice(adjectives)}-{random.choice(animals)}".lower()


def update_leave_from_mapping(mapping, use_default_if_missing=False):
    """Update a PyTree leave from a mapping dict[path, value] and coerce to the leave type"""

    def update(path, leave):
        key = join_path(path)
        info = mapping.get(key, leave if use_default_if_missing else None)

        if info is None:
            log.debug(f"No value found for `{key}`, setting to `{info}`")
            return None

        # this requires the leave to be callable...
        return type(leave)(info)

    return update


def sizeof_fmt(num, system="binary"):
    """Human readable version of a bytes number"""
    # fmt: off
    choice = {
        "binary": (("B", "KiB", "MiB", "GiB", "TiB"), 1024.0),
        "decimal": (("K", "M", "B",), 1000.0)
    }
    # fmt: on

    units, divisor = choice[system]

    for unit in units[:-1]:
        if abs(num) < divisor:
            return f"{num:3.1f} {unit}"
        num /= divisor

    return f"{num:.1f} {units[-1]}"
