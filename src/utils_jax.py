import json
import logging
import struct
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Sequence

import jax
from jax import numpy as jnp
from jax import tree_util
from jax.debug import visualize_sharding
from jax.sharding import NamedSharding, PartitionSpec

log = logging.getLogger(__name__)

KEY_SEP = "."

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

        dtype_str = str(jnp.dtype(SAFETENSOR_TO_JAX_DTYPE[meta["dtype"]]))

        header[name] = {
            "shape": tuple(meta["shape"]),
            "dtype": JaxDtypesEnum(dtype_str),
            "data_offsets": tuple(meta["data_offsets"]),
        }

    return header


def flatten_pytree_with_path(data, parse_type=lambda _: _):
    """Flatten a dict"""
    values, _ = jax.tree.flatten_with_path(data)
    return {join_path(path): parse_type(value) for path, value in values}


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

JaxDevicesEnum = StrEnum("JaxDevicesEnum", list(JAX_DEVICES))
JaxDevicesEnum.jax = property(lambda self: JAX_DEVICES[self.name])


def get_jax_dtypes():
    """Get available dtypes"""
    dtypes = {str(jnp.dtype(_)): _ for _ in SAFETENSOR_TO_JAX_DTYPE.values()}
    return dtypes


JAX_DTYPES = get_jax_dtypes()
JaxDtypesEnum = StrEnum("JaxDtypesEnum", list(JAX_DTYPES))
JaxDtypesEnum.jax = property(lambda self: JAX_DTYPES[self.name])

JaxFloatDtypesEnum = StrEnum(
    "JaxFloatDtypesEnum", [_ for _ in JAX_DTYPES if "float" in _]
)
JaxFloatDtypesEnum.jax = property(lambda self: JAX_DTYPES[self.name])

JaxIntDtypesEnum = StrEnum("JaxIntDtypesEnum", [_ for _ in JAX_DTYPES if "int" in _])
JaxIntDtypesEnum.jax = property(lambda self: JAX_DTYPES[self.name])


def dot_product_attention_simple(query, key, value, mask=None):
    """Simple scaled dot product attention, can be used with mps"""
    d_k = query.shape[-1]
    attn_logits = jnp.matmul(query, jnp.swapaxes(key, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -jnp.inf, attn_logits)

    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, value)
    return values


def update_leave_from_mapping(mapping, use_default_if_missing=False):
    """Update a PyTree leave from a mapping dict[path, value] and coerce to the leave type"""

    def update(path, leave):
        key = join_path(path)
        info = mapping.get(key, leave if use_default_if_missing else None)

        if info is None:
            log.debug(f"No value found for `{key}`, setting to `None`")
            return None

        return info

    return update


@dataclass(frozen=True)
class ShardingConfig:
    """Configurable sharding"""

    devices: Sequence[JaxDevicesEnum] = (tuple(JaxDevicesEnum)[0],)
    axis_names: Sequence[str] = ("batch",)
    axis_shapes: Sequence[int] = (1,)
    partition: Sequence[str] = ("batch",)

    @property
    def mesh_jax(self):
        """Mesh over the batch axis for distributed parallel data training"""
        return jax.make_mesh(
            axis_shapes=self.axis_shapes,
            axis_names=self.axis_names,
            devices=self.devices_jax,
        )

    @property
    def devices_jax(self):
        """Return actual device"""
        return [_.jax for _ in self.devices]

    @property
    def jax(self):
        """Return named sharding"""
        return NamedSharding(self.mesh_jax, PartitionSpec(*self.partition))

    def visualize(self, **kwargs):
        """Visualize sharding"""
        visualize_sharding(self.axis_shapes, self.jax, **kwargs)
