import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

import jax
import tomli_w
import tomllib
from jax import tree_util

log = logging.getLogger(__name__)


class register_dataclass_jax:
    """Decorator to register a dataclass with JAX."""

    def __init__(
        self,
        data_fields: Optional[list] = None,
        meta_fields: Optional[list] = None,
    ) -> None:
        self.data_fields = data_fields or []
        self.meta_fields = meta_fields or []

    def __call__(self, cls: Any) -> Any:
        jax.tree_util.register_dataclass(
            cls,
            data_fields=self.data_fields,
            meta_fields=self.meta_fields,
        )
        return cls


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


@dataclass
class Config:
    """Configuration class"""

    @classmethod
    def read(cls, path: str):
        """Read configuration from file"""
        log.info(f"Reading configuration from {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)
            return cls(**data)

    def write(self, path: str):
        """Write configuration to file"""
        log.info(f"Writing configuration to {path}")

        with open(path, "w") as f:
            tomli_w.dump(asdict(self), f)

    def update(self, **kwds):
        """Update configuration"""
        return replace(self, **kwds)
