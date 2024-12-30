import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

import jax
import tomli_w
import tomllib

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
