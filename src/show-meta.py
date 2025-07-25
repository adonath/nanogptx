from pathlib import Path

import tomli_w
import tyro
from safetensors import safe_open

TAB_WIDTH = 4


def show_meta(filename: str):
    """Open safetensors file and print metadata"""

    with safe_open(Path(filename), framework="numpy") as f:
        metadata = f.metadata()
        print(tomli_w.dumps(metadata, indent=TAB_WIDTH))


if __name__ == "__main__":
    config = tyro.cli(show_meta)
