import hashlib
import json
import random
import string
from enum import StrEnum
from pathlib import Path

TAB_WIDTH = 4
PATH_BASE = Path(__file__).parent.parent
PATH_DATA = PATH_BASE / "data"
FLOPS_UNIT = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS


class EncodingEnum(StrEnum):
    """Encoding enum"""

    gpt2 = "gpt2"
    char = "char"


class DatasetEnum(StrEnum):
    """Dataset enum"""

    shakespeare = "shakespeare"
    openwebtext = "openwebtext"
    fineweb_10b = "fineweb-10b"
    fineweb_100b = "fineweb-100b"
    fineweb_edu_10b = "fineweb-edu-10b"
    fineweb_edu_100b = "fineweb-edu-100b"
    tinystories = "tinystories"
    pile_uncopyrighted = "pile-uncopyrighted"


class PretrainedModels(StrEnum):
    """Pretrained models"""

    resume = "resume"
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


def get_checksum(array):
    """Compute a checksum for an array"""
    # TODO: replace by a checksum implementation that can be computed on a GPU
    return hashlib.md5(array.tobytes()).hexdigest()


def get_random_name():
    """Generate random adjective-animal name"""
    with (PATH_BASE / "assets/names.json").open("r") as f:
        data = json.load(f)

    letter = random.choice(string.ascii_lowercase)
    animals = data[letter]["animals"]
    adjectives = data[letter]["adjectives"]

    return f"{random.choice(adjectives)}-{random.choice(animals)}".lower()


def sizeof_fmt(num, system="binary"):
    """Human readable version of a large number"""
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
