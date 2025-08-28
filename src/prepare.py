import json
import logging
import lzma
import re
import tarfile
from dataclasses import dataclass
from functools import partial, reduce
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tiktoken
import tyro
from safetensors.numpy import safe_open, save_file
from tqdm import tqdm

from utils import DatasetEnum, EncodingEnum, get_checksum

log = logging.getLogger(__file__)

PATH_DATA = Path(__file__).parent.parent / "data"


@dataclass
class CharEncoding:
    """Character level encoding"""

    stoi: dict
    itos: dict
    name: str = "char"

    @property
    def n_vocab(self):
        """Voca size"""
        return len(self.stoi)

    @property
    def _special_tokens(self):
        return {"<|endoftext|>": self.n_vocab - 1}

    @classmethod
    def from_text(cls, text):
        """Generate encoding from text"""
        chars = sorted(set(text)) + [
            "\n\n",
        ]  # Add <and of text> character
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def shakespeare(cls):
        """Create from shakespeare unique chars"""
        text = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return cls.from_text(text)

    def encode_ordinary(self, sequence, **kwargs):
        """Encode sequence"""
        return [self.stoi[_] for _ in sequence]

    def encode(self, sequence, **kwargs):
        """Encode sequence"""
        return self.encode_ordinary(sequence, **kwargs)

    def decode(self, tokens):
        """Encode sequence"""
        return "".join([self.itos[_] for _ in tokens])


def prepocess(document: str):
    """Preprocess sequence"""
    return re.sub("\n\n\n+", "\n\n", document).strip()


def tokenize(encoding, document) -> list[np.ndarray]:
    """Tokenize a sequence"""
    eot = encoding._special_tokens["<|endoftext|>"]

    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(encoding.encode_ordinary(document))
    return np.array(tokens).astype(DTYPES[encoding.name])


def read_txt_shakespeare(filename) -> list[str]:
    """Read filename"""
    log.info(f"Reading {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    return data.split("\n\n")


def read_xz_from_tar(tar_path, xz_filename) -> list[str]:
    """Read a tarfile and extract the compressed content"""

    with tarfile.open(tar_path, "r") as tar:
        xz_file = tar.extractfile(xz_filename)
        decompressed_data = lzma.decompress(xz_file.read())
        data = decompressed_data.decode("utf-8")

    return data


def read_jsonl_pile_uncopyrighted(json_filename) -> list[str]:
    """Read JSON line format"""

    def extract_text(line):
        """Extract text from a single linee JSON file"""
        return json.loads(line)["text"]

    with json_filename.open("r") as json_file:
        data = list(map(extract_text, json_file))

    return data


def read_json_tinystories(json_filename) -> list[str]:
    """Read a tarfile and extract json"""

    with json_filename.open("r") as json_file:
        data = json.load(json_file)

    return [_["story"] for _ in data]


def read_parquet(filename) -> list[str]:
    """Read a parquet file for fineweb"""
    import pandas as pd

    data = pd.read_parquet(filename)

    return list(data["text"])


def write_safetensors(tokens, filename, encoding):
    """Write safetensors file"""
    log.info(f"Writing {filename}")

    metadata = {
        "n-tokens": str(len(tokens)),
        "encoding": encoding.name,
        "checksum": get_checksum(tokens),
    }

    data = {
        "tokens": tokens,
        "stats": np.bincount(tokens, minlength=encoding.n_vocab).astype(np.uint32),
    }

    save_file(data, filename, metadata=metadata)


def generate_summary(filenames, suffix):
    """Generate summary for a give set of shards"""
    names = []

    with safe_open(filenames[0], framework="numpy") as f:
        encoding = f.metadata()["encoding"]
        stats = np.zeros(ENCODINGS[encoding].n_vocab, dtype=np.uint32)
        n_tokens = 0

    for filename in filenames:
        with safe_open(filename, framework="numpy") as f:
            names.append({"name": filename.name, "checksum": f.metadata()["checksum"]})
            n_tokens += int(f.metadata()["n-tokens"])
            stats += f.get_tensor("stats")

    return {
        "encoding": encoding,
        f"n-tokens-{suffix}": n_tokens,
        f"shards-{suffix}": names,
        f"token-stats-{suffix}": stats.tolist(),
    }


def write_summary(path, shards_val_idxs):
    """Write summary stats file"""
    shards_train, shards_val = [], []

    for filename in Path(path).glob("*.safetensors"):
        idx = int(filename.stem.split("_")[-1])
        (shards_train, shards_val)[idx in shards_val_idxs].append(filename)

    data = generate_summary(shards_train, suffix="train")
    data.update(generate_summary(shards_val, suffix="val"))

    filename_json = path / "summary-stats.json"

    with filename_json.open("w") as json_file:
        log.info(f"Writing {filename_json}")
        json.dump(data, json_file, indent=4)


def expand_filenames_openwebtext(filenames):
    """Return the expanded filenames"""
    filenames_expanded = []

    for filename in filenames:
        with tarfile.open(filename, "r") as tar:
            names = tar.getnames()
            filenames_expanded.extend(zip(repeat(filename), names))

    return filenames_expanded


def apply(pipeline, filename):
    """Apply pipeline steps in series"""

    def step(x, f):
        if isinstance(x, Path):
            return f(x)

        return list(map(f, x))

    return reduce(step, pipeline, filename)


READ_METHODS = {
    DatasetEnum.shakespeare: read_txt_shakespeare,
    DatasetEnum.tinystories: read_json_tinystories,
    DatasetEnum.openwebtext: read_xz_from_tar,
    DatasetEnum.pile_uncopyrighted: read_jsonl_pile_uncopyrighted,
    DatasetEnum.fineweb_10b: read_parquet,
    DatasetEnum.fineweb_100b: read_parquet,
    DatasetEnum.fineweb_edu_10b: read_parquet,
    DatasetEnum.fineweb_edu_100b: read_parquet,
}

ENCODINGS = {
    EncodingEnum.gpt2: tiktoken.get_encoding("gpt2"),
    EncodingEnum.char: CharEncoding.shakespeare(),
}

DTYPES = {EncodingEnum.gpt2: np.uint16, EncodingEnum.char: np.uint16}


@dataclass
class DataPreparationConfig:
    """Data preparation config"""

    shard_size: int = int(1e8)  # Size of each data shard in the output files, in tokens
    shards_val: tuple[int] = (0,)  # Which shards to use for validation
    encoding: EncodingEnum = EncodingEnum.gpt2
    dataset: DatasetEnum = DatasetEnum.shakespeare
    n_process: int = max(1, cpu_count() - 2)
    chunksize: int = 16
    show_progress: bool = True
    write_summary_only: bool = False  # write summary statistics json file


def prepare(config):
    """Prepare and tokenize data"""
    encoding = ENCODINGS[config.encoding]

    steps = [
        READ_METHODS[config.dataset],
        prepocess,
        partial(tokenize, encoding),
    ]

    filenames = list((PATH_DATA / f"download/{config.dataset}").glob("*.*"))

    if config.dataset == DatasetEnum.openwebtext:
        filenames = expand_filenames_openwebtext(filenames)

    log.info(f"Found {len(filenames)} files to process.")
    path = PATH_DATA / "train" / f"{config.dataset}-{config.encoding}"
    path.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        total=config.shard_size,
        disable=not config.show_progress,
        unit="tokens",
    )

    with tqdm(**kwargs) as pbar, Pool(config.n_process) as pool:
        n_shard, n_tokens_total = 0, 0
        tokens_shard = np.empty((config.shard_size,), dtype=DTYPES[config.encoding])

        for result in pool.imap(
            partial(apply, steps), filenames, chunksize=config.chunksize
        ):
            # each process returns a list of token sequences, where each toke sequence typically represents
            # a document
            for tokens in result:
                pbar.set_description(f"Shard {n_shard}")

                if n_tokens_total + len(tokens) < config.shard_size:
                    tokens_shard[n_tokens_total : n_tokens_total + len(tokens)] = tokens
                    n_tokens_total += len(tokens)
                    pbar.update(len(tokens))
                    continue

                remainder = config.shard_size - n_tokens_total
                tokens_shard[n_tokens_total:] = tokens[:remainder]
                pbar.update(remainder)

                filename = path / f"{config.dataset}_shard_{n_shard:06d}.safetensors"
                write_safetensors(tokens_shard, filename=filename, encoding=encoding)

                n_shard += 1
                n_tokens_total = len(tokens) - remainder
                tokens_shard[:n_tokens_total] = tokens[remainder:]

            if n_tokens_total > 0:
                filename = path / f"{config.dataset}_shard_{n_shard:06d}.safetensors"
                write_safetensors(
                    tokens_shard[:n_tokens_total],
                    filename=filename,
                    encoding=encoding,
                )


if __name__ == "__main__":
    config = tyro.cli(DataPreparationConfig)

    if not config.write_summary_only:
        prepare(config)

    path = PATH_DATA / "train" / f"{config.dataset}-{config.encoding}"
    write_summary(path=path, shards_val_idxs=config.shards_val)
