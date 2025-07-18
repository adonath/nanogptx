import json
import logging
import lzma
import re
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from functools import partial, reduce
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tiktoken
import tyro
from jax import numpy as jnp
from safetensors.numpy import safe_open, save_file
from tqdm import tqdm

log = logging.getLogger(__file__)

PATH_DATA = Path(__file__).parent.parent / "data"


class EncodingEnum:
    """Tokenizer"""

    gpt2 = "gpt2"


DTYPES = {EncodingEnum.gpt2: np.uint16}


class DatasetEnum(StrEnum):
    """Dataset enum"""

    shakespeare = "shakespeare"
    openwebtext = "openwebtext"
    fineweb_10b = "fineweb-10b"
    fineweb_100b = "fineweb-100b"
    fineweb_edu_10b = "fineweb-edu-10b"
    fineweb_edu_100b = "fineweb-edu-100b"


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


def prepocess(sequence):
    """Preprocess sequence"""
    return re.sub("\n\n\n+", "\n\n", sequence).strip()


def tokenize(sequence, encoding=EncodingEnum.gpt2):
    """Tokenize a sequence"""
    enc = tiktoken.get_encoding(encoding)
    eot = enc._special_tokens["<|endoftext|>"]

    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(sequence))
    return jnp.array(tokens).astype(DTYPES[encoding])


def read_txt(filename):
    """Read filename"""
    log.info(f"Reading {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    return data


def read_xz_from_tar(tar_path, xz_filename):
    """Read a tarfile and extract the compressed content"""

    with tarfile.open(tar_path, "r") as tar:
        xz_file = tar.extractfile(xz_filename)
        decompressed_data = lzma.decompress(xz_file.read())
        data = decompressed_data.decode("utf-8")

    return data


def read_parquet():
    """Read a parquet file for fineweb"""
    ...


def write_safetensors(tokens, filename, encoding):
    """Write safetensors file"""
    log.info(f"Writing {filename}")

    metadata = {"n-tokens": str(len(tokens)), "encoding": encoding}

    n_vocab = tiktoken.get_encoding(config.encoding).n_vocab
    data = {
        "tokens": tokens,
        "stats": np.bincount(tokens, minlength=n_vocab),
    }

    save_file(data, filename, metadata=metadata)


def apply(pipeline, filename):
    """Apply pipeline steps in series"""

    def unpack(f, args):
        if isinstance(args, tuple):
            return f(*args)

        return f(args)

    return reduce(lambda x, f: unpack(f, x), pipeline, filename)


def write_summary(path, shards_val_idxs):
    """Write summary stats file"""
    filenames = list(Path(path).glob("*.safetensors"))

    for idx, filename in enumerate(filenames):
        with safe_open(filename, framework="numpy") as f:
            if idx == 0:
                stats = f.get_tensor("stats")
                n_tokens = int(f.metadata()["n-tokens"])
                encoding = f.metadata()["encoding"]
                continue

            n_tokens += int(f.metadata()["n-tokens"])
            stats += f.get_tensor("stats")

    shards_train, shards_val = [], []

    for filename in filenames:
        shard_idx = int(filename.stem.split("_")[-1])
        if shard_idx in shards_val_idxs:
            shards_val.append(filename.name)
            continue

        shards_train.append(filename.name)

    data = {
        "n-tokens": n_tokens,
        "encoding": encoding,
        "shards-train": shards_train,
        "shards-val": shards_val,
        "token-stats": stats.tolist(),
    }

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


PIPELINES = {
    DatasetEnum.shakespeare: partial(apply, [read_txt, prepocess, tokenize]),
    DatasetEnum.openwebtext: partial(apply, [read_xz_from_tar, prepocess, tokenize]),
}


def prepare(config):
    """Prepare and tokenize data"""
    pipeline = PIPELINES[config.dataset]
    filenames = (PATH_DATA / f"download/{config.dataset}").glob("*.*")

    if config.dataset == DatasetEnum.openwebtext:
        filenames = expand_filenames_openwebtext(filenames)

    log.info(f"Found {len(filenames)} files to process.")
    path = PATH_DATA / "train" / config.dataset
    path.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=config.shard_size,
        disable=not config.show_progress,
        unit="tokens",
    ) as pbar:
        with Pool(config.n_process) as pool:
            n_shard, n_tokens_total = 0, 0
            tokens_shard = np.empty((config.shard_size,), dtype=DTYPES[config.encoding])

            for tokens in pool.imap(pipeline, filenames, chunksize=config.chunksize):
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
                write_safetensors(
                    tokens_shard, filename=filename, encoding=config.encoding
                )

                n_shard += 1
                n_tokens_total = len(tokens) - remainder
                tokens_shard[:n_tokens_total] = tokens[remainder:]

            if n_tokens_total > 0:
                filename = path / f"{config.dataset}_shard_{n_shard:06d}.safetensors"
                write_safetensors(
                    tokens_shard[:n_tokens_total],
                    filename=filename,
                    encoding=config.encoding,
                )


if __name__ == "__main__":
    config = tyro.cli(DataPreparationConfig)

    if not config.write_summary_only:
        prepare(config)

    path = PATH_DATA / "train" / config.dataset
    write_summary(path=path, shards_val_idxs=config.shards_val)
