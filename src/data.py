import logging
import re
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tiktoken
import tqdm
import tyro
from jax import numpy as jnp
from safetensors.numpy import save_file

log = logging.getLogger(__file__)

PATH = Path(__file__).parent


class EncodingEnum:
    """Tokenizer"""

    gpt2 = "gpt2"


DTYPES = {EncodingEnum.gpt2: np.uint16}


# TODO: write acuumulated token stats as JSON


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

    shard_size: int = 1e8  # Size of each data shard in the output files, in tokens
    encoding: EncodingEnum = EncodingEnum.gpt2
    dataset: DatasetEnum = DatasetEnum.shakespeare
    n_process: int = max(1, cpu_count() - 2)
    chunksize: int = 16
    show_progress: bool = True


def prepocess(sequence):
    """Preprocess sequence"""
    return re.sub("\n\n\n+", "\n\n", sequence.decode("utf-8")).strip()


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


def read_tar(filename, member):
    """Read tar file"""
    with tarfile.open(filename, "r") as tar:
        for member in tar:
            yield tar.extractfile(member.name).read()


def read_parquet():
    """Read a parquet file"""
    ...


def split(tokens, fraction=0.9):
    """Split tokens into train and test"""
    length = len(tokens)
    return [tokens[fraction * length :], tokens[: fraction * length]]


def write_safetensors(tokens, filename):
    """Write safetensors file"""
    log.info(f"Writing {filename}")
    save_file({"tokens": tokens}, filename)


PIPELINES = {
    DatasetEnum.shakespeare: [read_txt, prepocess, tokenize, split],
    DatasetEnum.openwebtext: [read_tar, prepocess, tokenize],
}


def prepare(config):
    """Prepare and tokenize data"""
    pipeline = PIPELINES[config.dataset]

    filenames = (PATH / f"data/download/{config.dataset}").glob("*")

    with tqdm(
        total=config.shard_size,
        disable=not config.show_progress,
        unit="tokens",
    ) as pbar:
        with Pool(config.n_process) as pool:
            n_shard, n_tokens_total = 0, 0
            tokens_shard = np.empty((config.n_shards,), dtype=DTYPES[config.encoding])

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

                path = PATH / "data" / "train" / config.dataset
                filename = path / f"{config.dataset}_{split}_{n_shard:06d}.safetensors"
                write_safetensors(tokens_shard, filename=filename)

                n_shard += 1
                n_tokens_total = len(tokens) - remainder
                tokens_shard[:n_tokens_total] = tokens[remainder:]


if __name__ == "__main__":
    config = tyro.cli(DataPreparationConfig)
    prepare(config)
