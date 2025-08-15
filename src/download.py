import logging
import tarfile
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path

import requests
import tyro

from model import PretrainedModels
from prepare import DatasetEnum

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data"

N_THREADS_DEAFULT = cpu_count() // 2

MODEL_URLS = {
    PretrainedModels.gpt2_medium: ["https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",  "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json"],
    PretrainedModels.gpt2_large: ["https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors", "https://huggingface.co/openai-community/gpt2-large/resolve/main/config.json"],
    PretrainedModels.gpt2_xl: ["https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",  "https://huggingface.co/openai-community/gpt2-xl/resolve/main/config.json"],
    PretrainedModels.gpt2: ["https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors", "https://huggingface.co/openai-community/gpt2/resolve/main/config.json"],
}

# fmt: off
DATA_URLS = {
    DatasetEnum.shakespeare: ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",],
    DatasetEnum.openwebtext: [f"https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{idx:02d}.tar" for idx in range(21)],
    DatasetEnum.tinystories: ["https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",],
    DatasetEnum.pile_uncopyrighted: [f"https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/{idx:02d}.jsonl.zst" for idx in range(30)],
    DatasetEnum.fineweb_10b: [f"https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/{idx:03d}_00000.parquet" for idx in range(15)],
    DatasetEnum.fineweb_100b: [f"https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/100BT/{idx_a:03d}_{idx_b:05d}.parquet" for idx_a, idx_b in product(range(15), range(10))],
    DatasetEnum.fineweb_edu_10b: [f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{idx:03d}_00000.parquet" for idx in range(14)],
    DatasetEnum.fineweb_edu_100b: [f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{idx_a:03d}_{idx_b:05d}.parquet" for idx_a, idx_b in  product(range(14), range(10))],
}
# fmt: on


def extract_tar_and_remove(archive_path, extraction_path):
    """Extract and remove a tar archive"""
    mode = "r"

    if archive_path.name.endswith("gz"):
        mode += ":gz"

    log.info(f"Reading {archive_path}")
    with tarfile.open(archive_path, mode) as tar:
        for member in tar.getmembers():
            if not member.isreg():
                continue

            log.info(f"Extracting {member.name} to {extraction_path}")
            member.name = Path(member.name).name
            tar.extract(member, extraction_path)

    log.info(f"Deleting {archive_path}")
    archive_path.unlink()


def decompress_zst_file(input_filepath, output_filepath):
    """
    Decompresses a .zst file to a specified output file.

    Args:
        input_filepath (str): Path to the .zst file to decompress.
        output_filepath (str): Path where the decompressed data will be written.
    """
    import zstandard

    dctx = zstandard.ZstdDecompressor()

    with open(input_filepath, "rb") as ifh:
        with open(output_filepath, "wb") as ofh:
            log.info(f"Extracting {input_filepath} to {output_filepath}")
            dctx.copy_stream(ifh, ofh)


def download_file(url, path):
    """Download file"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        log.info(f"{path} already exists, skipping download!")
        return path

    log.info(f"Downloading from {url}")
    response = requests.get(url, params={"download": True})

    log.info(f"Saving to {path}")
    with path.open("wb") as file:
        file.write(response.content)

    if path.name.endswith((".tar", ".tar.gz")):
        extract_tar_and_remove(path, path.parent)

    if path.name.endswith(".zst"):
        decompress_zst_file(path, path.with_suffix(".jsonl"))

    return path


def download(
    model: PretrainedModels | None = None,
    dataset: DatasetEnum | None = None,
    n_threads: int = N_THREADS_DEAFULT,
):
    """Download GPT2 weights from Huggingface

    Parameters
    ----------
    key : str
        Model identifier
    """
    args = []

    if model is not None:
        model = PretrainedModels(model)
        urls = MODEL_URLS[model]
        paths = [
            DATA_PATH / "models" / model.value / Path(url).name for url in urls
        ]
        args.extend(zip(urls, paths))

    if dataset is not None:
        dataset = DatasetEnum(dataset)
        urls = DATA_URLS[dataset]
        paths = [
            DATA_PATH / "download" / dataset.value / Path(url).name for url in urls
        ]
        args.extend(zip(urls, paths))

    with ThreadPool(n_threads) as pool:
        result = pool.starmap_async(download_file, args)
        result.wait()


if __name__ == "__main__":
    path = tyro.cli(download)
