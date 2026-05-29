import logging
import tarfile
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path

import requests
import tyro

from utils import PATH_DATA, DatasetEnum, PretrainedModels

log = logging.getLogger(__name__)

N_THREADS_DEFAULT = cpu_count() // 2
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB


MODEL_URLS = {
    PretrainedModels.gpt2_medium: [
        "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
        "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json",
    ],
    PretrainedModels.gpt2_large: [
        "https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors",
        "https://huggingface.co/openai-community/gpt2-large/resolve/main/config.json",
    ],
    PretrainedModels.gpt2_xl: [
        "https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",
        "https://huggingface.co/openai-community/gpt2-xl/resolve/main/config.json",
    ],
    PretrainedModels.gpt2: [
        "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
        "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
    ],
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
    DatasetEnum.fineweb_edu_100b: [f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/{idx_a:03d}_{idx_b:05d}.parquet" for idx_a, idx_b in product(range(14), range(10))],
    DatasetEnum.hellaswag: [
        "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/test-00000-of-00001.parquet",
        "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/train-00000-of-00001.parquet",
        "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet",
    ]
}
# fmt: on


def extract_tar_and_remove(archive_path, extraction_path):
    """Extract and remove a tar archive"""
    mode = "r"

    if archive_path.name.endswith("gz"):
        mode += ":gz"

    log.info("Reading %s", archive_path)
    with tarfile.open(archive_path, mode) as tar:
        for member in tar.getmembers():
            if not member.isreg():
                continue

            log.info("Extracting %s to %s", member.name, extraction_path)
            member.name = Path(member.name).name
            tar.extract(member, extraction_path)

    log.info("Deleting %s", archive_path)
    archive_path.unlink()


def decompress_zst_file(input_filepath, output_filepath):
    """Decompress a .zst file to output_filepath"""
    import zstandard

    dctx = zstandard.ZstdDecompressor()

    with open(input_filepath, "rb") as ifh, open(output_filepath, "wb") as ofh:
        log.info("Extracting %s to %s", input_filepath, output_filepath)
        dctx.copy_stream(ifh, ofh)


def download_file(url, path):
    """Download file"""
    if path.exists():
        log.info("%s already exists, skipping download!", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading from %s", url)
    with requests.get(url, params={"download": True}, stream=True) as response:
        response.raise_for_status()
        log.info("Saving to %s", path)
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)

    if path.name.endswith((".tar", ".tar.gz")):
        extract_tar_and_remove(path, path.parent)

    if path.name.endswith(".zst"):
        decompress_zst_file(path, path.with_suffix(".jsonl"))

    return path


def download(
    model: PretrainedModels | None = None,
    dataset: DatasetEnum | None = None,
    n_threads: int = N_THREADS_DEFAULT,
):
    """Download model weights or a dataset from Huggingface

    Parameters
    ----------
    model : PretrainedModels | None
        Pretrained model identifier; weights and config are fetched to
        ``data/models/<model>``.
    dataset : DatasetEnum | None
        Dataset identifier; shards are fetched to ``data/download/<dataset>``.
    n_threads : int
        Number of parallel download workers.
    """
    args = []

    if model is not None:
        model = PretrainedModels(model)
        urls = MODEL_URLS[model]
        paths = [PATH_DATA / "models" / model.value / Path(url).name for url in urls]
        args.extend(zip(urls, paths))

    if dataset is not None:
        dataset = DatasetEnum(dataset)
        urls = DATA_URLS[dataset]
        paths = [
            PATH_DATA / "download" / dataset.value / Path(url).name for url in urls
        ]
        args.extend(zip(urls, paths))

    with ThreadPool(n_threads) as pool:
        # `.get()` re-raises any exception from the worker threads; `.wait()`
        # would silently swallow them.
        pool.starmap_async(download_file, args).get()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(download)
