import logging
import tarfile
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path

import requests
import tyro
from model import PretrainedModels

from data import DatasetEnum

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data"

N_THREADS_DEAFULT = cpu_count() // 2


MODEL_URLS = {
    PretrainedModels.gpt2_medium: "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
    PretrainedModels.gpt2_large: "https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors",
    PretrainedModels.gpt2_xl: "https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",
    PretrainedModels.gpt2: "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
}

# fmt: off
DATA_URLS = {
    DatasetEnum.shakespeare: ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",],
    DatasetEnum.openwebtext: [f"https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{idx:02d}.tar" for idx in range(21)],
    DatasetEnum.tinystories: ["https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",],
}
# fmt: on


def extract_tar_and_remove(archive_path, extraction_path):
    """Extract and remove a tar archive"""
    mode = "r"

    if archive_path.name.endswith("gz"):
        mode += ":gz"

    log.info(f"Reading {archive_path}")
    with tarfile.open(archive_path, mode) as tar:
        log.info(f"Extracting to {extraction_path}")
        for member in tar.getmembers():
            if member.isreg():
                member.name = Path(member.name).name
                tar.extract(member, extraction_path)

    log.info(f"Deleting {archive_path}")
    archive_path.unlink()


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
        url = MODEL_URLS[model]
        path = DATA_PATH / "models" / model.value / Path(url).name
        args.append((url, path))

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
