import logging
from enum import StrEnum
from pathlib import Path

import requests
import tyro
from model import PretrainedModels

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data"


class DatasetEnum(StrEnum):
    """Dataset enum"""

    shakespeare = "shakespeare"
    openwebtext = "openwebtext"


MODEL_URLS = {
    PretrainedModels.gpt2_medium: "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
    PretrainedModels.gpt2_large: "https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors",
    PretrainedModels.gpt2_xl: "https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",
    PretrainedModels.gpt2: "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
}

DATA_URLS = {
    DatasetEnum.shakespeare: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download_weights(
    model: PretrainedModels | None = None, dataset: DatasetEnum | None = None
):
    """Download GPT2 weights from Huggingface

    Parameters
    ----------
    key : str
        Model identifier
    """
    if model is not None:
        model = PretrainedModels(model)
        url = MODEL_URLS[model]
        path = DATA_PATH / "models" / model.value / Path(url).name

    if dataset is not None:
        dataset = DatasetEnum(dataset)
        url = DATA_URLS[dataset]
        path = DATA_PATH / "download" / dataset.value / Path(url).name

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        log.info(f"{path} already exists, skipping download!")
        return path

    log.info(f"Downloading from {url}")
    response = requests.get(url, params={"download": True})

    log.info(f"Saving to {path}")
    with open(path, mode="wb") as file:
        file.write(response.content)

    return path


if __name__ == "__main__":
    path = tyro.cli(download_weights)
