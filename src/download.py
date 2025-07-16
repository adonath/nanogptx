import logging
from pathlib import Path

import requests
import tyro
from model import PretrainedModels

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data"

MODEL_URLS = {
    PretrainedModels.gpt2_medium: "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
    PretrainedModels.gpt2_large: "https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors",
    PretrainedModels.gpt2_xl: "https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",
    PretrainedModels.gpt2: "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
}

DATA_URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download_weights(key: PretrainedModels):
    """Download GPT2 weights from Huggingface"""
    key = PretrainedModels(key)
    url = MODEL_URLS[key]

    path = DATA_PATH / "models" / key.value / Path(url).name
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
