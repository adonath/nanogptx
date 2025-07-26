from pathlib import Path

from train import Config

PATH = Path(__file__).parent.parent / "configs"

if __name__ == "__main__":
    config = Config()
    config.write(PATH / "default.toml")
