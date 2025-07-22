from pathlib import Path

from train import TrainerConfig

PATH = Path(__file__).parent.parent / "configs"

if __name__ == "__main__":
    config = TrainerConfig()
    config.write(PATH / "default.toml")
