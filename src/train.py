import enum
from dataclasses import dataclass
from typing import Literal

import tyro
from model import GPTConfig, PretrainedModels
from utils import Config

InitFrom = enum.Enum(
    "InitFrom", {_.name: _.value for _ in PretrainedModels} | {"scratch": "scratch"}
)


@dataclass(kw_only=True)
class TrainingConfig(Config):
    """Training configuration"""

    out_dir: str = "data/models/checkpoints"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        True  # if True, always save a checkpoint after each evals
    )
    init_from: InitFrom = InitFrom.scratch
    dataset: Literal["openwebtext", "shakespeare"] = "openwebtext"
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = (
        12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )


@dataclass(kw_only=True)
class WAndBConfig:
    """WAndB logging"""

    wandb_log: bool = False  # disabled by default
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"  # 'run' + str(time.time())


@dataclass(kw_only=True)
class OptimizerConfig:
    """Optimizer config"""

    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )


@dataclass
class GPTTrainer:
    """GPT trainer"""

    lr: float = 0.001
    batch_size: int = 8
    epoches: int = 10
    out_dir: str = "out"

    def train(self, x, moodel):
        pass


@dataclass(kw_only=True)
class GlobalConfig(TrainingConfig, GPTConfig, WAndBConfig, OptimizerConfig):
    """Global trainig config"""

    ...


if __name__ == "__main__":
    config = tyro.cli(GlobalConfig)
