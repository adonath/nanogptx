import enum
import json
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from model import GPT, GPTConfig, PretrainedModels
from safetensors import safe_open
from utils import PATH_DATA, Config, JaxDevicesEnum, asdict_str

log = logging.getLogger(__file__)

InitFrom = enum.Enum(
    "InitFrom", {_.name: _.value for _ in PretrainedModels} | {"scratch": "scratch"}
)


# fmt: off
@dataclass(kw_only=True)
class IOConfig:
    """Training configuration"""
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        True  # if True, always save a checkpoint after each evals
    )
    init_from: InitFrom = InitFrom.scratch
    dataset: Literal["openwebtext", "shakespeare"] = "openwebtext"
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size


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
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# fmt: on


Batch = namedtuple("Batch", ["x", "y"])


@dataclass
class DatasetLoader:
    """Dataset loader

    This dataset load supports array sharding for SPMD parallelism.


    To be used as:

    for batch in loader:
        x, y = batch.x, batch.y


    """

    batch_size: int = 12
    device: JaxDevicesEnum = list(JaxDevicesEnum)[0]
    block_size: int = 1024
    seed: int = 78127
    dtype: jnp.dtype = jnp.int64
    filenames: list[str] = field(default_factory=[])

    @classmethod
    def read(cls, path, key="shards-train", **kwargs):
        """Read from json summary file"""

        with path.open("r") as f:
            data = json.load(f)

        filenames = [path.parent / _ for _ in data[key]]

        return cls(filenames=filenames, **kwargs)

    @property
    def n_shards(self):
        """Number of shards"""
        return len(self.filenames)

    def __iter__(self):
        random_state = np.random.default_rng(self.seed)

        while True:
            # choose random shard
            shard_idx = random_state.integers(self.n_shards)

            # TODO: load straight to device for zero copy
            filename = self.filenames[shard_idx]
            with safe_open(filename, framework="numpy", device="cpu") as f:
                log.info(f"Reading {filename}")
                data = f.get_tensor("tokens")

            # we aim for a statistical coverage here...
            for idx in range(len(data) // (self.batch_size * self.block_size)):
                max_val = len(data) - self.block_size
                ix = random_state.integers(max_val, size=(self.batch_size,))

                spec = {"device": self.device.value, "dtype": self.dtype}

                x = jnp.stack(
                    [jnp.asarray(data[i : i + self.block_size], **spec) for i in ix]
                )
                y = jnp.stack(
                    [
                        jnp.asarray(data[i + 1 : i + 1 + self.block_size], **spec)
                        for i in ix
                    ]
                )
                yield Batch(x=x, y=y)


@dataclass
class GPTTrainer:
    """GPT trainer"""

    optimizer: optax.GradientTransformation = field(default=optax.adamw)
    max_iters: int = 60_000
    seed: int = 71363

    @classmethod
    def from_config(cls, config):
        """Create from config"""
        warmup_steps = (
            config.warmup_iters if config.init_from == InitFrom.scratch else 0
        )

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=config.lr_decay_iters,
            end_value=config.min_lr,
        )

        adamw = optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_scheduler,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            adamw,
            optax.apply_every(config.gradient_accumulation_steps),
        )

        return cls(optimizer=optimizer, seed=config.seed, max_iters=config.max_iters)

    def train(self, model, data_loader_train, data_loader_validate):
        """Train model"""

        def loss_fn(model, batch, rng_key):
            """Loss fucntion"""
            logits = model(batch.x, rng_key, is_training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch.y
            ).mean()
            return loss

        @jax.jit
        def train_step(model, opt_state, batch, rng):
            """Training step"""
            loss, grads = jax.value_and_grad(loss_fn)(model, batch, rng)
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            params = optax.apply_updates(model, updates)
            return params, opt_state, loss

        # Initialize optimizer state
        opt_state = self.optimizer.init(model)

        rng = jax.random.key(self.seed)

        n_iter = 0

        for batch in data_loader_train:  # x should be an iterable of batches
            rng, subkey = jax.random.split(rng)
            model, opt_state, loss = train_step(model, opt_state, batch, subkey)
            # Optionally log loss, save checkpoints, etc.
            print(f"N iter {n_iter}, Loss: {loss}")

            if n_iter == self.max_iters:
                break

            n_iter += 1

        return model


@dataclass(kw_only=True)
class GlobalConfig(Config, IOConfig, GPTConfig, WAndBConfig, OptimizerConfig):
    """Global trainig config"""

    ...


if __name__ == "__main__":
    config = tyro.cli(GlobalConfig)

    model = GPT.from_config(config)

    trainer = GPTTrainer.from_config(config)

    path_json = PATH_DATA / "train" / config.dataset / "summary-stats.json"
    data_loader_train = DatasetLoader.read(path_json, key="shards-train")
    data_loader_validate = DatasetLoader.read(path_json, key="shards-val")

    model = trainer.train(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_validate=data_loader_validate,
    )

    filename = (
        PATH_DATA / "checkpoints" / f"model-{config.wandb_run_name}-final.safetensors"
    )

    model.write(filename, metadata=asdict_str(config))
