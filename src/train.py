import enum
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from model import GPT, GPTConfig, PretrainedModels
from utils import Config, JaxDevicesEnum

InitFrom = enum.Enum(
    "InitFrom", {_.name: _.value for _ in PretrainedModels} | {"scratch": "scratch"}
)


DATASET_PATHS = {
    "openwebtext": "",
    "shakespeare": "",
}


@dataclass(kw_only=True)
class IOConfig:
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
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )


@dataclass
class DatasetLoader:
    """Dataset loader"""

    batch_size: int = 12
    device: JaxDevicesEnum = list(JaxDevicesEnum)[0]
    block_size: int = 1024
    seed: int = 78127
    data: Any | None = None
    dtype: jnp.dtype = jnp.int64

    @classmethod
    def read(cls, path, **kwargs):
        """Read from path"""
        data = np.memmap(path, dtype=np.uint16, mode="r")
        return cls(data=data, **kwargs)

    def __iter__(self):
        ix = np.random.randint(
            len(self.data) - self.block_size, size=(self.batch_size,)
        )
        x = jnp.stack(
            [
                jnp.asarray(
                    self.data[i : i + self.block_size],
                    dtype=self.dtype,
                    device=self.device,
                )
                for i in ix
            ]
        )
        y = jnp.stack(
            [
                jnp.asarray(
                    self.data[i + 1 : i + 1 + self.block_size],
                    dtype=self.dtype,
                    device=self.device,
                )
                for i in ix
            ]
        )
        return x, y


@dataclass
class GPTTrainer:
    """GPT trainer"""

    optimizer: optax.GradientTransformation = field(default=optax.adamw)
    n_epochs: int = 1_000

    @classmethod
    def from_config(cls, config):
        """Create from config"""

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_iters
            if config.init_from == InitFrom.scratch
            else 0,
            decay_steps=config.lr_decay_iters - config.iter_num,
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

        return cls(optimizer=optimizer)

    def train(self, model, data_loader_train, data_loader_validate):
        """Train model"""

        def loss_fn(model, batch, rng_key):
            """Loss fucntion"""
            inputs, targets = batch
            logits = model(inputs, rng_key, is_training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
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

        rng = jax.random.PRNGKey(0)
        for epoch in range(self.n_epochs):
            for batch in data_loader_train:  # x should be an iterable of batches
                rng, subkey = jax.random.split(rng)
                params, opt_state, loss = train_step(model, opt_state, batch, subkey)
                # Optionally log loss, save checkpoints, etc.
                print(f"Epoch {epoch}, Loss: {loss}")

        return params


@dataclass(kw_only=True)
class GlobalConfig(Config, IOConfig, GPTConfig, WAndBConfig, OptimizerConfig):
    """Global trainig config"""

    ...


if __name__ == "__main__":
    config = tyro.cli(GlobalConfig)

    model = GPT.from_config(config)

    trainer = GPTTrainer.from_config(config)

    data_loader_train = DatasetLoader.read(DATASET_PATHS[config.dataset])
    data_loader_validate = DatasetLoader.read(DATASET_PATHS[config.dataset])

    trainer.train(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_validate=data_loader_validate,
    )
