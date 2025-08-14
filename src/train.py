import enum
import json
import logging
import os
import time
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, partial
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import optax
import tomli_w
import tomllib
import tyro
from jax import tree_util
from jax.sharding import NamedSharding, PartitionSpec
from safetensors import safe_open
from tqdm import tqdm

import wandb
from model import GPT, GPTConfig, PretrainedModels
from prepare import DatasetEnum, EncodingEnum
from utils import (
    PATH_BASE,
    PATH_DATA,
    JaxDevicesEnum,
    JaxDtypesEnum,
    flatten_pytree_with_path,
    get_checksum,
    get_random_name,
    update_leave_from_mapping,
)

TAB_WIDTH = 4

log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

InitFromEnum = enum.StrEnum(
    "InitFrom",
    {_.name: _.value for _ in PretrainedModels}
    | {"scratch": "scratch", "resume": "resume"},
)


# fmt: off
@tree_util.register_dataclass
@dataclass
class WAndBConfig:
    """WAndB logging"""

    wandb_log: bool = False  # disabled by default
    wandb_project: str = "nanogptx"
    wandb_run_name: str = field(default_factory=get_random_name)
    wandb_tags: list[str] = field(default_factory=list)


@tree_util.register_dataclass
@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

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

    @property
    def optax(self):
        """Generate optax optimizer"""
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=self.warmup_iters,
            decay_steps=self.lr_decay_iters,
            end_value=self.min_lr,
        )

        adamw = optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_scheduler,
            b1=self.beta1,
            b2=self.beta2,
            weight_decay=self.weight_decay,
        )

        return optax.chain(
            optax.clip_by_global_norm(self.grad_clip),
            adamw,
            optax.apply_every(self.gradient_accumulation_steps),
        )
# fmt: on


Batch = namedtuple("Batch", ["x", "y", "idx_shard", "idx_batches"])


@tree_util.register_dataclass
@dataclass(frozen=True)
class DatasetIndex:
    """Dataset index"""

    dataset: DatasetEnum = DatasetEnum.openwebtext
    encoding: EncodingEnum = EncodingEnum.gpt2
    suffix: Literal["train", "val"] = "train"

    @property
    def path(self):
        """Data path"""
        path = PATH_DATA / f"train/{self.dataset}-{self.encoding}"

        if not path.exists():
            message = f"Training data '{self.dataset}' with encoding '{self.encoding}' not available."
            raise ValueError(message)

        return path

    @cached_property
    def _index(self):
        """Load index file"""
        if self.path.is_dir():
            path = self.path / "summary-stats.json"
        else:
            path = self.path

        with path.open("r") as f:
            data = json.load(f)

        return data

    @property
    def stats(self):
        """Token sttats"""
        return self._index[f"token-stats-{self.suffix}"]

    @property
    def n_vocab(self):
        """Vocab size"""
        return len(self.stats)

    @property
    def n_tokens_total(self):
        """N tokens"""
        return np.sum(self.stats)

    @property
    def filenames(self):
        """Filenames anc checksums"""
        return self._index[f"shards-{self.suffix}"]

    @property
    def filenames_absolute(self):
        """Filenames and checksums"""
        return [{**_, **{"name": self.path / _["name"]}} for _ in self.filenames]


@tree_util.register_dataclass
@dataclass
class DatasetLoader:
    """Dataset loading"""

    index: DatasetIndex = DatasetIndex(
        dataset=DatasetEnum.openwebtext, encoding=EncodingEnum.gpt2
    )
    batch_size: int = 16
    block_size: int = 1024
    verify: bool = True
    devices: Sequence[JaxDevicesEnum] = tuple(JaxDevicesEnum)
    seed: int = 8273
    dtype: JaxDtypesEnum = JaxDtypesEnum.int32

    @property
    def filenames(self):
        """Absolute filenames of the loader"""
        return self.index.filenames_absolute

    @property
    def mesh_jax(self):
        """Mesh over the batch axis for distributed parallel data training"""
        return jax.make_mesh(
            axis_shapes=(len(self.devices),),
            axis_names=("batch",),
            devices=self.devices_jax,
        )

    @property
    def devices_jax(self):
        """Return actual device"""
        return [_.jax for _ in self.devices]

    @property
    def sharding_batch(self):
        """Batch sharding"""
        return NamedSharding(self.mesh_jax, PartitionSpec(*self.mesh_jax.axis_names))

    @property
    def n_shards(self):
        """Number of shards"""
        return len(self.filenames)

    def iter(self, block_size):
        random_state = np.random.default_rng(self.seed)

        while True:
            # choose random shard
            idx_shard = random_state.integers(self.n_shards)

            filename, checksum = (
                self.filenames[idx_shard]["name"],
                self.filenames[idx_shard]["checksum"],
            )

            with safe_open(filename, framework="numpy", device="cpu") as f:
                # TODO: load straight to device for zero copy
                log.info(f"Reading {filename}")
                data = f.get_tensor("tokens")

                if self.verify and checksum != get_checksum(data):
                    raise ValueError(f"Checksum does not agree for {filename}")

            # we aim for a statistical coverage here...
            for _ in range(len(data) // self.batch_size):
                max_val = len(data) - block_size
                idx_batches = random_state.integers(max_val, size=(self.batch_size,))

                x = np.stack([data[i : i + block_size] for i in idx_batches])
                y = np.stack([data[i + 1 : i + 1 + block_size] for i in idx_batches])
                yield Batch(
                    x=jax.device_put(x, self.sharding_batch),
                    y=jax.device_put(y, self.sharding_batch),
                    idx_shard=idx_shard,
                    idx_batches=idx_batches,
                )


@tree_util.register_dataclass
@dataclass
class Trainer:
    """Training configuration"""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_interval: int = 1
    eval_interval: int = 2000
    eval_iters: int = 3
    always_save_checkpoint: bool = (
        True  # if True, always save a checkpoint after each evals
    )
    show_progress: bool = True  # show progress bar
    wandb_log: bool = False

    def train(self, model, data_loader_train, data_loader_validate, rng_key):
        """Train model"""

        def loss_fn(model, batch, rng_key, is_training):
            """Loss function"""
            logits = model(batch.x, rng_key, is_training=is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch.y
            ).mean()
            return loss

        def estimate_mean_loss(model, data_loader, n_iter):
            """Estimate mean loss across multiple iters"""
            losses = []
            rng_key = jax.random.key(8273)
            # the range constrains the infinite data loader
            for _, batch in zip(range(n_iter), data_loader):
                # the key can be ignored here, because dropout is skipped
                losses.append(loss_fn(model, batch, rng_key=rng_key, is_training=True))

            return np.mean(losses)

        @partial(jax.jit, donate_argnames=("model", "opt_state"))
        def train_step(model, opt_state, batch, rng):
            """Training step"""
            grads = jax.grad(loss_fn)(model, batch, rng, is_training=True)
            updates, opt_state = self.optimizer.optax.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state

        # TODO: compute flops from train_step
        flops = model.flops(batch_size=data_loader_train.batch_size)

        data_loader_train = data_loader_train.iter(block_size=model.config.block_size)
        data_loader_validate = data_loader_validate.iter(
            block_size=model.config.block_size
        )

        # Initialize optimizer state
        opt_state = self.optimizer.optax.init(model)

        with tqdm(
            total=self.optimizer.max_iters, disable=not self.show_progress
        ) as pbar:
            fps = float("NaN")

            for n_iter, batch in zip(
                range(self.optimizer.max_iters), data_loader_train
            ):
                time_start = time.perf_counter()

                if n_iter % self.eval_interval == 0:
                    loss_train = estimate_mean_loss(
                        model, data_loader_train, n_iter=self.eval_iters
                    )
                    loss_val = estimate_mean_loss(
                        model, data_loader_validate, n_iter=self.eval_iters
                    )
                    lr = opt_state[1].hyperparams["learning_rate"]
                    pbar.set_postfix_str(
                        f"Loss train: {loss_train:.3f}, Loss val: {loss_val:.3f}, lr: {lr:.5f}, tfps: {fps/1e12:.3f}"
                    )

                    if self.wandb_log:
                        wandb.log(
                            {
                                "iter": n_iter,
                                "loss-train": loss_train,
                                "loss-val": loss_val,
                                "lr": lr,
                                "shard": batch.idx_shard,
                                "fps": fps,
                            }
                        )

                rng_key = jax.random.fold_in(rng_key, n_iter)
                model, opt_state = train_step(model, opt_state, batch, rng_key)
                fps = flops.per_iter / (time.perf_counter() - time_start)
                pbar.update(1)

        return model


@tree_util.register_dataclass
@dataclass
class Config:
    """General config"""

    init_from: InitFromEnum = InitFromEnum.scratch
    seed: int = 9283  # Random seed
    devices: Sequence[JaxDevicesEnum] = tuple(JaxDevicesEnum)
    dtype: JaxDtypesEnum = JaxDtypesEnum.float32
    training: Trainer = field(default_factory=Trainer)
    loading: DatasetLoader = field(default_factory=DatasetLoader)
    model: GPTConfig = field(default_factory=GPTConfig)
    logging: WAndBConfig = field(default_factory=WAndBConfig)
    _key = None

    def __post_init__(self):
        # sync arguments after init
        self.training.wandb_log = self.logging.wandb_log

        # TODO: which gets precendence here data or model definition?
        self.loading.devices = self.devices

    @property
    def mesh_jax(self):
        """Mesh over the batch axis for distributed parallel data training"""
        return jax.make_mesh(
            axis_shapes=(len(self.devices),),
            axis_names=("batch",),
            devices=self.devices_jax,
        )

    @property
    def sharding_replicated(self):
        """Replicated sharding"""
        return NamedSharding(self.mesh_jax, PartitionSpec())

    @property
    def devices_jax(self):
        """Return actual device"""
        return [_.jax for _ in self.devices]

    @property
    def rng_key(self) -> jax.Array:
        """Generate random key for initialization"""
        if self._key is None:
            self._key = jax.random.PRNGKey(self.seed)

        # in general state based key generation is not a good idea in Jax!
        # however the config class never(!) crosses any jit and function transform
        # boundaries. So it is safe to use it here.
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @property
    def loading_val(self):
        """Validation dataset"""
        index = replace(self.loading.index, suffix="val")
        return replace(self.loading, index=index)

    @classmethod
    def read(cls, path: str):
        """Read configuration from file"""
        path = Path(path)
        log.info(f"Reading configuration from {path}")

        if path.suffix == ".safetensors":
            with safe_open(path, framework="numpy") as f:
                return cls.from_safetensors_meta(f.metadata())

        with path.open("rb") as f:
            data = flatten_pytree_with_path(tomllib.load(f))

        return jax.tree.map_with_path(
            update_leave_from_mapping(data, use_default_if_missing=True), cls()
        )

    @classmethod
    def from_safetensors_meta(cls, data):
        """Re-create config from safetensors meta"""
        return jax.tree.map_with_path(
            update_leave_from_mapping(data, use_default_if_missing=True), cls()
        )

    def to_safetensors_meta(self):
        """Create safetensors meta"""
        data = flatten_pytree_with_path(asdict(self), parse_type=str)
        return data

    def write(self, path: str):
        """Write configuration to file"""
        log.info(f"Writing configuration to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            tomli_w.dump(asdict(self), f)

    def __str__(self):
        return tomli_w.dumps(asdict(self), indent=TAB_WIDTH)


def get_configs():
    """Get configs from config folder"""
    filenames = (PATH_BASE / "configs").glob("*.toml")

    configs = {}

    # TODO: parse desription from the first line of the toml file
    for filename in filenames:
        configs[filename.stem] = (
            f"Default configuration from {filename.name}",
            Config.read(filename),
        )

    return configs


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(get_configs())

    if config.logging.wandb_log:
        run = wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run_name,
            tags=config.logging.wandb_tags,
            config=asdict(config),
        )

    data_loader_train = config.loading
    log.info(f"Training dataset has {data_loader_train.index.n_tokens_total} tokens.")

    data_loader_validate = config.loading_val
    log.info(
        f"Validation dataset has {data_loader_validate.index.n_tokens_total} tokens."
    )

    config.model.vocab_size = config.loading.index.n_vocab

    if config.init_from == InitFromEnum.scratch:
        model = GPT.from_config(config.model)
    elif config.init_from == InitFromEnum.resume:
        candidates = (PATH_DATA / "checkpoints").glob("*.safetensors")
        latest = max(candidates, key=os.path.getctime)
        model = GPT.read(latest)
    else:
        model = GPT.from_pretrained(config.init_from)

    log.info(f"{model.info()}")
    spec = {"device": config.sharding_replicated, "dtype": config.dtype.jax}

    model = config.training.train(
        model=model.init(rng_key=config.rng_key, **spec),
        data_loader_train=data_loader_train,
        data_loader_validate=data_loader_validate,
        rng_key=config.rng_key,
    )

    filename = (
        PATH_DATA
        / "checkpoints"
        / f"model-{config.logging.wandb_run_name}-final.safetensors"
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    model.write(filename, metadata=config.to_safetensors_meta())
