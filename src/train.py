import json
import logging
import os
import time
from collections import namedtuple
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, partial
from itertools import cycle
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import optax
import tomli_w
import tomllib
import tyro
from dacite import Config as DaciteConfig
from dacite import from_dict
from jax import numpy as jnp
from jax import tree_util
from safetensors import safe_open
from tqdm import tqdm

import wandb
from model import GPT, Axis, GPTConfig
from utils import (
    FLOPS_UNIT,
    PATH_BASE,
    PATH_DATA,
    DatasetEnum,
    EncodingEnum,
    InitFromEnum,
    get_checksum,
    get_random_name,
    sizeof_fmt,
)
from utils_jax import (
    JaxDevicesEnum,
    JaxDtypesEnum,
    JaxFloatDtypesEnum,
    JaxIntDtypesEnum,
    ShardingConfig,
    flatten_pytree_with_path,
    update_leave_from_mapping,
)

TAB_WIDTH = 4
DACITE_CAST = [
    InitFromEnum,
    JaxFloatDtypesEnum,
    DatasetEnum,
    JaxDevicesEnum,
    JaxDtypesEnum,
    JaxIntDtypesEnum,
    EncodingEnum,
    int,
    str,
    float,
    bool,
]
DACITE_CONFIG = DaciteConfig(cast=DACITE_CAST, strict=True)

log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


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


def stack_batches(batches):
    """Stack batches"""
    return Batch(
        x=jnp.concatenate([batch.x for batch in batches], axis=Axis.batch),
        y=jnp.concatenate([batch.y for batch in batches], axis=Axis.batch),
        idx_shard=None,
        idx_batches=None,
    )


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
    sharding: ShardingConfig = ShardingConfig()
    seed: int = 8273
    dtype: JaxIntDtypesEnum = JaxIntDtypesEnum.int32
    randomize_shards: bool = True

    @property
    def filenames(self):
        """Absolute filenames of the loader"""
        return self.index.filenames_absolute

    @property
    def n_shards(self):
        """Number of shards"""
        return len(self.filenames)

    def iter(self, block_size):
        random_state = np.random.default_rng(self.seed)

        shards = np.arange(self.n_shards)

        if self.randomize_shards:
            shards = random_state.permutation(shards)

        for idx_shard in cycle(shards):
            filename, checksum = (
                self.filenames[idx_shard]["name"],
                self.filenames[idx_shard]["checksum"],
            )

            with safe_open(filename, framework="numpy", device="cpu") as f:
                # TODO: load straight to device for zero copy, see also https://github.com/huggingface/safetensors/issues/636
                log.info(f"Reading {filename}")
                data = f.get_tensor("tokens")

                if self.verify and checksum != get_checksum(data):
                    raise ValueError(f"Checksum does not agree for {filename}")

            # we aim for a statistical coverage here...
            for _ in range(len(data) // self.batch_size // block_size):
                max_val = len(data) - block_size
                idx_batches = random_state.integers(max_val, size=(self.batch_size,))

                x = jnp.stack([data[i : i + block_size] for i in idx_batches])
                y = jnp.stack([data[i + 1 : i + 1 + block_size] for i in idx_batches])
                yield Batch(
                    x=jax.device_put(x, self.sharding.jax),
                    y=jax.device_put(y, self.sharding.jax),
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
    total_batch_size: int = 256  # total batch size in units of tokens

    def train(self, model, data_loader_train, data_loader_validate, rng_key):
        """Train model"""

        log.info(
            f"Using {self.optimizer.gradient_accumulation_steps} gradient accumulation steps."
        )

        def loss_fn(model, batch, rng_key, is_training):
            """Loss function"""
            logits = model(batch.x, rng_key, is_training=is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch.y
            ).mean()
            return loss

        @partial(jax.jit, static_argnames=("n_iter", "data_loader"))
        def estimate_mean_loss(model, data_loader, n_iter):
            """Estimate mean loss across multiple iters"""
            batch = stack_batches(
                [batch for _, batch in zip(range(n_iter), data_loader)]
            )
            return loss_fn(
                model, batch, rng_key=jax.random.key(8273), is_training=False
            )

        @partial(jax.jit, donate_argnames=("model", "opt_state"))
        def train_step(model, opt_state, batch, rng):
            """Training step"""
            value, grads = jax.value_and_grad(loss_fn)(
                model, batch, rng, is_training=True
            )
            updates, opt_state = self.optimizer.optax.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, value

        flops = model.flops(
            batch_size=data_loader_train.batch_size,
            dtype=data_loader_train.dtype,
            sharding=data_loader_train.sharding.jax,
        )

        data_loader_train = data_loader_train.iter(block_size=model.config.block_size)
        data_loader_validate = data_loader_validate.iter(
            block_size=model.config.block_size
        )

        # Initialize optimizer state
        opt_state = self.optimizer.optax.init(model)

        with tqdm(
            total=self.optimizer.max_iters, disable=not self.show_progress
        ) as pbar:
            for n_iter, batch in zip(
                range(self.optimizer.max_iters), data_loader_train
            ):
                time_start = time.perf_counter()

                rng_key = jax.random.fold_in(rng_key, n_iter)
                model, opt_state, loss_train = jax.block_until_ready(
                    train_step(model, opt_state, batch, rng_key)
                )

                # TODO: compute the actual flops from train_step for now just use 1/2 for fwd / bkw ratio
                dt = time.perf_counter() - time_start

                mfu = 3 * flops.per_iter / FLOPS_UNIT / dt
                tps = flops.tokens_per_iter / dt

                if n_iter % self.eval_interval == 0:
                    loss_val = estimate_mean_loss(
                        model, data_loader_validate, n_iter=self.eval_iters
                    )
                    lr = float(opt_state[1].hyperparams["learning_rate"])
                    pbar.set_postfix_str(
                        f"Loss train: {loss_train.item():.3f}, Loss val: {loss_val.item():.3f}, lr: {lr:.5f}, mfu: {(mfu):.0%}, tok/s: {sizeof_fmt(tps, system='decimal')}"
                    )

                    if self.wandb_log:
                        wandb.log(
                            {
                                "iter": n_iter,
                                "loss-train": loss_train,
                                "loss-val": loss_val,
                                "lr": lr,
                                "shard": batch.idx_shard,
                                "mfu": mfu,
                                "tok/s": tps,
                            }
                        )

                pbar.update(1)

        return model


@tree_util.register_dataclass
@dataclass
class Config:
    """General config"""

    init_from: InitFromEnum = InitFromEnum.scratch
    seed: int = 9283  # Random seed
    sharding: ShardingConfig = ShardingConfig(partition=())
    dtype: JaxFloatDtypesEnum = JaxFloatDtypesEnum.float32
    training: Trainer = field(default_factory=Trainer)
    loading: DatasetLoader = field(default_factory=DatasetLoader)
    model: GPTConfig = field(default_factory=GPTConfig)
    logging: WAndBConfig = field(default_factory=WAndBConfig)
    _key = None

    def __post_init__(self):
        # sync arguments after init
        self.training.wandb_log = self.logging.wandb_log

        # calculate the gradient accumulation based on total batch size
        tokens_per_iter = self.model.block_size * self.loading.batch_size
        accum_steps = self.training.total_batch_size // tokens_per_iter
        self.training.optimizer.gradient_accumulation_steps = max(accum_steps, 1)

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
            data = tomllib.load(f)

        return from_dict(data_class=cls, data=data, config=DACITE_CONFIG)

    @classmethod
    def from_safetensors_meta(cls, data):
        """Re-create config from safetensors meta"""
        # This requires some gymnatsics here...
        update = update_leave_from_mapping(data, use_default_if_missing=True)
        data = jax.tree.map_with_path(update, asdict(cls()))

        return from_dict(data_class=cls, data=data, config=DACITE_CONFIG)

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
        try:
            configs[filename.stem] = (
                f"Default configuration from {filename.name}",
                Config.read(filename),
            )
        except ValueError as e:
            log.warning(f"Error in file '{filename.name}', {e}")

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

    log.info(f"Using devices {config.sharding.jax.device_set}")

    if config.init_from == InitFromEnum.scratch:
        model = GPT.from_config(config.model)
    elif config.init_from == InitFromEnum.resume:
        candidates = (PATH_DATA / "checkpoints").glob("*.safetensors")
        latest = max(candidates, key=os.path.getctime)
        model = GPT.read(latest)
    else:
        model = GPT.from_pretrained(config.init_from)

    spec = {"device": config.sharding.jax, "dtype": config.dtype.jax}
    model = model.init(rng_key=config.rng_key, **spec)

    log.info(f"{model.info()}")

    model = config.training.train(
        model=model,
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
