import json
import logging
import os
import time
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
import wandb
from dacite import Config as DaciteConfig
from dacite import UnexpectedDataError, from_dict
from jax import numpy as jnp
from jax.tree_util import register_dataclass
from safetensors import safe_open
from safetensors.flax import save_file
from tqdm import tqdm

from evaluate import ModelEvaluator, load_hellaswag_examples
from model import DOT_PRODUCT_ATTENTION, GPT, GPTConfig
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
    join_path,
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
    Path,
]
DACITE_CONFIG = DaciteConfig(cast=DACITE_CAST, strict=True)
PROFILER_OPTIONS = jax.profiler.ProfileOptions()
# PROFILER_OPTIONS.python_tracer_level = 0
# PROFILER_OPTIONS.host_tracer_level = 0

log = logging.getLogger(__name__)


@dataclass
class WAndBConfig:
    """WAndB logging"""

    wandb_log: bool = False  # disabled by default
    wandb_project: str = "nanogptx"
    wandb_run_name: str = field(default_factory=get_random_name)
    wandb_tags: list[str] = field(default_factory=list)


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
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )

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

        def mask_fn(params):
            """Select 2D+ parameters for decay (skip biases and norm scales)"""
            return jax.tree.map(lambda x: x.ndim >= 2, params)

        adamw = optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_scheduler,
            b1=self.beta1,
            b2=self.beta2,
            weight_decay=self.weight_decay,
            mask=mask_fn,
        )

        return optax.chain(
            optax.clip_by_global_norm(self.grad_clip),
            adamw,
            optax.apply_every(self.gradient_accumulation_steps),
        )

    @staticmethod
    def _inject_state_index(opt_state):
        """Index of the inject_hyperparams state in the optimizer chain"""
        for i, state in enumerate(opt_state):
            if hasattr(state, "hyperparams"):
                return i
        raise ValueError("Optimizer chain has no inject_hyperparams state")

    def current_lr(self, opt_state):
        """Read the current learning rate from the inject_hyperparams state"""
        i = self._inject_state_index(opt_state)
        return float(opt_state[i].hyperparams["learning_rate"])


@register_dataclass
@dataclass
class Batch:
    """Batch of training data with enough state to resume the data loader"""

    x: jax.Array
    y: jax.Array
    idx_batches: np.ndarray
    idx: int = field(metadata=dict(static=True))
    idx_shard: int = field(metadata=dict(static=True))
    shard_cycle: int = field(metadata=dict(static=True))
    rng_state: dict = field(metadata=dict(static=True))


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
        # TODO: could be made an attribute for direct declaration of filenames
        return self.index.filenames_absolute

    @property
    def n_shards(self):
        """Number of shards"""
        return len(self.filenames)

    def iter(self, block_size, resume=None):
        random_state = np.random.default_rng(self.seed)

        shards = np.arange(self.n_shards)

        if self.randomize_shards:
            shards = random_state.permutation(shards)

        shard_cycle_start = 0
        resume_idx = -1

        if resume is not None:
            random_state.bit_generator.state = resume.rng_state
            shard_cycle_start = resume.shard_cycle
            resume_idx = resume.idx

        shard_cycle = shard_cycle_start

        while True:
            idx_shard = int(shards[shard_cycle % len(shards)])
            filename, checksum = (
                self.filenames[idx_shard]["name"],
                self.filenames[idx_shard]["checksum"],
            )

            with safe_open(filename, framework="numpy", device="cpu") as f:
                # TODO: load straight to device for zero copy, see also https://github.com/huggingface/safetensors/issues/636
                log.info("Reading %s", filename)
                data = f.get_tensor("tokens")

                if self.verify and checksum != get_checksum(data):
                    raise ValueError(f"Checksum does not agree for {filename}")

            max_val = len(data) - block_size
            n_batches = len(data) // self.batch_size // block_size
            start_idx = resume_idx + 1 if shard_cycle == shard_cycle_start else 0

            # we aim for a statistical coverage here...
            for idx in range(start_idx, n_batches):
                idx_batches = random_state.integers(max_val, size=(self.batch_size,))
                x = np.stack([data[i : i + block_size] for i in idx_batches])
                y = np.stack([data[i + 1 : i + 1 + block_size] for i in idx_batches])
                yield Batch(
                    x=jax.device_put(x, self.sharding.jax),
                    y=jax.device_put(y, self.sharding.jax),
                    idx_batches=idx_batches,
                    idx=idx,
                    idx_shard=idx_shard,
                    shard_cycle=shard_cycle,
                    rng_state=random_state.bit_generator.state,
                )

            shard_cycle += 1


@dataclass
class ProfileConfig:
    """Profiling configuration"""

    record_trace: bool = False  # Whether ro record a profile trace
    path: Path = PATH_BASE / ".profile"  # Where to save the trace
    warm_up: int = 5  # Number of warm up iterations
    n_iters: int = 2  # Number of iterations to capture


@dataclass
class Trainer:
    """Training configuration"""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    eval_interval: int = 2000
    eval_iters: int = 3
    eval_hellaswag: bool = False
    checkpoint_interval: int = 5000
    show_progress: bool = True  # show progress bar
    wandb_log: bool = False
    total_batch_size: int = 256  # total batch size in units of tokens
    checkpoint_path: Path = PATH_DATA / "checkpoints"
    filename_pattern: str = "model-n-iter-{n_iter}.safetensors"
    opt_state_pattern: str = "opt-state-n-iter-{n_iter}.safetensors"

    def _save_opt_state(self, opt_state, batch, n_iter):
        """Persist optimizer state and data loader resume info"""
        path = self.checkpoint_path / self.opt_state_pattern.format(n_iter=n_iter)
        log.info("Writing optimizer state to %s", path)
        save_file(
            flatten_pytree_with_path(opt_state),
            path,
            metadata={
                "rng_state": json.dumps(batch.rng_state),
                "shard_cycle": str(batch.shard_cycle),
                "idx": str(batch.idx),
                "idx_shard": str(batch.idx_shard),
            },
        )

    def _load_opt_state(self, path, fresh_opt_state):
        """Restore optimizer state into a freshly initialized pytree"""
        log.info("Reading optimizer state from %s", path)
        with safe_open(path, framework="numpy") as f:
            meta = f.metadata()
            arrays = {k: f.get_tensor(k) for k in f.keys()}

        def graft(tree_path, leaf):
            if isinstance(leaf, jax.Array):
                return jax.device_put(
                    jnp.asarray(arrays[join_path(tree_path)], dtype=leaf.dtype),
                    leaf.sharding,
                )
            return leaf

        opt_state = jax.tree.map_with_path(graft, fresh_opt_state)
        resume = Batch(
            x=None,
            y=None,
            idx_batches=None,
            idx=int(meta["idx"]),
            idx_shard=int(meta["idx_shard"]),
            shard_cycle=int(meta["shard_cycle"]),
            rng_state=json.loads(meta["rng_state"]),
        )
        return opt_state, resume

    def train(
        self,
        model,
        data_loader_train,
        data_loader_validate,
        rng_key,
        metadata,
        resume_from=-1,
        resume_opt_state_path=None,
    ):
        """Train model"""

        log.info(
            "Using %d gradient accumulation steps.",
            self.optimizer.gradient_accumulation_steps,
        )

        def loss_fn(model, x, y, rng_key, is_training):
            """Loss function"""
            logits = model(x, rng_key, is_training=is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss

        # Dropout is disabled in eval, but loss_fn still requires a key.
        eval_rng_key = jax.random.key(8273)

        def estimate_mean_loss(model, data_loader, n_iter):
            """Estimate mean loss across multiple iters"""
            losses = []
            # the range constrains the infinite data loader
            for _, batch in zip(range(n_iter), data_loader):
                value = loss_fn(
                    model, batch.x, batch.y, rng_key=eval_rng_key, is_training=False
                )
                losses.append(value)
            return jnp.mean(jnp.asarray(losses))

        @partial(jax.jit, donate_argnames=("model", "opt_state"))
        def train_step(model, opt_state, x, y, rng):
            """Training step"""
            value, grads = jax.value_and_grad(loss_fn)(
                model, x, y, rng, is_training=True
            )
            updates, opt_state = self.optimizer.optax.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, value

        flops = model.flops(
            batch_size=data_loader_train.batch_size,
            dtype=data_loader_train.dtype,
            sharding=data_loader_train.sharding.jax,
        )

        if self.eval_hellaswag:
            evaluator = ModelEvaluator()
            eval_sharding = data_loader_train.sharding.devices_jax[0]
            data_loader_hellaswag = load_hellaswag_examples(
                path=PATH_DATA / "download/hellaswag/validation-00000-of-00001.parquet",
                out_sharding=eval_sharding,
            )

        # Initialize optimizer state; restore from checkpoint if resuming
        opt_state = self.optimizer.optax.init(model)
        resume_batch = None

        if resume_opt_state_path is not None:
            opt_state, resume_batch = self._load_opt_state(
                resume_opt_state_path, opt_state
            )

        data_loader_train = data_loader_train.iter(
            block_size=model.config.block_size, resume=resume_batch
        )
        data_loader_validate = data_loader_validate.iter(
            block_size=model.config.block_size
        )

        # `dt` accumulates wall time across the eval window so the average step
        # time is measured over many dispatches, not the single async dispatch
        # of the current step.
        time_start = time.perf_counter()
        start_iter = max(resume_from, 0) + 1
        n_iter_last_eval = start_iter - 1

        with tqdm(
            total=self.optimizer.max_iters,
            initial=start_iter - 1,
            disable=not self.show_progress,
        ) as pbar:
            for n_iter, batch in zip(
                range(start_iter, self.optimizer.max_iters + 1), data_loader_train
            ):
                if self.profile.record_trace and n_iter == self.profile.warm_up:
                    jax.profiler.start_trace(
                        self.profile.path, profiler_options=PROFILER_OPTIONS
                    )
                    log.info("Starting profiler, recording to %s", self.profile.path)

                sub_rng_key = jax.random.fold_in(rng_key, n_iter)
                model, opt_state, loss_train = train_step(
                    model, opt_state, batch.x, batch.y, sub_rng_key
                )

                if self.profile.record_trace and n_iter == (
                    self.profile.warm_up + self.profile.n_iters
                ):
                    loss_train = jax.block_until_ready(loss_train)
                    jax.profiler.stop_trace()
                    log.info("Stop profiling")

                if n_iter % self.eval_interval == 0:
                    loss_train = jax.block_until_ready(loss_train)
                    dt = time.perf_counter() - time_start
                    step_time = dt / (n_iter - n_iter_last_eval)

                    # train_step does one fwd+bwd; PaLM-style 3x ratio.
                    mfu = 3 * flops.per_iter / FLOPS_UNIT / step_time
                    tps = flops.tokens_per_iter / step_time

                    loss_val = estimate_mean_loss(
                        model, data_loader_validate, n_iter=self.eval_iters
                    )
                    lr = self.optimizer.current_lr(opt_state)
                    pbar.set_postfix_str(
                        f"Loss train: {loss_train.item():.3f}, Loss val: {loss_val.item():.3f}, lr: {lr:.5f}, mfu: {(mfu):.0%}, tok/s: {sizeof_fmt(tps, system='decimal')}"
                    )

                    log_info = {
                        "n-iter": n_iter,
                        "loss-train": loss_train,
                        "loss-val": loss_val,
                        "lr": lr,
                        "shard": batch.idx_shard,
                        "mfu": mfu,
                        "tok/s": tps,
                    }

                    if self.eval_hellaswag:
                        model_single_device = jax.device_put(model, eval_sharding)
                        log_info["hellaswag-acc"] = evaluator.evaluate(
                            model=model_single_device, data_loader=data_loader_hellaswag
                        )

                    if self.wandb_log:
                        wandb.log(log_info)

                    time_start = time.perf_counter()
                    n_iter_last_eval = n_iter

                if n_iter % self.checkpoint_interval == 0:
                    metadata["n-iter"] = str(n_iter)
                    model.write(
                        self.checkpoint_path
                        / self.filename_pattern.format(n_iter=n_iter),
                        metadata=metadata,
                    )
                    self._save_opt_state(opt_state, batch, n_iter)

                pbar.update(1)

        return model


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

    def __post_init__(self):
        # sync arguments after init
        self.training.wandb_log = self.logging.wandb_log

        # calculate the gradient accumulation based on total batch size
        tokens_per_iter = self.model.block_size * self.loading.batch_size
        accum_steps = self.training.total_batch_size // tokens_per_iter
        self.training.optimizer.gradient_accumulation_steps = max(accum_steps, 1)
        self.training.checkpoint_path = (
            PATH_DATA / "checkpoints" / self.logging.wandb_run_name
        )
        self.training.profile.path = (
            self.training.profile.path / self.logging.wandb_run_name
        )

    @property
    def loading_val(self):
        """Validation dataset"""
        index = replace(self.loading.index, suffix="val")
        return replace(self.loading, index=index)

    @classmethod
    def read(cls, path: str):
        """Read configuration from file"""
        path = Path(path)
        log.info("Reading configuration from %s", path)

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
        log.info("Writing configuration to %s", path)
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
        except (ValueError, UnexpectedDataError) as e:
            log.warning("Error in file '%s', %s", filename.name, e)

    return configs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = tyro.extras.overridable_config_cli(get_configs())

    resume_from, run_id, resume_path, resume_opt_state_path = -1, None, None, None
    if config.init_from == InitFromEnum.resume:
        candidates = (PATH_DATA / "checkpoints").glob("**/model-n-iter-*.safetensors")
        resume_path = max(candidates, key=os.path.getctime)

        with safe_open(resume_path, framework="numpy") as f:
            log.info("Reading metadata from %s", resume_path)
            resume_from = int(f.metadata()["n-iter"])
            run_id = f.metadata().get("wandb-run-id")

        resume_opt_state_path = (
            resume_path.parent
            / config.training.opt_state_pattern.format(n_iter=resume_from)
        )
        if not resume_opt_state_path.exists():
            raise FileNotFoundError(
                f"Optimizer state not found alongside model: {resume_opt_state_path}"
            )

    metadata = config.to_safetensors_meta()

    if config.logging.wandb_log:
        run = wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run_name,
            tags=config.logging.wandb_tags,
            config=asdict(config),
            id=run_id,
            resume="allow",
        )
        metadata["wandb-run-id"] = str(run.id)

    data_loader_train = config.loading
    log.info("Training dataset has %d tokens.", data_loader_train.index.n_tokens_total)

    data_loader_validate = config.loading_val
    log.info(
        "Validation dataset has %d tokens.", data_loader_validate.index.n_tokens_total
    )

    log.info("Using devices %s", config.sharding.jax.device_set)
    log.info("Using `%s` dot product implementation.", DOT_PRODUCT_ATTENTION)

    if resume_path is not None:
        model = GPT.read(resume_path, transpose_weights=False)
    else:
        model = GPT.from_init(config.init_from, config.model)

    spec = {"device": config.sharding.jax, "dtype": config.dtype.jax}
    init_key, train_key = jax.random.split(jax.random.key(config.seed))
    model = model.init(rng_key=init_key, **spec)

    log.info("%s", model.info())

    model = config.training.train(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_validate=data_loader_validate,
        rng_key=train_key,
        metadata=metadata,
        resume_from=resume_from,
        resume_opt_state_path=resume_opt_state_path,
    )

    filename = (
        PATH_DATA
        / "checkpoints"
        / config.logging.wandb_run_name
        / "model-final.safetensors"
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    model.write(filename, metadata=config.to_safetensors_meta())
