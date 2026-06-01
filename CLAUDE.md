# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch reimplementation of Karpathy's nanoGPT in **pure JAX** (no Flax/Equinox/Haiku). The goal is a clean, hackable, educational GPT-2 codebase. Modules are plain `@dataclass`es registered as JAX pytrees, configuration is kept entirely separate from model state, and everything serializes to safetensors.

## Environment & commands

The project uses [pixi](https://pixi.sh) (conda + pypi) with per-scenario environments defined in `pixi.toml`. There is no `pip install` path — always go through pixi. Key environments: `prepare` (data IO), `cpu`, `gpu`, `cpu-profile`/`gpu-profile`, `gpu-mps` (experimental, broken ops), `all-cpu` (adds notebook), `tensorboard`.

The standard workflow is always four steps: **download → prepare → train → sample**.

```bash
# Train a small char-level model on Tiny Shakespeare (CPU, ~2 min on M1)
pixi run download --dataset shakespeare
pixi run --environment prepare prepare --dataset shakespeare --encoding char --shard-size 1000000 --shards-val 1
pixi run --environment cpu train train-shakespeare-char
pixi run --environment cpu sample --init-from resume --sampler.max-new-tokens 500 --sampler.num-samples 5
```

- `pixi run <task>` maps to `python src/<task>.py` (see `[tasks]` in `pixi.toml`); scripts run with `src/` on the path, so intra-package imports are flat (`from model import GPT`).
- Every sub-command exposes `--help` (powered by tyro) listing the full nested config tree.
- The `train` command's positional arg (e.g. `train-shakespeare-char`) selects a base config; any field can be overridden inline with dotted flags, e.g. `--training.optimizer.max-iters 5 --loading.batch_size=16`.
- Profiling: add `--training.profile.record-trace`; traces land in `.profile/<run name>/` and open with `xprof` (in the `tensorboard` env).

### Lint

Pre-commit runs `ruff` and `ruff-format` with `--line-length=120`. Run `pre-commit run --all-files` or `ruff check --line-length=120 src/`.

### Tests

`tests/tests.py` holds a few sanity checks (mostly TODO). They import modules flatly, so run with `src/` on the path, e.g. `PYTHONPATH=src pixi run --environment cpu python -m pytest tests/tests.py`.

### CI

`.github/workflows/ci.yml` runs the full download → prepare → train (5 iters) → sample loop on Shakespeare-char on every push to `main`. Use that command sequence as the smoke test.

## Architecture

### Modules are dataclasses, config is separate (`src/model.py`)

The central design decision (documented in the docstring at the top of `model.py`): every neural-net "module" (`GPT`, `Block`, `CausalSelfAttention`, `MLP`, `Linear`, `LayerNorm`, `Embedding`, `Dropout`, `Gelu`) is a `@register_dataclass @dataclass` whose fields are **only the parameters** (plus `static=True` metadata fields for non-array hyperparams like `n_head`). This makes each module a JAX pytree that composes into one big pytree (`GPT`), so `jax.tree.map`, `jit`, `grad`, sharding, etc. all work natively on the model.

Consequences to respect when editing:
- **Never store config on a module.** Construct via `.from_config(config)` / `.from_n_features(...)` alternative constructors. Derived attributes (`vocab_size`, `n_embd`, `config`, `lm_head`) are `@property`s computed from parameter shapes, so a module can never go out of sync with its own state. `GPT.config` reconstructs a `GPTConfig` by reading shapes back out.
- `__call__` is the forward pass. `GPT.__call__` is `jax.jit`ed with `is_training`/`inference` as static args. Pass `rng_key` and `is_training` explicitly through the call tree (dropout needs them).
- The LM head shares weights with the token embedding (`GPT.lm_head` wraps `self.wte.weight`).

### Lazy initialization via `ArrayInfo`

`GPT.from_config(...)` does **not** allocate arrays. It builds a pytree whose leaves are `ArrayInfo` (shape + dtype + sharding + an `init` callable + optional `post_init` hook) — see `[[project_array_info_lazy_init]]`. Nothing is materialized until you call `model.init(rng_key, dtype, device)`, which `tree.map`s every `ArrayInfo` leaf to a real array. This lets `abstract_call` / `model.flops()` do shape/dtype/sharding checking via `jax.eval_shape` with zero FLOPs. When reading from a `read()`-produced model, the `init` callable is `InitializeFromSafetensors`, so `.init()` is the actual disk load — not a no-op.

### Serialization round-trips config through safetensors metadata

`GPT.write()` saves params plus a flattened `GPTConfig` (or full `Config`) into the safetensors `__metadata__`. `GPT.read()` reads the header, rebuilds the config (`from_safetensors_meta`, or from a sibling HuggingFace `config.json` for GPT-2), reconstructs the module skeleton, and grafts `ArrayInfo` loaders onto the leaves. HuggingFace GPT-2 weights need `transpose_weights=True` (Conv1D → Linear); resumed nanogptx checkpoints use `transpose_weights=False`. `GPT.from_init(InitFromEnum, config)` dispatches between `scratch`, `resume` (newest checkpoint by ctime), and pretrained `gpt2*` names.

### Hierarchical config (`src/train.py` `Config`, `src/utils*.py`)

Config is a tree of dataclasses (`Config` → `Trainer`/`OptimizerConfig`/`DatasetLoader`/`GPTConfig`/`WAndBConfig`/`ShardingConfig`). It's serialized to/from TOML and safetensors metadata via `dacite` with `cast=` for type coercion (errors caught early). TOML files in `configs/` are auto-discovered by `get_configs()` and exposed as named CLI configs through `tyro.extras.overridable_config_cli`. `Config.__post_init__` derives `gradient_accumulation_steps` from `total_batch_size`, wires checkpoint/profile paths to the run name, etc. To regenerate `configs/default.toml` (the documented reference of all options) run `python src/dump-config.py`.

### Devices, dtypes, sharding (`src/utils_jax.py`)

`JaxDevicesEnum`, `JaxDtypesEnum`, `JaxFloatDtypesEnum`, `JaxIntDtypesEnum` are `StrEnum`s **built dynamically at import** from what the running JAX backend actually exposes, each with a `.jax` property returning the real object. This is why dtype/device config values are strings that validate against the live backend. `ShardingConfig` builds a `NamedSharding` over a device mesh; only **SPMD data-parallel** sharding (batches split across devices) is implemented. `XLA_FLAGS=--xla_force_host_platform_device_count=8` in the `cpu` env fakes 8 devices for developing sharded code on CPU.

### Training loop (`src/train.py` `Trainer.train`)

Single `jit`ed `train_step` with donated `model`/`opt_state` buffers; optimizer is optax adamw wrapped in `inject_hyperparams` (so the live LR can be read back), `clip_by_global_norm`, and `apply_every` for gradient accumulation. Weight decay is masked to 2D+ params (skips biases and norm scales). The model, optimizer state, and **data-loader resume state** are all checkpointed so `--init-from resume` continues deterministically: `Batch` carries the RNG bit-generator state, shard cycle, and within-shard index; `DatasetLoader.iter` is an **infinite generator** (`while True`) that cycles shards forever — bound it with `zip(range(...), loader)` rather than materializing (see `[[feedback_infinite_generators]]`). Logging is via WandB (opt-in, `--logging.wandb-log`), with MFU/tokens-per-sec estimated from `model.flops()`.

### Data pipeline (`src/download.py`, `src/prepare.py`)

`download.py` fetches model weights or dataset shards from HuggingFace (URL registries `MODEL_URLS` / `DATA_URLS`) into `data/`, decompressing tar/zst inline. `prepare.py` runs a small `read → preprocess → tokenize` function pipeline (per-dataset `READ_METHODS`, `gpt2`/`char` encodings) over a multiprocessing pool, writing fixed-size token shards as safetensors with per-shard checksums, then a `summary-stats.json` index. The loader verifies those checksums at train time.

### Adding things

- **New dataset:** add to `DatasetEnum` (`src/utils.py`), add download URLs to `DATA_URLS` (`src/download.py`, handle decompression there), add a `read_*` function to `READ_METHODS` (`src/prepare.py`).
- **New config:** copy `configs/default.toml`, edit, drop it in `configs/` — it's auto-discovered and validated.
</content>
