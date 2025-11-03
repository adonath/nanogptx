# NanoGTPX: A "nanoGPT" Implementation in Pure JAX


[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/nanogptx/ci.yml?branch=main)](https://github.com/adonath/nanogptx/actions/workflows/ci.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/nanogptx)](https://img.shields.io/github/commit-activity/m/adonath/nanogptx)
[![License](https://img.shields.io/github/license/adonath/nanogptx)](https://img.shields.io/github/license/adonath/nanogptx)

<p align="center">
<img width="90%" src="https://raw.githubusercontent.com/adonath/nanogptx/main/assets/nanogpt-jax.jpg" alt="Banner"/>
</p>

## Purpose of this Repository
The purpose of this repository is mostly documenting my own learning progress on recent developments in AI. I first wanted to learn more transformers and the process of training LLMs and at the same time I wanted to learn about [JAX](https://docs.jax.dev/en/latest/). Give these goals a reasonable project was to re-implement **[nanoGPT](https://github.com/karpathy/nanoGPT) in pure JAX**. In the process I have found that I typically ended up with much cleaner code, compared to PyTorch. So I decided to split the code base up into smaller reusable and more modular parts and release it. Now it can be used for **educational purposes, or as a clean and hackable starting point for small scale experiments on modified architectures, training strategies or experiments in interpretability**. I think cooking a new experiment needs to start from a clean lab, so **happy cooking**!

**Note**: if you need minimal production grade implementations of LLMs you might rather want to check out [official JAX LLM examples](https://github.com/jax-ml/jax-llm-examples) or for large scale experiments and training checkout [Levanter](https://github.com/stanford-crfm/levanter), which is based on [Equinox](https://docs.kidger.site/equinox/).

## Getting started
This repositiry comes with mutiple pre-defined environments in a `pixi.toml` file. This makes it very covenient to run the model in CPU, GPU and even TPU (soon...) environments.
To get started, you first [install pixi](https://pixi.sh/latest/installation/) using:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

And then proceed with one of the options:

### (a) Training a Small Model on "Tiny Shakespeare" and CPU
To train a small transformer model with character level encoding on the "Tiny Shakespeare" dataset you can use:

```bash
pixi run download --dataset shakespeare
pixi run --environment prepare prepare --dataset shakespeare --encoding char --shard-size 1000000 --shards-val 1
pixi run --environment cpu train train-shakespeare-char
pixi run --environment cpu sample --init-from resume --sampler.max-new-tokens 500 --sampler.num-samples 5
```
The workflow always consists of those four steps. The training should finish in <2 minutes on a M1 type machine.
All the sub-commands have a `--help` option which shows you the available configuration options.


### (b) Training a GPT2 124m Model on Fineweb10b and multiple GPUs
To train a GPT2 124m model on the Fineweb10b dataset on two GPUs you can use for example:
```bash
pixi run download --dataset fineweb_10b
pixi run --environment prepare prepare --dataset fineweb_10b
pixi run --environment gpu train train-fineweb-10b --sharding.devices cuda:0,cuda:1 --loading.sharding.devices cuda:0,cuda:1
pixi run --envrionment gpu sample --init-from resume --sampler.max-new-tokens 500 --sampler.num-samples 5
```
`nanogptx` supports a simple SPMD (single program multiple data) distribution strategy, meaning groups of batches are evaluated in parallel on the configured devices.


## NanoGPTX Features

Here are some of the features of the `nanogptx` implementation:

- 🗄️ **Hierarchical configuration:** I do like hierarchical configuration as long as it is not too deep (3-4 levels max). The confiuration system is based on TOML and dataclasses, combined with JAX pytree operations and [dacite](https://github.com/konradhalas/dacite) for serialization and deserialization. The configuration requires setting defaults and on I/O it does type coercion to the default type, to catch errors early.
- 💻 **CLI:** after careful consideration I have decied to support a CLI via [tyro](https://brentyi.github.io/tyro/). The code overhead is minimal, as all the configuration is in dataclasses anyway. And tyro offers a nice default interface as well as nested commands.
- ✨ **Abstract evaluation and lazy initialization:** I think it is useful to not fully instantiate a model on creation, but rather instantiate an abstract description of the array shapes, dtypes and shardings. This allows for an abstract evaluation which catches shape, dtype and sharding errors early without using any flops.
- 🗃️ **Minimal provenance:** The implementation supports minimal provenance of model configs, datasets and training. This includes logging of which batch is trained on, saving configs in model files and verifying data hashes.
- 🐍 **Support for Pixi enviromments:** this repository includes a `pixi.toml` with pre-defined environments for many scenarios such as CPU, CPU and even TPU. I have also tried to support MPS, via `jax-metal`, but ran into multiple issus with missing support for operations.
- 💯 **Sharding strategies**: Currently only SPMD is supported, other strategies might follow...
- 📚 **Data preprocessing pipeline:** A minimal function based pre-processing pipeline for tokenization and custom document cleaning / pre-processing.
- 📇 **Logging**: just as the original nanoGPT this project uses [WandB](https://wandb.ai/) for logging. I have considered alternatives (especially local solutions), but found other solutions introduced more complexity with fewer features.

## How to work with this Repository
This repository can be used as template for your own small to mid-scale research and educational projects. You can explore different training strategies,
modified architectures etc. As everything is in pure JAX, you can modify any small component in the model, without the need of implementing whole new layers.
`nanogptx` still provides the whole scalable infrastructure.

### Profiling
There is a dedicated option and environment for profiling available. The approch follows the [programmatic capture section in the JAX docs](https://docs.jax.dev/en/latest/profiling.html#programmatic-capture).
An you can enable it using:

```bash
pixi run --environment cpu train train-shakespeare-char --training.profile.record
pixi run --environment gpu train train-fineweb-10b-8-gpus --training.profile.record
```

The profiles will be stored in `.profile/<run name>/` and can be opened using e.g. `xprof`:

```bash
pixi shell --environment tensorboard
xprof --logdir=.profile/<run name>/ --port=6006
```

### Adding a new Config
If you would like to add a new config, I would suggest to start from the `config/default.toml` file, which includes all available configuration options. Copy the file to a new name and edit as needed. As long as the file is in the `config/` folder it will be automatically discovered and validated.

### Adding a new Dataset
If you would like to add a new dataset follow these steps:

- Add a new entry to the `DatasetEnum` in `src/utils.py` with a short identifier of your dataset
- Add the download urls in the registry in `src/download.py`, decompressing / unzip / untar should also happen at this step
- Add a custom read function in `src/prepare.py` as needed and add it to the registry

## Acknowledgements

Thanks to [@fcrespo82](https://github.com/fcrespo82) for the names list from the [Ubuntu Name generator](https://ubuntu-name-generator.crespo.com.br).
