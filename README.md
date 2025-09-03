# NanoGTPX: A "nanoGPT" Implementation in Pure JAX


[![Release](https://img.shields.io/github/v/release/adonath/nanogptx)](https://img.shields.io/github/v/release/adonath/nanogptx)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/nanogptx/main.yml?branch=main)](https://github.com/adonath/nanogptx/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/nanogptx/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/nanogptx)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/nanogptx)](https://img.shields.io/github/commit-activity/m/adonath/nanogptx)
[![License](https://img.shields.io/github/license/adonath/nanogptx)](https://img.shields.io/github/license/adonath/nanogptx)

<p align="center">
<img width="61.8%" src="https://raw.githubusercontent.com/adonath/nanogptx/main/assets/nanogpt-jax.jpg" alt="Banner"/>
</p>

## Purpose of this Repository
The purpose of this repository is mostly documenting my own learning progress on recent developments in AI. I first wanted to learn more transformers and the process of training LLMs and at the same time I wanted to learn about [JAX](https://docs.jax.dev/en/latest/). Give these goals a reasonable project was to re-implement **[nanoGPT](https://github.com/karpathy/nanoGPT) in pure JAX**. In the process I have found that I typically ended up with much cleaner code, compared to PyTorch. So I decided to split the code base up into smaller reusable and more modular parts and release it. Now it can be used for **educational purposes, or as a clean and hackable starting point for small scale experiments on modified architecures, training strategies or experiments in interpretability**. I think cooking a new experiment needs to start from a clean lab, so **happy cooking**!

**Note**: if you need minimal production grade implementations of LLMs you might rather want to check out [official JAX LLM examples](https://github.com/jax-ml/jax-llm-examples) or for large scale experiments and training checkout [Levanter](https://github.com/stanford-crfm/levanter), which is based on [Equinox](https://docs.kidger.site/equinox/).

## Getting started
This repositiry comes with mutiple pre-defined environments in a `pixi.toml` file. This makes it very covenient to run the model in CPU, GPU and even TPU (soon...) environments.
To get started, you first [install pixi](https://pixi.sh/latest/installation/), then proceed with one of the options:

### (a) Training a Small Model on "Tiny Shakespeare" and CPU
To train a small transformer model with character level encoding on the "tiny Shakespeare" dataset you can use:

```bash
pixi run download --dataset shakespeare
pixi run prepare --dataset shakespeare --encoding char --shard-size 1000000 --shards-val 1
pixi run train --environment cpu train-shakespeare-char
pixi run sample --init-from resume --sample.rmax-new-tokens 500 --sampler.num-samples 5
```
The workflow always consists of those four steps. The training should finish in <2 minutes on a M1 type machine.


### (b) Training a GPT2 124m Model on Fineweb10b and GPU
To train a GPT2 124m model on the Fineweb10b dataset on two GPUs you can use for example:
```bash
pixi run download --dataset fineweb_10b
pixi run prepare --dataset fineweb_10b
pixi run train  --environment gpu train-fineweb-10b --sharding.devices cuda:0,cuda:1 --loading.sharding.devices cuda:0,cuda:1
pixi run sample --init-from resume --max-new-tokens 500 --num-samples 5
```
`nanogptx` supports a simple SPMD (single program multiple data) distribution strategy, meaning groups of batches are evaluated in parallel on the configured devices.


## NanoGPTX Features

Here are some of the features of this implementation:

- 🗄️ **Hierarchical configuration:** I do like hierarchical configuration as long as it is not too deep (3-4 levels max). The confiuration system is based on TOML and dataclasses, combined with JAX pytree operations for serialization and deserialization. The configuration requires setting defaults and on I/O it does type coercion to the default type, to catch errors early.
- 💻 **CLI:** after careful consideration I have decied to support a CLI via [tyro](https://brentyi.github.io/tyro/). The code overhead is minimal, as all the configuration is in dataclasses anyway. And tyro offers a nice default interface as well as nested commands.
- ✨ **Abstract evaluation and lazy initialization:** I think it is useful to not fully instantiate a model on creation, but rather instantiate an abstract description of the array shapes, dtypes and shardings. This allows for an abstract evaluation which catches shape, dtype and sharding errors early without using any flops.
- 🗃️ **Minimal provenance:** The implementation supports minimal provenance of model configs, datasets and training. This includes logging of which batch is trained on, saving configs in model files and verifying data hashes.
- 🐍 **Support for Pixi enviromments:** this repository includes a `pixi.toml` with pre-defined environments for many scenarios such as CPU, CPU and even TPU. I have also tried to support MPS, via `jax-metal`, but ran into multiple issus with missing support for operations.
- 💯 **Sharding strategies**: Currently only SPMD is supported, other strategies might follow...
- 📚 **Data preprocessing pipeline:** A minimal function based pre-processing pipeline for tokenization and custom document cleaning / pre-processing.
- 📇 **Logging**: just as the original nanoGPT this project uses [WandB](https://wandb.ai/) for logging. I have considered alternatives (especially local solutions), but found other solutions introduced more complexity with fewer features.

## How to work with this Repository
This repository can be used as template for your own small to mid-scale research and educational projects.

### Adding a new Dataset
If you would like to add a new dataset follow these steps:

- Add a new entry to the `DatasetEnum` in `src/utils.py` with a short identifier of your dataset
- Add the download urls in `src/download.py`, decompressing / unzip / untar should also happen at this step
- Add a custom read function in `src/prepare.py` as needed.

### Adding a new Model
TODO:

### Adding a new Config
TODO:


## Acknowledgements

Thanks to [@fcrespo82](https://github.com/fcrespo82) for the names list from the [Ubuntu Name generator](https://ubuntu-name-generator.crespo.com.br).
