# A "nanoGPT" implementation in pure JAX


[![Release](https://img.shields.io/github/v/release/adonath/nanogpt-jax)](https://img.shields.io/github/v/release/adonath/nanogpt-jax)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/nanogpt-jax/main.yml?branch=main)](https://github.com/adonath/nanogpt-jax/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/nanogpt-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/nanogpt-jax)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/nanogpt-jax)](https://img.shields.io/github/commit-activity/m/adonath/nanogpt-jax)
[![License](https://img.shields.io/github/license/adonath/nanogpt-jax)](https://img.shields.io/github/license/adonath/nanogpt-jax)

<p align="center">
<img width="61.8%" src="https://raw.githubusercontent.com/adonath/nanogpt-jax/main/assets/nanogpt-jax.jpg" alt="Banner"/>
</p>

## Purpose of this Repository
The purpose of this repository is mostly documenting my own learning progress on recent developments in AI. I also wanted to learn more about JAX and found in the process that I typically ended up with simple and cleaner code compared to working with PyTorch, so I eneded up with this as clean as possible re-implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) in pure JAX. It can be used for educational purposes, or as a clean, but still hackable starting point for small scale experiments on modified architecures, training strategies or experiments in interpretability. I think cooking a new experiment needs to start from a clean lab, so happy cooking!

**Note**: if you need minimal production grade implementations of LLMS you might rather want to check out [official JAX LLM examples](https://github.com/jax-ml/jax-llm-examples).

## Getting started
This repositiry comes with mutiple pre-defined environmenst in a `pixi.toml` file. This makes ir very covenient to run the model in CPU, GPU and even TPU environments.

To get started you first [install pixi](https://pixi.sh/latest/installation/). Then you can just execute:

```bash
pixi run download --dataset shakespeare
pixi run prepare --dataset shakespeare --encoding char --shard-size 1000000 --shards-val 1
pixi run train train-shakespeare-char
pixi run sample --init-from resume --max-new-tokens 500 --num-samples 5
```

## Features

Here are some of the features of this implementation:

- **Hierarchical configuration:** I do like hierarchical configuration as loing as it is not too deep. The confiuration system is based on TOML and dataclasses, combined with JAX pytree operations for serialization and deserialization. This might seem like a misue of the the system, but it is extremely simple and works rather well. It does type coercion to the default type so you get sort of minimal Pydantic experience.
- **CLI:** after careful consideration I have decied to support a CLI via `tyro`. The overhead is minimal, as all the configuration is in dataclasses anyway. If you don't like it you can remove it.
- **Abstract evaluation and lazy initialization:** I think it is useful to not full instantiate a model on creation, but rather instantiate an abstract description of the array shapes, dtypes and shardings. This allows for an abstract evaluation which catches shape and dtype errors early.
- **Minimal provenance:** The implementation supports minimal provenance of model configs, datasets etc.
- **Support for Pixi enviromments:** this repository includes a `pixi.toml` with pre-defined environments for many scenarios such as CPU, CPU and even TPU.
- **Sharding strategies**: TODO: support for configurable sharding strategies.


## Acknowledgements

Thanks to @fcrespo82 for the names list from the [Ubuntu Name generator](https://ubuntu-name-generator.crespo.com.br).
