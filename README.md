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
The purpose of this repository is mostly documenting my own learning progress on recent developments in AI. At the same time I wanted to provide an as clean as possible re-implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) in pure JAX. It can be used for educational purposes, or as a clean, but still hackable starting point for small scale experiments on modified architecures, training strategies or experiments in interpretability. I also wanted to learn more about JAX and found in the process that I typically ended up with simple and cleaner code compared to working with PyTorch. So the implementation is also for the meticulous among us!

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

## Acknowledgements

Thanks to @fcrespo82 for the names list from the [Ubuntu Name generator](https://ubuntu-name-generator.crespo.com.br).
