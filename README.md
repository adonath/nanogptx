# A "nanoGPT" implementation in pure JAX


[![Release](https://img.shields.io/github/v/release/adonath/nanogpt-jax)](https://img.shields.io/github/v/release/adonath/nanogpt-jax)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/nanogpt-jax/main.yml?branch=main)](https://github.com/adonath/nanogpt-jax/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/nanogpt-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/nanogpt-jax)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/nanogpt-jax)](https://img.shields.io/github/commit-activity/m/adonath/nanogpt-jax)
[![License](https://img.shields.io/github/license/adonath/nanogpt-jax)](https://img.shields.io/github/license/adonath/nanogpt-jax)

<p align="center">
<img width="61.8%" src="https://raw.githubusercontent.com/adonath/nanogpt-jax/main/assets/nanogpt-jax.jpg" alt="Banner"/>
</p>

The purpose of this repository is a clean as possible re-implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) in pure JAX.
It can be used for educational purposes, or as a clean starting point for small scale experiments for modified architecures,
training strategies or experments in interpretability.

## Getting started
This repositiry comes with mutiple pre-defined environmenst in a `pixi.toml` file. This makes ir very covenient to
run the model in CPU, GPU and even TPU environments.

To get started you first [install pixi](https://pixi.sh/latest/installation/). The you can just execute:

```bash
pixi run download --dataset shakespeare
pixi run prepare --dataset shakespeare --encoding char --shard-size 1000000 --shards-val 1
pixi run train train-shakespeare-char
pixi run sample --init-from resume --max-new-tokens 500 --num-samples 5
```
