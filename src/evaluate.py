import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import jax
import numpy as np
import pandas as pd
import tiktoken
import tyro
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy_with_integer_labels

from model import GPT
from utils import PATH_DATA, InitFromEnum

log = logging.getLogger(__name__)

TAB_WIDTH = 4


class BenchmarkEnum(StrEnum):
    """Supported evaluation benchmarks"""

    hellaswag = "hellaswag"
    lambada_openai = "lambada-openai"


@dataclass
class EvaluationExample:
    """A single multiple-choice evaluation example"""

    ctx: str
    endings: list[str]
    tokens: jax.Array
    mask: jax.Array
    label: int


@dataclass
class LambadaExample:
    """A single LAMBADA last-word prediction example"""

    ctx: str
    target: str
    tokens: jax.Array
    mask: jax.Array


@lru_cache(maxsize=1)
def _get_gpt2_encoder():
    """Load and cache the GPT-2 tiktoken encoder"""
    return tiktoken.get_encoding("gpt2")


def tokenize_example(example, out_sharding=None) -> EvaluationExample:
    """Tokenize a single multiple-choice example"""
    enc = _get_gpt2_encoder()

    ctx_tokens = enc.encode(example.ctx)
    tok_rows, mask_rows = [], []

    for end in example.endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    n_endings = len(tok_rows)
    max_len = max(len(row) for row in tok_rows)
    tokens = np.zeros((n_endings, max_len), dtype=np.int32)
    mask = np.zeros((n_endings, max_len), dtype=bool)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = np.asarray(tok_row)
        mask[i, : len(mask_row)] = np.asarray(mask_row)

    return EvaluationExample(
        tokens=jax.device_put(tokens, out_sharding),
        mask=jax.device_put(mask, out_sharding),
        label=int(example.label),
        ctx=example.ctx,
        endings=example.endings,
    )


def load_hellaswag_examples(path, out_sharding=None):
    """Yield examples from a parquet file, cycling indefinitely"""
    data = pd.read_parquet(path)

    while True:
        for row in data.itertuples(index=False):
            yield tokenize_example(row, out_sharding=out_sharding)


def tokenize_lambada_example(text, out_sharding=None) -> LambadaExample:
    """Tokenize a single LAMBADA example into context + target last word"""
    enc = _get_gpt2_encoder()

    ctx, _, target = text.strip().rpartition(" ")

    ctx_tokens = enc.encode(ctx)
    target_tokens = enc.encode(" " + target)

    tokens = np.asarray(ctx_tokens + target_tokens, dtype=np.int32)[None, :]
    mask = np.zeros_like(tokens, dtype=bool)
    mask[0, len(ctx_tokens) :] = True

    return LambadaExample(
        tokens=jax.device_put(tokens, out_sharding),
        mask=jax.device_put(mask, out_sharding),
        ctx=ctx,
        target=target,
    )


def load_lambada_examples(path, out_sharding=None):
    """Yield examples from a JSON-lines file, cycling indefinitely"""
    with open(path) as fh:
        texts = [json.loads(line)["text"] for line in fh if line.strip()]

    while True:
        for text in texts:
            yield tokenize_lambada_example(text, out_sharding=out_sharding)


@dataclass
class ModelEvaluator:
    """Model evaluator"""

    benchmark: BenchmarkEnum = BenchmarkEnum.hellaswag
    init_from: InitFromEnum = InitFromEnum.gpt2
    n_examples: int = 64
    print_results: bool = False

    @staticmethod
    def print_result(idx, example, num_correct, pred, avg_loss):
        """Print results"""
        title = f"Example {idx}"
        title += "\n" + "-" * len(title)
        print(title)
        print("Eval:")
        print(
            f"\tAcc: {num_correct / idx:.4f} predicted: {pred}, actual: {example.label}".expandtabs(
                TAB_WIDTH
            )
        )
        print(f"Context:\n\t{example.ctx}".expandtabs(TAB_WIDTH))
        print("Endings:")
        for i, end in enumerate(example.endings):
            print(
                f"\t{i} (loss: {avg_loss[i].item():.4f}) {end}".expandtabs(TAB_WIDTH)
            )
        print()

    @staticmethod
    def print_lambada_result(idx, example, num_correct, correct, pred):
        """Print results"""
        title = f"Example {idx}"
        title += "\n" + "-" * len(title)
        print(title)
        print("Eval:")
        print(
            f"\tAcc: {num_correct / idx:.4f} correct: {bool(correct)}".expandtabs(
                TAB_WIDTH
            )
        )
        print(f"Context:\n\t{example.ctx}".expandtabs(TAB_WIDTH))
        print(f"Target: {example.target!r}".expandtabs(TAB_WIDTH))
        print(f"Predicted: {pred!r}".expandtabs(TAB_WIDTH))
        print()

    def evaluate_hellaswag(self, model, data_loader) -> float:
        """Evaluate hellaswag accuracy"""
        num_correct = 0
        n_seen = 0
        rng_key = jax.random.key(9232)

        for idx, example in zip(range(1, self.n_examples + 1), data_loader):
            logits = model(
                example.tokens,
                rng_key=rng_key,
                is_training=False,
                inference=False,
            )
            shift_losses = softmax_cross_entropy_with_integer_labels(
                logits=logits[..., :-1, :], labels=example.tokens[..., 1:]
            )
            avg_loss = jnp.mean(shift_losses, where=example.mask[..., 1:], axis=1)
            pred = jnp.argmin(avg_loss, axis=0)

            num_correct += int(pred == example.label)
            n_seen = idx

            if self.print_results:
                self.print_result(idx, example, num_correct, pred, avg_loss)

        return num_correct / n_seen if n_seen else 0.0

    def evaluate_lambada(self, model, data_loader) -> float:
        """Evaluate LAMBADA last-word prediction accuracy

        A prediction counts as correct only if the model greedily decodes
        every token of the target last word, matching ``lm-eval-harness``'s
        ``lambada_openai`` accuracy metric.
        """
        enc = _get_gpt2_encoder()
        num_correct = 0
        n_seen = 0
        rng_key = jax.random.key(9232)

        for idx, example in zip(range(1, self.n_examples + 1), data_loader):
            logits = model(
                example.tokens,
                rng_key=rng_key,
                is_training=False,
                inference=False,
            )
            preds = jnp.argmax(logits[..., :-1, :], axis=-1)
            targets = example.tokens[..., 1:]
            target_mask = example.mask[..., 1:]

            correct = bool(jnp.all((preds == targets) | ~target_mask))
            num_correct += int(correct)
            n_seen = idx

            if self.print_results:
                pred = enc.decode(np.asarray(preds[target_mask]).tolist())
                self.print_lambada_result(idx, example, num_correct, correct, pred)

        return num_correct / n_seen if n_seen else 0.0

    def run(self, model) -> float:
        """Load the selected benchmark and evaluate accuracy"""
        if self.benchmark == BenchmarkEnum.hellaswag:
            data_loader = load_hellaswag_examples(
                PATH_DATA / "download/hellaswag/validation-00000-of-00001.parquet"
            )
            return self.evaluate_hellaswag(model=model, data_loader=data_loader)

        data_loader = load_lambada_examples(
            PATH_DATA / "download/lambada-openai/lambada_test_en.jsonl"
        )
        return self.evaluate_lambada(model=model, data_loader=data_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = tyro.cli(ModelEvaluator)

    model = GPT.from_init(evaluator.init_from).init()

    accuracy = evaluator.run(model=model)
    log.info("%s accuracy: %.4f", evaluator.benchmark.value, accuracy)
