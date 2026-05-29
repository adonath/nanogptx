import logging
from dataclasses import dataclass
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


@dataclass
class EvaluationExample:
    """A single multiple-choice evaluation example"""

    ctx: str
    endings: list[str]
    tokens: jax.Array
    mask: jax.Array
    label: int


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
    """Load examples from a parquet file"""
    data = pd.read_parquet(path)

    for row in data.itertuples(index=False):
        yield tokenize_example(row, out_sharding=out_sharding)


@dataclass
class ModelEvaluator:
    """Model evaluator"""

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

    def evaluate(self, model, data_loader) -> float:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = tyro.cli(ModelEvaluator)

    hellaswag = load_hellaswag_examples(
        PATH_DATA / "download/hellaswag/validation-00000-of-00001.parquet"
    )

    model = GPT.from_init(InitFromEnum.gpt2).init()

    accuracy = evaluator.evaluate(model=model, data_loader=hellaswag)
    log.info("Overall accuracy: %.4f", accuracy)
