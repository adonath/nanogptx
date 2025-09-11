import logging
from collections import namedtuple
from dataclasses import dataclass

import jax
import numpy as np
import tiktoken
import tyro
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy_with_integer_labels

from model import GPT
from utils import PATH_DATA, InitFromEnum

log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TAB_WIDTH = 4
EvaluationExample = namedtuple(
    "EvaluationExample", ["ctx", "endings", "tokens", "mask", "label"]
)


def tokenize_example(example):
    """Tokenize a single example"""
    enc = tiktoken.get_encoding("gpt2")

    ctx_tokens = enc.encode(example.ctx)
    tok_rows, mask_rows = [], []

    for end in example.endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = np.zeros((4, max_len), dtype=np.int64)
    mask = np.zeros((4, max_len), dtype=np.int64)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = np.asarray(tok_row)
        mask[i, : len(mask_row)] = np.asarray(mask_row)

    return EvaluationExample(
        tokens=tokens,
        mask=mask,
        label=int(example.label),
        ctx=example.ctx,
        endings=example.endings,
    )


def load_hellaswag_examples(path):
    """Load examples from a parquet file"""
    import pandas as pd

    data = pd.read_parquet(path)

    for idx in range(len(data)):
        yield tokenize_example(data.iloc[idx])


@dataclass
class ModelEvaluator:
    """Model evaluator"""

    n_examples: int = 1000
    show_context: bool = False

    def evaluate(self, model, data_loader):
        num_correct = 0
        for _, example in zip(range(1, self.n_examples + 1), data_loader):
            logits = model(
                example.tokens,
                rng_key=jax.random.key(9232),
                is_training=False,
                inference=False,
            )
            shift_losses = softmax_cross_entropy_with_integer_labels(
                logits=logits[..., :-1, :], labels=example.tokens[..., 1:]
            )
            avg_loss = jnp.mean(shift_losses, where=example.mask[..., 1:], axis=1)
            pred = jnp.argmin(avg_loss, axis=0)

            num_correct += int(pred == example.label)

            title = f"Example {_}"
            title += "\n" + "-" * len(title)
            print(title)
            print("Eval:")
            print(
                f"\tAcc: {num_correct/_:.4f} predicted: {pred}, actual: {example.label}".expandtabs(
                    TAB_WIDTH
                )
            )

            # debug: pretty print a few examples, and the losses in each case
            if self.show_context:
                print(f"Context:\n\t{example.ctx}".expandtabs(TAB_WIDTH))
                print("Endings:")
                for idx, end in enumerate(example.endings):
                    print(
                        f"\t{idx + 1} (loss: {avg_loss[idx].item():.4f}) {end}".expandtabs(
                            TAB_WIDTH
                        )
                    )

            print()


if __name__ == "__main__":
    evaluator = tyro.cli(ModelEvaluator)

    hellaswag = load_hellaswag_examples(
        PATH_DATA / "download/hellaswag/validation-00000-of-00001.parquet"
    )

    model = GPT.from_init(InitFromEnum.gpt2).init()

    evaluator.evaluate(model=model, data_loader=hellaswag)
