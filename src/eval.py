import logging
from collections import namedtuple
from dataclasses import dataclass

import jax
import tiktoken
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy_with_integer_labels

from utils import PATH_DATA

log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


EvaluationExample = namedtuple(
    "EvaluationExample", ["ctx", "endings", "tokens", "mask", "label"]
)


def tokenize_example(example):
    """Tokenize a single example"""
    enc = tiktoken.get_encoding("gpt2")

    ctx_tokens = enc.encode(example["ctx"])
    tok_rows, mask_rows = [], []

    for end in example["endings"]:
        end_tokens = enc.encode(
            " " + end
        )  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = jnp.zeros((4, max_len), dtype=jnp.int64)
    mask = jnp.zeros((4, max_len), dtype=jnp.int64)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = jnp.asarray(tok_row)
        mask[i, : len(mask_row)] = jnp.asarray(mask_row)

    return EvaluationExample(
        tokens=tokens,
        mask=mask,
        label=example["label"],
        ctx=example["ctx"],
        endings=example["endings"],
    )


@dataclass
class ModelEvaluator:
    """Model evaluator"""

    def evaluate(self, model, data_loader):
        # model = torch.compile(model
        datas = []
        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        for example in data_loader:
            logits = model(
                example.tokens, rng_key=jax.random.key(9232), is_training=True
            )
            shift_losses = softmax_cross_entropy_with_integer_labels(
                logits=logits[..., :-1, :], labels=example.tokens[..., 1:]
            )
            avg_loss = jnp.mean(shift_losses, where=example.mask[..., 1:], axis=1)

            # accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            print(
                f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
            )

            # debug: pretty print a few examples, and the losses in each case
            if num_total < 10:
                print("---")
                print(f"Context:\n {example['ctx']}")
                print(f"Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                print(f"predicted: {pred_norm}, actual: {label}")

        # now write the data to a .bin file
        filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_val.bin")
        write_evalfile(filename, datas)

        return
