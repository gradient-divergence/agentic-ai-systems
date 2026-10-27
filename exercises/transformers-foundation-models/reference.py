"""Reference solution for testing exercise 12's learner harness.

Run test_acceptance.py with TRANSFORMERS_SOLUTION set to this file.
Run this file to print the 64-row comparison table.
"""

import runpy
from pathlib import Path

import numpy as np

SCAFFOLD = runpy.run_path(str(Path(__file__).with_name("starter.py")))
scaled_dot_product_attention = SCAFFOLD["scaled_dot_product_attention"]


def generate(initial, Wq, Wk, Wv, steps=64, cached=False):
    vectors = [np.asarray(initial, dtype=float).copy()]
    keys, values, generated, counts = [], [], [], []
    total = 0
    for n in range(1, steps + 1):
        if cached:
            latest = vectors[-1][None, :]
            keys.append(latest @ Wk)
            values.append(latest @ Wv)
            output, _ = scaled_dot_product_attention(
                latest @ Wq, np.vstack(keys), np.vstack(values)
            )
            total += n
        else:
            x = np.asarray(vectors)
            mask = np.triu(np.full((n, n), -np.inf), k=1)
            output, _ = scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, mask)
            total += n * n
        vectors.append(output[-1].copy())
        generated.append(vectors[-1])
        counts.append(total)
    return np.asarray(generated), np.asarray(counts)


if __name__ == "__main__":
    SCAFFOLD["print_comparison"](generate)
