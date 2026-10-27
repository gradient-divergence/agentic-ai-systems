"""Exercise 12: complete generate, then run this file and test_acceptance.py.

Set TRANSFORMERS_SOLUTION to a submission's Python file to run acceptance checks.
The canonical printed attention function is loaded alone, without running the
demo's other regions. Return (generated_vectors, cumulative_score_counts), with
one row per step; the starting vector is not a generated row. In the cached run,
retain previous keys and values and compute only the new query's attention row.
"""

from pathlib import Path

import numpy as np

from demos.regions import extract_snippets_from_text

SOURCE = Path(__file__).resolve().parents[2] / "demos/deep_learning_demo.py"
CORE = next(
    region
    for region in extract_snippets_from_text(SOURCE.read_text(), SOURCE)
    if region.snippet_id == "fnd.transformers-foundation-models.core-01"
)
NAMESPACE = {}
exec(compile(CORE.content, str(SOURCE), "exec"), NAMESPACE)
scaled_dot_product_attention = NAMESPACE["scaled_dot_product_attention"]


def generator_inputs(seed=7):
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=8)
    projections = tuple(rng.normal(size=(8, 8)) / np.sqrt(8) for _ in range(3))
    return initial, *projections


def generate(initial, Wq, Wk, Wv, steps=64, cached=False):
    """Append the last attention output at each step, using fixed projections.

    For n input vectors, a full causal attention call computes n*n scores;
    the new-row cached call computes n. Record actual cumulative work.
    """
    raise NotImplementedError("generate with the full causal mask, then retain keys and values")


def print_comparison(generate_fn=generate):
    inputs = generator_inputs()
    full, full_scores = generate_fn(*inputs, cached=False)
    cached, cached_scores = generate_fn(*inputs, cached=True)
    print("n | cached scores | full scores | max vector difference")
    for i in range(64):
        print(
            f"{i + 1} | {cached_scores[i]} | {full_scores[i]} | {np.max(np.abs(full[i] - cached[i])):.3g}"
        )


if __name__ == "__main__":
    print_comparison()
