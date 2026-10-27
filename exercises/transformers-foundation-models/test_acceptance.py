"""Exercise 12 acceptance; starter fails deliberately, reference solution passes.

Run explicitly with the companion Python and repository/companion PYTHONPATH.
Set TRANSFORMERS_SOLUTION=/path/to/submission.py to test a completed generator.
"""

import importlib.util
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

PATH = Path(os.environ.get("TRANSFORMERS_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("transformers_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)
SCAFFOLD = runpy.run_path(str(Path(__file__).with_name("starter.py")))
ATTENTION = SCAFFOLD["scaled_dot_product_attention"]


def oracle(inputs, steps):
    initial, Wq, Wk, Wv = inputs
    vectors = [initial.copy()]
    for n in range(1, steps + 1):
        x = np.asarray(vectors)
        mask = np.where(np.arange(n)[None, :] > np.arange(n)[:, None], -np.inf, 0)
        output, _ = ATTENTION(x @ Wq, x @ Wk, x @ Wv, mask)
        vectors.append(output[-1])
    return np.asarray(vectors[1:])


@pytest.mark.parametrize("seed", [7, 19])
def test_every_generated_vector_matches_full_causal_attention(seed):
    inputs = SCAFFOLD["generator_inputs"](seed)
    expected = oracle(inputs, 64)
    original = [value.copy() for value in inputs]
    full, full_scores = SOLUTION.generate(*inputs, steps=64, cached=False)
    cached, cached_scores = SOLUTION.generate(*inputs, steps=64, cached=True)
    assert np.asarray(full).shape == np.asarray(cached).shape == (64, 8)
    np.testing.assert_allclose(full, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cached, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cached, full, rtol=0, atol=1e-12)
    n = np.arange(1, 65)
    np.testing.assert_array_equal(cached_scores, n * (n + 1) // 2)
    np.testing.assert_array_equal(full_scores, n * (n + 1) * (2 * n + 1) // 6)
    for value, before in zip(inputs, original):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("cached", [False, True])
def test_reported_scores_are_actual_attention_work_and_full_calls_are_causal(cached, monkeypatch):
    calls = []

    def counted(queries, keys, values, mask=None):
        n = len(keys)
        assert len(values) == n
        if cached:
            assert queries.shape == (1, 8)
        else:
            assert queries.shape == (n, 8)
            assert mask is not None and np.asarray(mask).shape == (n, n)
            future = np.arange(n)[None, :] > np.arange(n)[:, None]
            assert np.isneginf(np.asarray(mask)[future]).all()
            assert (np.asarray(mask)[~future] == 0).all()
        calls.append((len(queries), n))
        return ATTENTION(queries, keys, values, mask)

    monkeypatch.setattr(SOLUTION, "scaled_dot_product_attention", counted)
    _, scores = SOLUTION.generate(*SCAFFOLD["generator_inputs"](), steps=64, cached=cached)
    assert calls == [(1 if cached else n, n) for n in range(1, 65)]
    np.testing.assert_array_equal(scores, np.cumsum([q * k for q, k in calls]))
