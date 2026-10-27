"""Acceptance checks for exercise 6; the unfinished starter fails deliberately.

Run with the companion Python: python -m pytest -p no:cacheprovider test_acceptance.py
Set TRAINING_CONTROL_SOLUTION to a Python file to check a different submission.
The demo is the reference solution; this file is not part of the default test suite.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.training_control_demo import exercise_six_results  # noqa: E402

PATH = Path(os.environ.get("TRAINING_CONTROL_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("exercise_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)


def test_plain_gradient_descent_agrees_to_twelve_decimals():
    rng = np.random.default_rng(7)
    for _ in range(20):
        w, g = rng.normal(size=(2, 30))
        np.testing.assert_allclose(
            SOLUTION.decay_step(w, g, 0.1, 0.2), w - 0.1 * (g + 0.2 * w), rtol=0, atol=1e-12
        )


def test_fixed_history_penalty_and_decoupled_updates_differ():
    assert SOLUTION.decay_step(2.0, 3.0, 0.1, 0.2, denominator=2) == pytest.approx(1.81, abs=1e-12)
    assert 2 - 0.1 * (3 + 0.2 * 2) / 2 == pytest.approx(1.83, abs=1e-12)


def test_first_adam_update_has_decoupled_decay():
    path = SOLUTION.adam(np.array([[1.0]]), np.array([-1.0]), 0.1, 1, w0=[2], decay=0.2)
    assert path[-1, 0] == pytest.approx(1.86, abs=1e-8)


def test_three_decays_reproduce_seed_seven_and_the_generated_key():
    first, second = (
        exercise_six_results(SOLUTION.adam, seed=7),
        exercise_six_results(SOLUTION.adam, seed=7),
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, exercise_six_results(seed=7), rtol=0, atol=1e-12)
