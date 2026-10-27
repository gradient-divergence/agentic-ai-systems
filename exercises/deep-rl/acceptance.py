"""Exercise 6 checks; run `python acceptance.py candidate.py [report.json]`.

The report records both targets' ten episode lengths and starting estimates
for each of five seeds, shared episode seeds, and all settings. A fresh rerun
must reproduce its bytes; test_deep_rl.py must pass unchanged. Those two checks
are required separately and are not certified by this report checker.
Value comparisons discount the unit reward for each push; adoption uses lengths.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from demos.deep_rl.depot import adopt
from demos.deep_rl.dqn import EVALUATION_SEEDS, TRAINING_SEEDS, td_target


def check_target(candidate):
    online = np.array([[4.0, 3.0], [1.0, 2.0], [5.0, 1.0]])
    target = np.array([[2.0, 8.0], [3.0, 4.0], [0.0, 9.0]])
    reward, terminated = np.array([1.0, 1.0, -2.0]), np.array([False, False, True])
    actual = candidate(reward, terminated, online, target, 0.9)
    np.testing.assert_allclose(actual, [2.8, 4.6, -2])
    original = td_target(reward, terminated, target, 0.9)
    different = np.argmax(online, axis=1) != np.argmax(target, axis=1)
    np.testing.assert_array_equal(
        actual[~different | terminated], original[~different | terminated]
    )
    assert np.all(actual[different & ~terminated] != original[different & ~terminated])


def check_report(report):
    assert report["training_seeds"] == list(TRAINING_SEEDS)
    assert report["evaluation_seeds"] == list(EVALUATION_SEEDS)
    settings = report["settings"]
    assert settings["budget"] > 0 and settings["gamma"] < 1
    gamma = settings["gamma"]
    for key in ("widths", "eta", "batch_size", "capacity", "refresh", "epsilon_steps"):
        assert key in settings
    runs = report["runs"]
    assert [r["seed"] for r in runs] == list(TRAINING_SEEDS)
    savings, estimates = [], []
    for row in runs:
        baseline = np.asarray(row["original_returns"])
        candidate = np.asarray(row["double_returns"])
        assert baseline.shape == candidate.shape == (10,)
        assert np.all(np.isfinite(baseline)) and np.all(np.isfinite(candidate))
        assert np.isfinite(row["original_starting_estimate"]), "nonfinite original starting estimate"
        assert np.isfinite(row["double_starting_estimate"]), "nonfinite double starting estimate"
        savings.append(candidate - baseline)
        estimates.append(
            dict(
                seed=row["seed"],
                original_estimate=row["original_starting_estimate"],
                original_realized=float(((1 - gamma**baseline) / (1 - gamma)).mean()),
                double_estimate=row["double_starting_estimate"],
                double_realized=float(((1 - gamma**candidate) / (1 - gamma)).mean()),
            )
        )
    # adopt uses savings = baseline_cost - candidate_cost. Negate returns to
    # use the chapter's same paired rule when higher return is the improvement.
    adoption = adopt(dict(baseline_costs=np.zeros(10), seeds=[(-s).tolist() for s in savings]))
    return dict(adopt=adoption, estimates=estimates)


if __name__ == "__main__":
    source = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("exercise_double_learning", source)
    if spec is None or spec.loader is None:
        raise ValueError("candidate must be an importable Python file")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    check_target(candidate.double_target)
    if len(sys.argv) > 2:
        print(json.dumps(check_report(json.loads(Path(sys.argv[2]).read_text())), sort_keys=True))
    else:
        print(
            "target checks pass; five-seed report, rerun bytes and unchanged chapter tests still required"
        )
