"""Exercise 6 accepts the fitted-sample identity, not the population formula.

The trained log-odds must match reference log-odds plus the fitted comparison
gap divided by beta. The population discrepancy is a reported diagnostic;
its tolerance is not an exercise pass condition. The starter is unfinished.

Run with the companion Python, -p no:cacheprovider. RLHF_ALIGNMENT_SOLUTION
selects a submitted Python file; the demo contains the reference solution.
These tests are outside the default suite under companion/tests/.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.preference_demo import KINDS, exercise_windows, preference_probability  # noqa: E402
from demos.supervised_learning_demo import (  # noqa: E402
    calibration_error,
    logit,
    platt,
    reliability,
)

PATH = Path(os.environ.get("RLHF_ALIGNMENT_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("exercise_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOLUTION
SPEC.loader.exec_module(SOLUTION)


@pytest.fixture(scope="module")
def report():
    return SOLUTION.exercise_six(seed=7)


@pytest.fixture(scope="module")
def checked(report):
    training, calibration, later = exercise_windows(7)

    def rates(rows, label):
        return np.array([rows[label][rows["kind"] == k].mean() for k in range(5)])

    reference = logit(rates(training["labeled"], "true_cause"))
    target = reference + logit(rates(training["pairs"], "chosen")) / 0.5
    p = preference_probability(np.asarray(report["logits"]))
    _, slope, intercept = platt(p[calibration["kind"]], calibration["true_cause"])
    mapped = preference_probability(report["slope"] * logit(p) + report["intercept"])

    def error(q):
        prediction = q[later["kind"]]
        table = [
            reliability(
                later["true_cause"][later["kind"] == k], prediction[later["kind"] == k], bins=1
            )[0]
            for k in range(5)
        ]
        return calibration_error(table)

    return dict(
        target=target,
        reference_choices=(reference >= 0).astype(int).tolist(),
        tuned_choices=(p >= 0.5).astype(int).tolist(),
        mapped_choices=(mapped >= 0.5).astype(int).tolist(),
        slope=slope,
        intercept=intercept,
        reference_error=error(preference_probability(reference)),
        before_error=error(p),
        after_error=error(mapped),
    )


def test_trained_logits_match_the_fitted_sample_identity(report, checked):
    """Accept reference log-odds + fitted comparison gap / beta within 0.02."""
    np.testing.assert_allclose(
        report["logits"],
        checked["target"],
        rtol=0,
        atol=0.02,
        err_msg="Exercise 6 accepts the fitted-sample identity; population gaps are diagnostic only",
    )
    np.testing.assert_allclose(report["target_logits"], checked["target"], rtol=0, atol=1e-9)
    assert report["beta"] == 0.5


def test_population_specialization_is_reported_separately(report):
    """Check the diagnostic's accuracy without requiring the population tolerance."""
    gaps = abs(np.asarray(report["logits"]) - 3 * logit(np.asarray(KINDS.shares)))
    diagnostic = report["population_check"]
    np.testing.assert_allclose(diagnostic["logit_gaps"], gaps, rtol=0, atol=1e-9)
    assert diagnostic["max_logit_gap"] == pytest.approx(float(gaps.max()), abs=1e-9)
    assert diagnostic["tolerance"] == 0.02
    assert diagnostic["passed"] == bool(gaps.max() <= 0.02)


def test_platt_slope_reverses_threefold_log_odds(report, checked):
    assert report["slope"] == pytest.approx(1 / 3, abs=0.05)
    assert report["slope"] == pytest.approx(checked["slope"], abs=1e-6)
    assert report["intercept"] == pytest.approx(checked["intercept"], abs=1e-6)


def test_map_calibration_is_checked_on_a_later_window(report, checked):
    for key in ("reference_error", "before_error", "after_error"):
        assert report[key] == pytest.approx(checked[key], abs=1e-9)
    assert abs(checked["after_error"] - checked["reference_error"]) < 0.02
    assert checked["after_error"] < checked["before_error"]
    assert report["window_order"] == ["training", "calibration", "later check"]
    assert report["calibration_reports"] == report["later_reports"] == 10000


def test_every_most_likely_cause_is_preserved(report, checked):
    for key in ("reference_choices", "tuned_choices", "mapped_choices"):
        assert report[key] == checked[key]
    assert checked["reference_choices"] == checked["tuned_choices"] == checked["mapped_choices"]


def test_seed_seven_reproduces_every_digit(report):
    assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(
        SOLUTION.exercise_six(seed=7), sort_keys=True, allow_nan=False
    )
