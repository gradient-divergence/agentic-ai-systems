"""Exercise 6 acceptance; select a submission with SEMI_SELF_SUPERVISED_SOLUTION.

The unfinished starter fails deliberately. The submission keeps self_train's
signature and returns (model, rounds), with share_one, base_rate, guard and total
in each round. guard is 'stop' or 'continue'; total counts labels actually added.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from demos.semi_self_supervised_demo import inputs, prepare
from demos.supervised_learning_demo import arrays
from refund_agent.split import GoldenSet

PATH = Path(os.environ.get("SEMI_SELF_SUPERVISED_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("semi_self_supervised_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)


@pytest.mark.parametrize(
    "pool,share,guard",
    [
        ([0.1, 0.2, 0.3, 0.4], 0.0, "stop"),
        ([9.6, 9.7, 9.8, 9.9], 1.0, "stop"),
        ([0.1, 0.2, 0.3, 9.9], 0.25, "continue"),
        ([0.1, 9.7, 9.8, 9.9], 0.75, "continue"),
    ],
)
def test_guard_precedes_refit_and_includes_both_band_edges(pool, share, guard, capsys):
    X, y = np.array([[0.0], [10.0]]), np.array([0, 1])
    model, report = SOLUTION.self_train(
        X, y, np.array(pool)[:, None], factory=lambda: KNeighborsClassifier(n_neighbors=1)
    )
    assert len(report) == 1
    assert report[0]["share_one"] == share and report[0]["base_rate"] == 0.5
    assert report[0]["guard"] == guard
    added = 0 if guard == "stop" else 4
    assert report[0]["total"] == added and model.n_samples_fit_ == 2 + added
    output = capsys.readouterr().out
    assert f"share_one {share:.4f}" in output and "base_rate 0.5000" in output
    assert f"guard {guard}" in output


def test_refused_seed_seven_stops_9277_labels_and_reads_neither_frozen_month(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("the submission read a frozen month")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    window, pools, august, september, _ = prepare()
    X, y, _ = arrays(window["training"])
    pool = inputs(pools["refused tier 1"])
    results, outputs = [], []
    for _ in range(2):
        model, report = SOLUTION.self_train(X, y, pool)
        results.append((report, model.predict_proba(X[:20])))
        outputs.append(capsys.readouterr().out)
        assert len(report) == 1 and report[0]["accepted"] == 9277
        assert report[0]["share_one"] == 0 and report[0]["base_rate"] == pytest.approx(
            0.03624873, abs=1e-7
        )
        assert report[0]["guard"] == "stop" and report[0]["total"] == 0
        assert august.log == [] and september.log == []
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert (
        outputs[0]
        == outputs[1]
        == ("round 1: proposed 9277; share_one 0.0000; base_rate 0.0362; guard stop\n")
    )


def test_later_round_keeps_the_original_labeled_base_rate():
    X = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    pool = np.array([0.1, 1.1, 1.5, 8.2, 4.1, 5.7, 5.8, 5.9])[:, None]
    model, report = SOLUTION.self_train(
        X, y, pool, factory=lambda: KNeighborsClassifier(n_neighbors=3)
    )
    assert len(report) == 2
    assert [
        (r["accepted"], r["share_one"], r["base_rate"], r["guard"], r["total"]) for r in report
    ] == [(4, 0.25, 0.5, "continue", 4), (4, 0.75, 0.5, "continue", 8)]
    # After round 1 the combined set's rate is 0.4. Substituting it for the
    # original labeled rate would wrongly stop round 2 at its upper edge.
    assert model.n_samples_fit_ == 14
