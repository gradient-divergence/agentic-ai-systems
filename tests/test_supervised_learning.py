"""Tests for the listings and outputs of the chapter on supervised learning."""

import importlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

pytest.importorskip("sklearn")

from refund_agent import contracts, rules
from refund_agent.data import history
from refund_agent.data.history import generate_history

COMPANION = Path(__file__).resolve().parents[1]
DEMO = COMPANION / "demos" / "supervised_learning_demo.py"
CHAPTER = COMPANION.parent / "mainmatter" / "supervised-learning.qmd"
OUTPUTS = COMPANION / "book_outputs"
FIXTURE = COMPANION / "demos" / "data" / "supervised_learning_recalibration_07.json"
REGIONS = (
    "split-01",
    "fit-01",
    "boost-01",
    "fit-02",
    "biasvar-01",
    "decide-01",
    "calibrate-01",
    "trace-01",
    "trace-02",
    "trace-03",
    "august-01",
    "record-01",
    "record-02",
    "gate-01",
    "bootstrap-01",
    "sweep-01",
    "shift-01",
    "shift-02",
    "numbers-01",
    "keys-01",
    "figures-01",
)
# The regions the chapter prints as listings, in the order it prints them.
PRINTED = ("split-01", "fit-01", "boost-01", "decide-01", "calibrate-01", "shift-01")
ONE_THREAD = {
    name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
}
MARK = "=== after the run ==="
# Runs the demo as a script in a fresh interpreter, as the output gate does, then
# prints what the tests read from its globals. `{patch}` may replace the generator.
WRAPPER = """
import json, runpy, sys
sys.path.insert(0, {companion!r})
import refund_agent.data.history as h
{patch}
ns = runpy.run_path({demo!r}, run_name="__main__")
print({mark!r})
frozen = {{name: ns[name] for name in ("august", "september")}}
print(json.dumps({{
    "planted": ns["planted"],
    "manifests": {{name: golden.manifest for name, golden in frozen.items()}},
    "logs": {{name: golden.log for name, golden in frozen.items()}},
    "intact": {{name: golden.intact() for name, golden in frozen.items()}},
    "window": sorted(ns["window"]),
    "trace": ns["decision_snapshot"],
    "trace_self_training": ns["trace_self_training"],
    "trace_logs_before": ns["trace_logs_before"],
    "trace_logs_after": ns["trace_logs_after"],
    "trace_selection": {{key: ns["trace_selection"][key] for key in ("selected", "grants", "signer")}},
    "bootstrap_costs": {{"placeholder": ns["placeholder_cost"].tolist(), "tier": ns["tier_cost"].tolist()}},
}}))
"""
SCRAMBLE = """
original = h.generate_history
cut = h.datetime(2026, 8, 1, tzinfo=h.timezone.utc)
def scrambled(seed=7, rows=h.DEFAULT_ROWS):
    flip = lambda r: r.model_copy(update={"found_false_30d": 1 - r.found_false_30d})
    return [flip(r) if r.paid and r.at >= cut else r for r in original(seed, rows)]
h.generate_history = scrambled
"""


def demo():
    return importlib.import_module("demos.supervised_learning_demo")


def region(source: str, name: str) -> str:
    """The lines of one book region of the demo, as the chapter would print them."""
    opened = re.search(rf"^# region book:fnd\.supervised-learning\.{name}\b.*\n", source, re.M)
    closed = source.index(f"# endregion book:fnd.supervised-learning.{name}\n")
    assert opened and opened.end() < closed, name
    return source[opened.end() : closed]


def output(name: str) -> str:
    return (OUTPUTS / f"_fnd.supervised-learning.{name}.txt").read_text(encoding="utf-8")


def run_script(patch: str = "") -> tuple[str, dict]:
    code = WRAPPER.format(companion=str(COMPANION), demo=str(DEMO), mark=MARK, patch=patch)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    printed, _, after = result.stdout.partition(MARK + "\n")
    return printed, json.loads(after)


@pytest.fixture(scope="module")
def run() -> tuple[str, dict]:
    return run_script()


def test_import_prints_nothing_and_writes_nothing(tmp_path: Path) -> None:
    code = (
        "import contextlib, io, os, sys\n"
        f"sys.path.insert(0, {str(COMPANION)!r})\n"
        "buffer = io.StringIO()\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    import demos.supervised_learning_demo as d\n"
        "    from demos.supervised_learning_demo import windows, platt, boosted, frozen\n"
        "assert buffer.getvalue() == '', buffer.getvalue()\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
        "# Importing freezes nothing, reads nothing and fits nothing.\n"
        "for name in ('history', 'window', 'august', 'september', 'models', 'maps'):\n"
        "    assert not hasattr(d, name), name\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_windows_follow_the_day_each_label_became_final() -> None:
    d = demo()
    paid = [r for r in generate_history(rows=6000) if r.paid]
    window = d.windows(paid)
    assert sum(len(rows) for rows in window.values()) == len(paid)
    assert all(r.label_final_at <= d.FIT_BY for r in window["training"])
    assert all(d.FIT_BY < r.label_final_at <= d.GO_LIVE for r in window["validation"])
    assert all(r.at < d.GO_LIVE < r.label_final_at for r in window["open"])
    assert all(d.GO_LIVE <= r.at < d.SHIFT for r in window["test"])
    assert all(r.at >= d.SHIFT for r in window["shifted"])
    # Nothing fitted may use a label still open when the score starts work.
    fitted = window["training"] + window["validation"]
    assert max(r.label_final_at for r in fitted) <= d.GO_LIVE
    # The boundary cases: final exactly at a cut belongs to the earlier window.
    row = window["training"][0]
    month = timedelta(days=30)
    at_fit = row.model_copy(update={"at": d.FIT_BY - month, "label_final_at": d.FIT_BY})
    at_live = row.model_copy(update={"at": d.GO_LIVE - month, "label_final_at": d.GO_LIVE})
    assert d.windows([at_fit])["training"] == [at_fit]
    assert d.windows([at_live])["validation"] == [at_live]
    assert (d.FIT_BY, d.GO_LIVE, d.SHIFT) == tuple(
        datetime(2026, m, 1, tzinfo=timezone.utc) for m in (7, 8, 9)
    )
    assert d.SHIFT == history.DRIFT_AT


def test_a_frozen_month_has_a_manifest_a_fingerprint_and_a_logged_read() -> None:
    d = demo()
    paid = [r for r in generate_history(rows=6000) if r.paid]
    month = d.windows(paid)["test"]
    golden = d.frozen(month, "history-august")
    # The outcome label in the fields of chapter 4's label policy.
    policy = d.OUTCOME_POLICY
    assert (policy.version, policy.window_days) == ("outcome-1", 30)
    assert policy.exclusions == ("requests that were not paid",)
    assert (
        policy.effective_at == history.START_AT and "found false within 30 days" in policy.records
    )
    manifest = golden.manifest
    assert manifest["name"] == "history-august" and manifest["label_policy"] == "outcome-1"
    assert manifest["requests"] == len(month) and manifest["by_tier"] == {1: len(month)}
    assert manifest["by_policy_version"] == {"2026-07-01": len(month)}
    assert (manifest["first"], manifest["last"]) == ("2026-08-01", "2026-08-31")
    assert golden.intact() and golden.log == []
    # The rows keep their order, and each label is its row's own outcome.
    assert list(golden.requests) == month
    for row, label in zip(golden.requests, golden.labels):
        assert (label.request_id, label.found_false_30d) == (row.request_id, row.found_false_30d)
        assert label.label_final_at == row.label_final_at == row.at + timedelta(days=30)
        assert label.tier == 1
    # A read is logged, and returns the arrays of exactly those rows.
    X, y, amount = d.consulted(golden, "one read")
    assert golden.log == [
        {"entry": 1, "set": "history-august", "reader": "stage 1 report", "purpose": "one read"}
    ]
    for got, want in zip((X, y, amount), d.arrays(month)):
        np.testing.assert_array_equal(got, want)
    d.consulted(golden, "a second read")
    assert [entry["entry"] for entry in golden.log] == [1, 2]
    # One flipped outcome, or one changed amount, is another cohort: the fingerprint moves.
    first = month[0]
    flipped = first.model_copy(update={"found_false_30d": 1 - first.found_false_30d})
    repriced = first.model_copy(update={"amount_cents": first.amount_cents + 1})
    for changed in (flipped, repriced):
        other = d.frozen([changed, *month[1:]], "history-august")
        assert other.manifest["sha256"] != manifest["sha256"] and other.intact()
    assert d.frozen(month, "history-august").manifest == manifest  # the same rows, the same record
    # September's rows were decided under the next policy version.
    september = d.frozen(d.windows(paid)["shifted"], "history-september")
    assert set(september.manifest["by_policy_version"]) == {"2026-09-01"}
    assert september.manifest["sha256"] != manifest["sha256"]


def test_arrays_fill_a_missing_scan_and_read_only_the_allowlist() -> None:
    d = demo()
    rows = [r for r in generate_history(rows=400) if r.paid]
    X, y, amount = d.arrays(rows)
    assert X.shape == (len(rows), 8) and not np.isnan(X).any()
    assert set(y.tolist()) <= {0, 1}
    assert amount.max() < 50 and amount.min() > 0  # tier 1 only
    changed = [r.model_copy(update={"found_false_30d": 1 - r.found_false_30d}) for r in rows]
    np.testing.assert_array_equal(d.arrays(changed)[0], X)


def test_log_loss_values_and_the_base_rate_floor() -> None:
    d = demo()
    assert d.log_loss(np.array([1]), np.array([0.9])) == pytest.approx(0.1054, abs=5e-5)
    assert d.log_loss(np.array([1]), np.array([0.1])) == pytest.approx(2.3026, abs=5e-5)
    y = np.array([1] * 499 + [0] * 13267)
    rate = y.mean()
    entropy = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
    assert d.log_loss(y, np.full(len(y), rate)) == pytest.approx(entropy)
    assert entropy == pytest.approx(0.1558, abs=5e-5)
    # The base rate minimizes the loss among constant predictions.
    assert all(d.log_loss(y, np.full(len(y), q)) > entropy for q in (0.02, 0.03, 0.05, 0.1))


@pytest.mark.parametrize(
    "y,p,expected", [(1, 1.0, 0.0), (0, 0.0, 0.0), (1, 0.0, math.inf), (0, 1.0, math.inf)]
)
def test_log_loss_at_endpoints_uses_the_observed_class(y, p, expected) -> None:
    with np.errstate(all="raise"):
        assert demo().log_loss(np.array([y]), np.array([p])) == expected


def test_log_loss_mixes_correct_endpoints_and_uncertain_predictions() -> None:
    with np.errstate(all="raise"):
        loss = demo().log_loss(np.array([0, 1, 1, 0]), np.array([0.0, 1.0, 0.8, 0.3]))
    assert loss == pytest.approx((-math.log(0.8) - math.log(0.7)) / 4)


def test_three_trees_fit_what_the_last_left() -> None:
    d = demo()
    xs, ys = [1, 2, 3, 4, 5, 6], [0, 0, 4, 4, 10, 10]
    F, cuts, errors = [0.0] * 6, [], []
    for _ in range(3):
        cut, tree = d.stump(xs, [y - f for y, f in zip(ys, F)])
        F = [f + tree(x) for x, f in zip(xs, F)]
        cuts.append(cut)
        errors.append(sum((y - f) ** 2 for y, f in zip(ys, F)))
    assert cuts == [4, 2, 4] and errors == [16, 4, 1]
    assert F == [0.5, 0.5, 3.5, 3.5, 10.0, 10.0]


def test_expected_costs_of_the_router_and_the_refund() -> None:
    d = demo()
    exact = [[Fraction(c) for c in row] for row in d.ROUTER]
    tickets = {
        ("0.90", "0.06", "0.04"): (("0.64", "1.92", "4.56"), "standard"),
        ("0.75", "0.05", "0.20"): (("2.20", "2.10", "3.80"), "priority"),
        ("0.21", "0.00", "0.79"): (("7.90", "2.79", "1.05"), "escalation"),
    }
    for ticket, (costs, route) in tickets.items():
        got = d.expected_costs([Fraction(p) for p in ticket], exact)
        assert got == [Fraction(c) for c in costs]
        assert d.QUEUES[got.index(min(got))] == route
    # Direction matters: escalation sent to standard costs 10, the reverse 5.
    assert d.ROUTER[2][0] == 10 and d.ROUTER[0][2] == 5
    # Standard and priority tie at an escalation probability of 2/9 with no priority.
    tie = d.expected_costs([Fraction(7, 9), Fraction(0), Fraction(2, 9)], exact)
    assert tie[0] == tie[1] == Fraction(20, 9)
    refund = [[0, 4.0, 20.0], [49.99, 4.0, 0]]
    pay, refer, deny = d.expected_costs([0.88, 0.12], refund)
    assert (round(pay, 2), refer, round(deny, 2)) == (6.00, 4.00, 17.60)


def test_band_edges_and_the_tier_threshold() -> None:
    d = demo()
    assert d.band(4.0, 40.0, 20.0) == pytest.approx((0.100, 0.800))
    assert d.band(4.0, 49.99, 20.0)[0] == pytest.approx(0.080, abs=5e-4)
    low, high = d.band(4.0, 5.0, 20.0)
    assert low == pytest.approx(high)  # the edges meet at $5: no band below it
    assert d.TIER_THRESHOLD == round(d.REVIEW / 49.99, 3) == 0.080
    for amount, risk in ((40.0, 0.05), (40.0, 0.5), (40.0, 0.9), (10.0, 0.3)):
        costs = d.expected_costs([1 - risk, risk], [[0, 4.0, 20.0], [amount, 4.0, 0]])
        low, high = d.band(4.0, amount, 20.0)
        cheapest = costs.index(min(costs))
        assert cheapest == (0 if risk < low else 2 if risk > high else 1)


def test_gate_and_realized_cost() -> None:
    d = demo()
    score = np.array([0.02, 0.079, 0.080, 0.30])
    amount = np.array([49.0, 49.0, 10.0, 10.0])
    np.testing.assert_array_equal(d.gate(score), [True, True, False, False])
    np.testing.assert_array_equal(d.gate(score, amount), [True, True, True, True])
    np.testing.assert_array_equal(d.gate(np.array([0.5]), np.array([8.0])), [False])
    y = np.array([0, 1, 1, 0])
    cost = d.realized_cost(y, amount, d.gate(score))
    np.testing.assert_allclose(cost, [0.0, 49.0, 4.0, 4.0])  # nothing is denied on a score


def test_reliability_and_calibration_error() -> None:
    d = demo()
    example = [(600, 0.05, 24), (300, 0.20, 72), (100, 0.60, 50)]
    assert d.calibration_error(example) == pytest.approx(0.028)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.4, 5000)
    y = (rng.random(5000) < p).astype(int)
    table = d.reliability(y, p)
    assert len(table) == 5 and sum(n for n, _, _ in table) == 5000
    assert [n for n, _, _ in table] == [1000] * 5
    assert sum(wrong for _, _, wrong in table) == int(y.sum())
    assert d.calibration_error(table) < 0.02  # a score that is calibrated by construction
    assert d.calibration_error(d.reliability(y, np.clip(3 * p, 0, 0.99))) > 0.2
    # Equal scores stay in one bin, and an empty bin is dropped.
    tied = d.reliability(np.array([0, 1, 0, 1, 1, 0]), np.array([0.1, 0.1, 0.1, 0.5, 0.6, 0.9]))
    assert [n for n, _, _ in tied] == [3, 1, 2]
    assert tied[0] == (3, pytest.approx(0.1), 1)


def test_pooled_calibration_does_not_license_an_amount_dependent_rule() -> None:
    d = demo()
    # Equal numbers of $49.99 requests at a true risk of 0.12 and $5 requests at zero,
    # every one scored 0.06: the pooled record is perfect, and the amount rows are not.
    amount = np.array([49.99] * 500 + [5.00] * 500)
    y = np.array([1] * 60 + [0] * 440 + [0] * 500)
    score = np.full(1000, 0.06)
    pooled = d.reliability(y, score)
    assert pooled == [(1000, pytest.approx(0.06), 60)]
    assert d.calibration_error(pooled) == pytest.approx(0.0)
    under, over = d.by_amount(y, score, amount)
    assert under == ("Under \\$25", 500, pytest.approx(0.06), 0)
    assert over == ("\\$25 and over", 500, pytest.approx(0.06), 60)
    # Both rules pay every request, though paying a $49.99 one risks $6 against a $4 review.
    assert d.gate(score).all() and d.gate(score, amount).all()
    assert 0.06 * 49.99 < d.REVIEW < 0.12 * 49.99 == pytest.approx(6.00, abs=0.005)
    paid_all = d.realized_cost(y, amount, d.gate(score, amount)).mean()
    by_true_risk = d.realized_cost(y, amount, amount < 25).mean()
    assert paid_all == pytest.approx(60 * 49.99 / 1000) and by_true_risk == pytest.approx(2.0)
    # The record's rows: five score bins, then the same requests cut at $25.
    rng = np.random.default_rng(5)
    p = rng.uniform(0.01, 0.3, 2000)
    y = (rng.random(2000) < p).astype(int)
    amount = rng.uniform(1, 49.99, 2000)
    rows = d.record_table(y, p, amount)
    assert rows[0] == "| Requests | Count | Mean score | Wrong | Rate, 95% interval |"
    assert len(rows[0]) <= 79
    names = [row.split("|")[1].strip() for row in rows[2:]]
    assert names == [f"Score bin {k}" for k in range(1, 6)] + ["Under \\$25", "\\$25 and over"]
    counts = [int(row.split("|")[2]) for row in rows[2:]]
    assert sum(counts[:5]) == sum(counts[5:]) == 2000
    wrong = [int(row.split("|")[4]) for row in rows[2:]]
    assert sum(wrong[:5]) == sum(wrong[5:]) == int(y.sum())


def test_one_half_needs_equal_increments_not_equal_mistake_costs() -> None:
    d = demo()

    def threshold(costs):  # rows: sound claim, wrong payment; columns: pay, refer
        adds = costs[0][1] - costs[0][0], costs[1][0] - costs[1][1]
        return adds[0] / (adds[0] + adds[1])

    # An $8 refund with a $4 review: the mistakes cost $4 and $8, the increments $4 and $4.
    refund = [[0, 4.0], [8.0, 4.0]]
    assert threshold(refund) == 0.5 == d.band(4.0, 8.0, 20.0)[0]
    pay, refer = d.expected_costs([0.5, 0.5], refund)
    assert pay == refer == 4.0
    # Equal mistake costs give one half only when right decisions cost nothing.
    assert threshold([[0, 5.0], [5.0, 0]]) == 0.5
    assert threshold([[0, 5.0], [5.0, 1.0]]) == pytest.approx(5 / 9)
    # The threshold is the risk at which the two actions' expected costs cross.
    for costs in (refund, [[0, 4.0], [49.99, 4.0]], [[1.0, 3.0], [9.0, 2.0]]):
        t = threshold(costs)
        at, below, above = (d.expected_costs([1 - q, q], costs) for q in (t, t - 0.01, t + 0.01))
        assert at[0] == pytest.approx(at[1]) and below[0] < below[1] and above[0] > above[1]


def test_roc_auc_counts_a_tie_as_half() -> None:
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(6)
    y = (rng.random(400) < 0.3).astype(int)
    for score in (rng.random(400), np.round(rng.random(400), 1), np.zeros(400)):
        gaps = score[y == 1][:, None] - score[y == 0][None, :]
        by_pairs = (gaps > 0).mean() + (gaps == 0).mean() / 2
        assert roc_auc_score(y, score) == pytest.approx(by_pairs)
    assert roc_auc_score(y, np.zeros(400)) == 0.5  # every pair tied


def test_a_two_parameter_map_judged_where_it_was_fit_can_still_be_far_off() -> None:
    d = demo()
    # Five groups whose risk does not rise with the score: no sigmoid of the log-odds
    # fits them, so even on its own fitting rows the map's calibration error stays large.
    score = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 400)
    y = np.concatenate([[1] * k + [0] * (400 - k) for k in (40, 280, 40, 280, 120)])
    repair, _, _ = d.platt(score, y)
    own_window = d.reliability(y, repair(score))
    assert len(own_window) == 5 and d.calibration_error(own_window) > 0.2
    # An isotonic map judged on its own rows reports zero, whatever happens next.
    pooled = d.isotonic(score, y)(score)
    assert d.calibration_error(d.reliability(y, pooled)) == pytest.approx(0.0)


def test_platt_recovers_a_known_distortion() -> None:
    d = demo()
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.01, 0.3, 60000)
    y = (rng.random(len(truth)) < truth).astype(int)
    distorted = 1 / (1 + np.exp(-(2.0 * d.logit(truth) + 1.0)))  # logit = 2 * truth's + 1
    repair, w1, w0 = d.platt(distorted, y)
    assert w1 == pytest.approx(0.5, abs=0.03) and w0 == pytest.approx(-0.5, abs=0.06)
    assert np.abs(repair(distorted) - truth).max() < 0.02
    # With an intercept, the mapped score's mean equals the window's observed rate.
    assert repair(distorted).mean() == pytest.approx(y.mean(), abs=1e-4)


def test_shift_base_rate_is_elkans_second_theorem() -> None:
    d = demo()
    assert d.shift_base_rate(0.5, 0.20, 0.01) == pytest.approx(0.039, abs=5e-4)
    assert d.shift_base_rate(0.9, 0.20, 0.01) == pytest.approx(0.267, abs=5e-4)
    rng = np.random.default_rng(2)
    for p, b, b2 in rng.uniform(0.01, 0.99, size=(200, 3)):
        theorem = b2 * (p - p * b) / (b - p * b + b2 * p - b * b2)  # Elkan (2001), theorem 2
        assert d.shift_base_rate(p, b, b2) == pytest.approx(theorem)
        assert d.shift_base_rate(d.shift_base_rate(p, b, b2), b2, b) == pytest.approx(p)
    assert d.shift_base_rate(0.3, 0.1, 0.1) == pytest.approx(0.3)


def test_a_class_weight_multiplies_the_learned_odds() -> None:
    d = demo()
    grid = np.linspace(0.001, 0.999, 99801)
    for mu, omega in ((0.10, 9.0), (0.01, 26.6), (0.0362, 26.59)):
        weighted_loss = -(omega * mu * np.log(grid) + (1 - mu) * np.log(1 - grid))
        best = grid[weighted_loss.argmin()]
        assert d.odds(best) == pytest.approx(omega * d.odds(mu), rel=2e-3)
        assert d.shift_base_rate(best, 0.5, 1 / (1 + omega)) == pytest.approx(mu, rel=2e-3)
    learned = 9 * d.odds(0.10) / (1 + 9 * d.odds(0.10))
    assert learned == pytest.approx(0.50)


def test_the_bias_variance_identity_and_its_estimator() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        mu = rng.uniform(0.01, 0.99)
        drawn = rng.uniform(0.01, 0.99, 7)
        parts = mu * (1 - mu) + (drawn.mean() - mu) ** 2 + drawn.var()
        direct = np.mean(mu * (1 - drawn) ** 2 + (1 - mu) * drawn**2)
        assert parts == pytest.approx(direct)
    d = demo()
    pool = [r for r in generate_history(rows=3000) if r.paid]
    points = [d.example_request(pool, "damaged", low=1, high=4999)]
    truth = d.planted_risk(points[0])

    class Constant:
        """A model that ignores its training set: no variance, a known bias."""

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.75), np.full(len(X), 0.25)])

    rows = d.bias_variance(
        {"constant": Constant, "boosted": d.boosted}, pool, points, sets=6, size=400
    )
    (name, mu, mean, bias2, variance, drawn), boosted = rows
    assert name == "constant" and mu[0] == pytest.approx(truth)
    assert variance[0] == pytest.approx(0.0) and bias2[0] == pytest.approx((0.25 - truth) ** 2)
    assert drawn.shape == (6, 1) and (drawn == 0.25).all()  # one prediction for each set
    assert boosted[4][0] > 0 and boosted[3][0] >= 0


def test_wilson_interval_and_bootstrap() -> None:
    d = demo()
    assert d.wilson_interval(10, 100) == pytest.approx((0.0552, 0.1744), abs=5e-5)
    assert d.wilson_interval(0, 15) == pytest.approx((0.0, 0.2039), abs=5e-5)
    values = np.arange(100.0)
    mean, low, high = d.bootstrap(values)
    assert mean == 49.5 and low < 49.5 < high
    assert d.bootstrap(values) == (mean, low, high)  # seeded


@pytest.fixture(scope="module")
def trace_fits():
    d = demo()
    window = d.windows([r for r in generate_history(seed=7) if r.paid])
    frozen = {
        name: d.frozen(window.pop(key), name)
        for name, key in (("august", "test"), ("september", "shifted"))
    }
    Xt, yt, _ = d.arrays(window["training"])
    Xv, yv, _ = d.arrays(window["validation"])
    logistic = d.make_pipeline(d.StandardScaler(), d.LogisticRegression(max_iter=1000)).fit(Xt, yt)
    trees = d.boosted().fit(Xt, yt)
    mapping = d.platt(d.risk(trees, Xv), yv)[0]
    return window, logistic, trees, mapping, frozen


def test_request_trace_uses_the_fits_without_outcomes_or_frozen_reads(trace_fits) -> None:
    d = demo()
    assert callable(getattr(d, "request_trace", None)), "C1's per-request trace is missing"
    window, logistic, trees, mapping, frozen = trace_fits
    row, snapshot = d.request_trace(window["open"], logistic, trees, mapping)
    assert row.request_id == "history-r-7-030054"
    assert row.at < d.GO_LIVE < row.label_final_at and row.amount_cents >= 2500
    X = d.feature_matrix([row])
    standardized = logistic[0].transform(X)[0]
    fit = snapshot["logistic"]
    np.testing.assert_allclose([item["input"] for item in fit["inputs"]], standardized)
    np.testing.assert_allclose(
        [item["coefficient"] for item in fit["inputs"]], logistic[1].coef_[0]
    )
    contributions = [item["contribution"] for item in fit["inputs"]]
    np.testing.assert_allclose(contributions, standardized * logistic[1].coef_[0])
    assert fit["log_odds"] == pytest.approx(sum(contributions) + fit["intercept"])
    assert fit["probability"] == pytest.approx(d.risk(logistic, X)[0])
    raw = d.risk(trees, X)[0]
    assert snapshot["boosted"] == pytest.approx({"as_trained": raw, "mapped": mapping(raw)})
    assert snapshot["boosted"]["mapped"] >= d.TIER_THRESHOLD and raw <= 0.1
    assert snapshot["costs"] == pytest.approx(
        {"pay": mapping(raw) * row.amount_cents / 100, "review": d.REVIEW}
    )
    assert snapshot["drafts"] == {"tier_threshold": "refer", "review_cost": "pay"}
    assert snapshot["permitted_action"] == "drafts only; a person executes"
    assert {item["feature"] for item in fit["inputs"]} == {
        "amount_cents",
        "damaged",
        "changed_mind",
        "delivery_age_days",
        "prior_refund_count",
        "paid_cents",
        "order_evidence_age_seconds",
        "shipment_evidence_age_seconds",
    }
    changed = [
        r.model_copy(update={"found_false_30d": 1 - r.found_false_30d})
        for r in reversed(window["open"])
    ]
    same_row, same_snapshot = d.request_trace(changed, logistic, trees, mapping)
    assert same_row.request_id == row.request_id and same_snapshot == snapshot
    assert all(
        name not in json.dumps(snapshot)
        for name in ("found_false_30d", "label_final_at", "pseudo_label")
    )
    assert all(not golden.log and golden.intact() for golden in frozen.values())


def test_bootstrap_means_reproduce_the_same_2000_paired_draws() -> None:
    d = demo()
    assert callable(getattr(d, "bootstrap_means", None)), "C3's resampled means are missing"
    values = np.array([0, 0, 0, -4, 36])
    indices = np.random.default_rng(7).integers(0, len(values), size=(2000, len(values)))
    means = d.bootstrap_means(values)
    np.testing.assert_array_equal(means, values[indices].mean(axis=1))
    assert d.bootstrap(values) == pytest.approx(
        (values.mean(), *np.quantile(means, [0.025, 0.975]))
    )


def test_trace_matches_self_training_and_the_selected_delivery(trace_fits) -> None:
    from demos import delegated_model_demo as selection
    from demos import semi_self_supervised_demo as semi

    d = demo()
    window, logistic, trees, mapping, _ = trace_fits
    row, snapshot = d.request_trace(window["open"], logistic, trees, mapping)
    pool = semi.inputs(window["open"])
    probabilities = trees.predict_proba(pool)
    index = next(
        i for i, candidate in enumerate(window["open"]) if candidate.request_id == row.request_id
    )
    confident = probabilities.max(axis=1) >= 0.9
    assert confident[index] and trees.classes_[probabilities[index].argmax()] == 0
    Xt, yt, _ = d.arrays(window["training"])
    _, rounds = semi.self_train(Xt, yt, pool, rounds=1)
    assert rounds[0]["accepted"] == int(confident.sum())
    labels = trees.classes_[probabilities[confident].argmax(axis=1)]
    assert rounds[0]["by_class"] == {
        int(label): int((labels == label).sum()) for label in trees.classes_
    }
    record = json.loads((OUTPUTS / "_fnd.automl-and-selection.record-full-01.txt").read_text())
    assert (
        record["selected"] == selection.BASELINE
        and record["grants"] == "none; drafts only, a person executes"
    )
    delivery = next(item for item in record["deliveries"] if item["name"] == record["selected"])
    baseline = selection.fit_delivery(
        selection.BASELINE, "person", delivery["settings"], window, selection.commission_for(window)
    )
    assert baseline.predict([row], mapped=False)[0] == pytest.approx(
        snapshot["boosted"]["as_trained"]
    )
    assert baseline.predict([row])[0] == pytest.approx(snapshot["boosted"]["mapped"])
    assert "refer" == ("pay" if d.gate(baseline.predict([row])[0]) else "refer")


def test_the_placeholder_rule_pays_every_request_in_the_history() -> None:
    order = history._order(7, 0, history.START_AT)
    request = history._request(7, 0, history.START_AT, order)
    state = contracts.State(request=request, trace_id="t")
    assert rules.risk_score(state) == 0.1 < rules._RISK_THRESHOLD == 0.2
    assert "ignore" not in request.text.casefold()


def test_a_map_judged_on_its_own_window_reports_no_error() -> None:
    d = demo()
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.9, 1500)
    y = (rng.random(1500) < p / 6).astype(int)  # the score overstates the risk sixfold
    window = {"decided": "one window", "requests": 1500}
    report = d.recalibration_report(window, window, p, y, p, y)
    assert report["calibration_error"] == 0.0 and report["map_fit_on"] == report["judged_on"]
    later = (rng.random(1500) < p / 6).astype(int)
    honest = d.recalibration_report(window, {"decided": "later"}, p, y, p, later)
    assert honest["calibration_error"] > 0.0


def test_cut_off_sweep_finds_the_cheapest_cut_off() -> None:
    d = demo()
    y = np.array([1, 0, 1, 0])
    amount = np.array([40.0, 40.0, 10.0, 10.0])
    score = np.array([0.5, 0.01, 0.5, 0.01])  # expected losses 20, 0.4, 5, 0.1
    costs = d.cut_off_sweep(y, amount, score, d.CUT_OFFS)
    best, cost, (low, high), at_review = d.describe_sweep(costs, d.CUT_OFFS)
    assert cost == pytest.approx(2.0) and (low, high) == (0.5, 5.0)  # refer both wrong ones
    assert at_review == pytest.approx(2.0) and best == 0.5
    assert costs[-1] == pytest.approx(12.5)  # a cut-off of $60 pays everything


def test_printed_listings_run_in_chapter_order_on_their_own(tmp_path: Path) -> None:
    # The listings the chapter prints, in its order, and nothing else of the demo.
    chapter = CHAPTER.read_text(encoding="utf-8")
    included = re.findall(r"_includes/snippets/_fnd\.supervised-learning\.([\w-]+)\.py", chapter)
    assert tuple(included) == PRINTED
    source = DEMO.read_text(encoding="utf-8")
    script = tmp_path / "printed_listings.py"
    script.write_text("\n\n".join(region(source, name) for name in PRINTED), encoding="utf-8")
    # A reader's setting: the companion importable, and no other line of the demo run.
    env = {**os.environ, **ONE_THREAD, "PYTHONPATH": str(COMPANION)}
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    # Alone, they print what the chapter prints under them, digit for digit.
    assert result.stdout == "".join(output(name) for name in PRINTED)
    # Every other name comes from a printed listing: the one import from the demo is
    # explicit, in the listing that first uses it.
    imports = re.findall(
        r"^from demos\.supervised_learning_demo import (.+?)(?:\s+#.*)?$", source, re.M
    )
    assert imports == ["boosted"]
    assert "from demos.supervised_learning_demo import boosted" in region(source, "calibrate-01")


def test_outputs_match_a_fresh_run(run: tuple[str, dict]) -> None:
    printed, _ = run
    committed = "".join(output(name) for name in REGIONS)
    assert printed == committed
    assert "36,000 requests, 17,704 paid" in printed
    for line in (
        "training   13,766  decided 2025-04-01 to 2026-05-31",
        "validation    988  decided 2026-06-01 to 2026-07-01",
        "test          999  decided 2026-08-01 to 2026-08-31",
        "shifted       966  decided 2026-09-01 to 2026-09-30",
    ):
        assert line in printed


def test_constructed_logistic_step_prints_the_worked_values(run: tuple[str, dict]) -> None:
    printed, _ = run
    for line in (
        "constructed request: log-odds -1.9900; probability 0.1203; wrong-payment loss 2.12; sound-claim loss 0.1281",
        "constructed wrong payment: error -0.88; gradient (-0.88, -0.44); bias gradient -0.88",
        "step 0.1: weights (0.788, 0.664); bias -2.912; probability 0.143; loss 1.95",
    ):
        assert line in printed


def test_ticket_two_under_zero_one_costs_prints_the_likeliest_route(run: tuple[str, dict]) -> None:
    printed, _ = run
    assert (
        "ticket 2 under 0-1 costs: standard 0.25, priority 0.95, escalation 0.80, route standard"
        in printed
    )


def test_constructed_bootstrap_prints_paired_savings_and_repeated_draws(
    run: tuple[str, dict],
) -> None:
    printed, _ = run
    assert "constructed savings (0, 0, 0, -4, 36): mean $6.40" in printed
    assert "resample requests (1, 4, 4, 2, 5): savings (0, -4, -4, 0, 36); mean $5.60" in printed
    assert "chance of missing request 5: (4/5)^5 = 0.3277 (0.33)" in printed


def test_trace_prints_a_separate_later_outcome_and_grants_no_authority(run) -> None:
    printed, after = run
    trace = output("trace-01")
    assert trace in printed
    snapshot, later = trace.split("Decision snapshot ends\n")
    assert "found_false_30d" not in snapshot and "pseudo-label" not in snapshot
    assert (
        "Later observed outcome: final 2026-08-01T02:19:54+00:00; found_false_30d 0; paid cost $0.00"
        in later
    )
    assert "would accept in round 1; pseudo-label 0" in later
    assert later.splitlines()[1].startswith("Self-training on the open pool 985,")
    assert "Chapter" not in trace
    assert (
        "Only the paid outcome is observed; the referral outcome and cost were never recorded."
        in later
    )
    assert (
        "offline tier draft refer" in later
        and "Promotion granted: none; drafts only, a person executes" in later
    )
    assert after["trace_self_training"] == {"accepted_round": 1, "pseudo_label": 0}
    assert (
        after["trace_logs_before"] == after["trace_logs_after"] == {"august": [], "september": []}
    )


@pytest.fixture(scope="module")
def captured_trace():
    tools = vars(_regions)
    names = {f"fnd.supervised-learning.trace-{index:02}" for index in (1, 2, 3)}
    snippets = {
        snippet.snippet_id: snippet
        for snippet in tools["extract_snippets_from_text"](DEMO.read_text(), DEMO)
        if snippet.snippet_id in names
    }
    assert snippets.keys() == names, "C6's marked trace excerpts are missing"
    assert all(snippet.output for snippet in snippets.values())
    return tools["capture_outputs"](snippets, sys.executable)


@pytest.mark.parametrize("name,start,stop", [("trace-02", 3, 18), ("trace-03", 18, 23)])
def test_trace_excerpts_are_captured_slices_of_the_same_record(captured_trace, name, start, stop):
    complete = captured_trace["fnd.supervised-learning.trace-01"]
    excerpt = captured_trace[f"fnd.supervised-learning.{name}"]
    assert complete == output("trace-01")
    assert excerpt.split() == "".join(complete.splitlines(keepends=True)[start:stop]).split()
    assert all(len(line) <= 79 for line in excerpt.splitlines())
    assert excerpt == output(name)
    assert len(excerpt.splitlines()) == {"trace-02": 17, "trace-03": 6}[name]
    assert "Later observed outcome" not in excerpt and "pseudo-label" not in excerpt


def test_trace_draft_line_fits_the_printed_measure() -> None:
    assert len(output("trace-03").splitlines()[2]) <= 89


def test_printed_bootstrap_distribution_matches_august_gate_values(run) -> None:
    printed, after = run
    stored = output("bootstrap-01")
    assert stored in printed
    lines = stored.splitlines()
    assert len(lines) == 2003
    draws = np.array([float(line.split()[1]) for line in lines[3:]])
    assert [int(line.split()[0]) for line in lines[3:]] == list(range(1, 2001))
    d = demo()
    costs = after["bootstrap_costs"]  # Already-consulted arrays; this test makes no new read.
    paired = np.array(costs["placeholder"]) - np.array(costs["tier"])
    indices = np.random.default_rng(7).integers(0, len(paired), size=(2000, len(paired)))
    expected = paired[indices].mean(axis=1)
    np.testing.assert_allclose(draws, expected, atol=5e-10, rtol=0)
    mean, low, high = d.bootstrap(paired)
    assert f"mean saving ${mean:.6f}; 95% percentiles ${low:.6f} to ${high:.6f}" in stored


def test_held_out_months_are_frozen_and_read_once_each_after_every_fit(
    run: tuple[str, dict],
) -> None:
    printed, after = run
    # Each held-out month is a frozen set with one logged consultation, and the
    # script's own windows no longer hold a held-out request.
    assert after["window"] == ["open", "training", "validation"]
    assert after["intact"] == {"august": True, "september": True}
    for name, requests, version in (
        ("august", 999, "2026-07-01"),
        ("september", 966, "2026-09-01"),
    ):
        manifest, log = after["manifests"][name], after["logs"][name]
        assert manifest["name"] == f"history-{name}" and manifest["label_policy"] == "outcome-1"
        assert manifest["requests"] == requests and manifest["by_tier"] == {"1": requests}
        assert manifest["by_policy_version"] == {version: requests}
        assert len(manifest["sha256"]) == 64 and manifest["sha256"][:16] in printed
        assert [(e["entry"], e["set"], e["reader"]) for e in log] == [
            (1, f"history-{name}", "stage 1 report")
        ]
    # August's manifest and its read come after both maps are fit, September's after that.
    before, mark, rest = printed.partition("name: history-august")
    assert mark and "map: slope, intercept" in before
    assert rest.index("August: ") < rest.index("September: ") < rest.index("history-september")
    # Flip every held-out outcome: nothing printed before August's manifest may change,
    # and both fingerprints must.
    scrambled, flipped = run_script(SCRAMBLE)
    assert scrambled.partition("name: history-august")[0] == before
    assert scrambled != printed
    for name in ("august", "september"):
        assert flipped["manifests"][name]["sha256"] != after["manifests"][name]["sha256"]


def test_demo_reaches_held_out_rows_only_through_a_logged_consultation() -> None:
    source = DEMO.read_text(encoding="utf-8")
    freeze_august = source.index('august = frozen(window.pop("test"), "history-august")')
    freeze_september = source.index(
        'september = frozen(window.pop("shifted"), "history-september")'
    )
    first_fit = source.index("# region book:fnd.supervised-learning.fit-01")
    first_read = source.index("consulted(august, ")
    last_fit = source.index("# endregion book:fnd.supervised-learning.calibrate-01")
    # Frozen straight after the split and before any fit, read only after the last one.
    assert freeze_august < freeze_september < first_fit < last_fit < first_read
    assert source.count("consulted(august, ") == source.count("consulted(september, ") == 1
    # From the freeze on, no line names a held-out window, a frozen row or a frozen
    # label; the only way to them is `consult`, which `consulted` alone calls.
    after_freeze = source[source.index("del history", freeze_september) + len("del history") :]
    for name in ('"test"', '"shifted"', ".requests", ".labels", ".correct"):
        assert name not in after_freeze, name
    assert not re.search(r"(?<![\w.\"])history(?![\w-])", after_freeze)  # the full list is gone
    assert source.count(".consult(") == 1 and "def consulted(golden, purpose):" in source
    # No printed listing reads a held-out month.
    for name in PRINTED:
        assert "consulted(" not in region(source, name) and "august" not in region(source, name)


def test_exercise_fixture_is_the_report_with_the_planted_error(run: tuple[str, dict]) -> None:
    _, after = run
    fixture = json.loads(FIXTURE.read_text())
    assert fixture == after["planted"]
    assert fixture["map_fit_on"] == fixture["judged_on"]  # the planted error
    assert fixture["calibration_error"] == 0.0
    assert sum(b["requests"] for b in fixture["bins"]) == fixture["judged_on"]["requests"]
