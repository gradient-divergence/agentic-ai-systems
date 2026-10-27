"""Check a submitted build against the chapter's acceptance conditions."""

import os
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from demos import ml_data_evaluation_demo as demo
from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY
from refund_agent.split import CUTS, fingerprint, time_split

PATH = Path(os.environ.get("ML_DATA_EVALUATION_SOLUTION", Path(__file__).with_name("starter.py")))
SOLUTION = runpy.run_path(str(PATH))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return SOLUTION["build"](tmp_path_factory.mktemp("exercise"), seed=7)


def test_slice_dates_and_both_fingerprints(result):
    parent, child = result["parent"], result["slice"]
    data = generate(7)
    canonical = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    assert parent.requests == tuple(canonical["golden"])
    expected = tuple(
        r for r in parent.requests if r.at >= datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert child.requests == expected and len(expected) == 60
    assert min(r.at for r in child.requests) >= CUTS[0] + timedelta(days=LABEL_POLICY.window_days)
    for frozen in (parent, child):
        assert frozen.intact()
        assert fingerprint(frozen.requests, frozen.labels) == frozen.manifest["sha256"]
    assert child.manifest["parent_sha256"] == parent.manifest["sha256"]


def test_shared_journal_has_exactly_parent_and_slice_reads(result):
    parent, child = result["parent"], result["slice"]
    assert child.log is parent.log and len(child.log) == 2
    assert [r["set"] for r in child.log] == [parent.manifest["name"], child.manifest["name"]]


def test_every_tier_reports_errors_recall_precision_with_intervals(result):
    from demos.ml_data_evaluation_demo import wilson_interval

    rows = result["report"]
    child = result["slice"]
    data = generate(7)
    training = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))["training"]
    labels = {d.request_id: d for d in data.correct}
    flags = demo.fit_and_predict(
        demo.decision_time_inputs(data, training),
        [demo.payment_forbidden(labels[r.request_id]) for r in training],
        demo.decision_time_inputs(data, list(child.requests)),
        demo.INPUTS,
    )
    expected = demo.tier_counts(
        list(zip(child.requests, child.labels)),
        demo.baseline_decisions(data, list(child.requests)),
        dict(zip((r.request_id for r in child.requests), flags)),
    )
    actual = {r["tier"]: (r["requests"], r["errors"], r["tp"], r["fp"], r["fn"]) for r in rows}
    assert actual == {tier: (n, errors, tp, fp, fn) for tier, n, errors, tp, fp, fn in expected}
    assert {r["tier"] for r in rows} == {1, 2, 3, "All"}
    for row in rows:
        assert 0 <= row["errors"] <= row["requests"]
        for name, numerator, denominator in (
            ("error_rate", row["errors"], row["requests"]),
            ("recall", row["tp"], row["tp"] + row["fn"]),
            ("precision", row["tp"], row["tp"] + row["fp"]),
        ):
            rate, interval = row[name]
            if denominator:
                assert rate == pytest.approx(numerator / denominator)
                assert interval == pytest.approx(wilson_interval(numerator, denominator))
            else:
                assert rate is None and interval is None
