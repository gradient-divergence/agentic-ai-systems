"""Acceptance checks for sign, versions, invariants and freshness at use."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from refund_agent.sources.data_systems_budget import budget_hours, rate_upper

HERE = Path(__file__).resolve().parent
name = os.environ.get("DATA_SYSTEMS_CONTRACT", "contract_06.py")
spec = importlib.util.spec_from_file_location("data_systems_submission_06", HERE / name)
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
FEED = {r["id"]: r for r in json.loads((HERE / "feed-06.json").read_text())["records"]}


def test_old_negative_purchase_refused():
    assert submission.problem(FEED["old-negative-purchase"]) == "purchase_sign"


def test_new_sign_rule_and_old_refund():
    assert submission.problem(FEED["new-refund"]) is None
    assert submission.problem(FEED["old-refund"]) is None


def test_unknown_type_refused():
    assert submission.problem(FEED["unknown-type"]) == "transaction_vocabulary"


def test_unknown_version_refused():
    assert submission.problem(FEED["unknown-version"]) == "version_unknown"


def test_recorded_basis_and_budget():
    assert submission.OWNER == "the card feed team"
    assert submission.VERSION == "card@2" and submission.UNITS == "USD cents"
    assert "30 refunds" in submission.BASIS
    assert submission.BUDGET_SECONDS == pytest.approx(
        3600 * budget_hours(rate_upper(30, 600), 0.01)
    )


@pytest.mark.parametrize("offset,expected", [(0, None), (0.001, "budget")])
def test_freshness_at_and_beyond_budget(offset, expected):
    record = dict(FEED["new-refund"],
                  age_seconds=submission.BUDGET_SECONDS + offset)
    assert submission.problem(record) == expected


def test_future_observation_refused():
    record = dict(FEED["new-refund"], age_seconds=-0.001)
    assert submission.problem(record) == "budget"


@pytest.mark.parametrize(
    "at,expected",
    [("2026-09-30T23:59:59+00:00", "version_not_effective"),
     ("2026-10-01T00:00:00+00:00", None)],
)
def test_announced_version_at_effective_boundary(at, expected):
    assert submission.problem(dict(FEED["new-refund"], at=at)) == expected
