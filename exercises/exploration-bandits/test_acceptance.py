"""Exercise 6 acceptance; point EXPLORATION_BANDITS_SOLUTION to a submission.

The chapter demo supplies the reference solution, without changing test_offers.py.
"""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from refund_agent.offers.evaluation import OfferPackage, SealedRecords
from refund_agent.offers.policies import Capped, EpsilonGreedy, EstimateTable
from refund_agent.offers.replay import training_rows
from refund_agent.split import GoldenSet


@pytest.fixture(scope="module")
def solution():
    def forbidden(*args, **kwargs):
        raise AssertionError("an amount-context run consulted a frozen set")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GoldenSet, "consult", forbidden)
        patch.setattr(OfferPackage, "load", forbidden)
        patch.setattr(SealedRecords, "__init__", forbidden)
        patch.setattr(SealedRecords, "load", forbidden)
        patch.setattr(SealedRecords, "consult", forbidden)
        path = Path(os.environ.get("EXPLORATION_BANDITS_SOLUTION", Path(__file__).with_name("starter.py")))
        spec = importlib.util.spec_from_file_location("exploration_exercise_solution", path)
        assert spec is not None and spec.loader is not None, f"cannot load solution {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(scope="module")
def results(solution):
    first = solution.exercise_six_runs(seed=7)
    second = solution.exercise_six_runs(seed=7)
    for cap in (0.05, 0.20):
        assert first[cap]["run"].log.fingerprint == second[cap]["run"].log.fingerprint
        assert first[cap]["report"] == second[cap]["report"]
        assert json.dumps([r.record() for r in first[cap]["run"].log.events], sort_keys=True) == \
            json.dumps([r.record() for r in second[cap]["run"].log.events], sort_keys=True)
    return first


def test_amount_context_is_decision_time_reason_and_amount(solution):
    row = training_rows()[0]
    assert solution.amount_context(replace(row, amount_cents=2499)) == f"{row.reason}|under25"
    assert solution.amount_context(replace(row, amount_cents=2500)) == f"{row.reason}|25plus"


@pytest.mark.parametrize("cap", [0.05, 0.20])
def test_both_logs_preserve_probabilities_maturity_and_six_contexts(cap, results):
    result = results[cap]["run"]
    assert len(result.tables) == 6
    assert len(result.log.rows) == 9837
    policy = Capped(EpsilonGreedy(0.3), cap)
    for t, row in enumerate(result.log.rows, 1):
        assert row.table_key == f"{row.reason}|{'under25' if row.amount_cents < 2500 else '25plus'}"
        table = EstimateTable(row.table_answers, row.table_no_contacts)
        p = policy.probabilities(table, t, row.eligible_offers)
        assert row.table_fingerprint == table.fingerprint
        np.testing.assert_array_equal(row.probabilities, p)
        assert row.propensity == p[("standard", "dated", "referenced").index(row.offer)]
        assert sum(row.probabilities) == pytest.approx(1)
        assert row.observed_at >= row.matures_at
    # Reconstruct the tables from append-only events to check every update, not
    # merely each final timestamp. Early answers cannot influence later snapshots.
    tables = {key: EstimateTable() for key in result.tables}
    for event in result.log.events:
        assert event.table_key == f"{event.reason}|{'under25' if event.amount_cents < 2500 else '25plus'}"
        table = tables[event.table_key]
        if event.reward is None:
            assert event.table_fingerprint == table.fingerprint
            np.testing.assert_array_equal(event.table_answers, table.answers)
            np.testing.assert_array_equal(event.table_no_contacts, table.no_contacts)
        else:
            assert event.observed_at >= event.matures_at
            index = ("standard", "dated", "referenced").index(event.offer)
            table.answers[index] += 1
            table.no_contacts[index] += event.reward
    for key, table in tables.items():
        np.testing.assert_array_equal(result.tables[key].answers, table.answers)
        np.testing.assert_array_equal(result.tables[key].no_contacts, table.no_contacts)
        assert result.tables[key].fingerprint == table.fingerprint
