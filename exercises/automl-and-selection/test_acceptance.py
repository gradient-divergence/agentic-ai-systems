"""Check a submitted build against the chapter's acceptance conditions."""

import os
import runpy
from pathlib import Path

import pytest

from refund_agent.selection import decide

PATH = Path(os.environ.get("AUTOML_SELECTION_SOLUTION", Path(__file__).with_name("starter.py")))
SOLUTION = runpy.run_path(str(PATH))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return SOLUTION["build"](tmp_path_factory.mktemp("exercise"), seed=7)


def test_commission_precedes_all_forty_trials(result):
    built, record = result["built"], result["record"]
    assert built.commission.budget == 40 and built.commission.intact()
    assert built.events[0]["event"] == "commission"
    assert len(built.trials) == 40
    assert all(len(check.results) == 4 for check in built.checks.values())
    assert len(record.log) == 1 and record.frozen["requests"] == 985


def test_decision_follows_intervals_and_defense_claims_no_authority(result):
    record = result["record"]
    for row in record.evidence[1:]:
        assert record.decisions[row["name"]] == decide(
            result["built"].checks[row["name"]].passed,
            row["difference"][0],
            tuple(row["difference"][1:]),
        )
    assert result["defense"].strip() and result["authority"] == "selection only"


def test_seed_seven_reproduces_trials_and_losses(result, tmp_path):
    again = SOLUTION["build"](tmp_path, seed=7)
    assert result["built"].trials == again["built"].trials
    assert result["record"].evidence == again["record"].evidence
