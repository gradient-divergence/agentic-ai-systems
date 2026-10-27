"""Check a submitted build against the chapter's acceptance conditions."""

import os
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from demos import causal_inference_demo as demo

PATH = Path(os.environ.get("CAUSAL_INFERENCE_SOLUTION", Path(__file__).with_name("starter.py")))
SOLUTION = runpy.run_path(str(PATH))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from unittest.mock import patch

    from refund_agent.split import GoldenSet
    from refund_agent.tools import Tools

    with (
        patch.object(
            GoldenSet, "consult", side_effect=AssertionError("rehearsal read a frozen window")
        ),
        patch.object(Tools, "_execute", side_effect=AssertionError("rehearsal paid")),
        patch.object(
            Tools, "_request_review", side_effect=AssertionError("rehearsal opened review")
        ),
    ):
        return SOLUTION["build"](tmp_path_factory.mktemp("exercise"), seed=7)


def test_plan_and_assignment_probabilities(result):
    assert result["plan"].cap == 15 and demo.check_plan(result["plan"]) == "accepted"
    records = result["records"]
    assert records and all(r["probability"] == (0.5 if r["in_set"] else 1.0) for r in records)
    selected = [r for r in records if r["in_set"]]
    low, high = demo.coin_range(len(selected))
    assert low <= sum(r["action"] == "refer" for r in selected) <= high
    changed = demo.rehearse(replace(result["plan"], seed=19), result["rows"], result["scores"])
    assert all(
        (a["action"], a["probability"]) == (b["action"], b["probability"])
        for a, b in zip(records, changed)
        if not a["in_set"]
    )


def test_size_losses_repeatability_and_no_new_read(result, tmp_path):
    size = result["size"]
    canonical = demo.planning_history(seed=7)
    assert [r.request_id for r in result["rows"]] == [r.request_id for r in canonical["validation"]]
    assert result["records"] == demo.rehearse(
        result["plan"], canonical["validation"], canonical["score"]
    )
    assert size == demo.sizing(
        result["plan"], canonical["validation"], canonical["score"], difference=2
    )
    assert size["set_requests"] > 0 and size["per_arm"] > 0 and size["months"] > 0
    assert size["expected_loss"] <= size["worst_case_loss"]
    again = SOLUTION["build"](tmp_path, seed=7)
    assert result["records"] == again["records"] and size == again["size"]
    assert result["frozen_logs_before"] == result["frozen_logs_after"] == []
    assert result["executions"] == result["reviews_opened"] == 0
