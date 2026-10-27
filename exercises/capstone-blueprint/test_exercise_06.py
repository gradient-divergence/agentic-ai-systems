"""Select the reference only for reference verification, never in product code."""

import json
import os
import runpy
from pathlib import Path

import pytest

from refund_agent.launch.capstone_blueprint_packet import EvidenceClaims

ROOT = Path(__file__).resolve().parent
NAME = (
    "capstone_blueprint_reference.py"
    if os.environ.get("CAPSTONE_BLUEPRINT_USE_REFERENCE") == "1"
    else "capstone_blueprint_starters.py"
)
SOLUTIONS = runpy.run_path(str(ROOT / NAME))


def test_exercise_01():
    assert SOLUTIONS["exercise_01"]() == (
        "consistent",
        "reversible",
        "signed_in_order",
        "complete",
    )


def test_exercise_02():
    answer = SOLUTIONS["exercise_02"]()
    assert (
        answer["stage"],
        answer["launch_signed"],
        answer["paid_refunds_undone"],
    ) == (6, False, False)


def test_exercise_03():
    packet = json.loads((ROOT / "packet_03.json").read_text())
    assert SOLUTIONS["exercise_03"](packet) == {
        "verdict": "hold",
        "not_counted": ("rehearsal", "injection_suite"),
    }


def test_exercise_04():
    answer = SOLUTIONS["exercise_04"]()
    assert float(answer["manual_hours"]) == pytest.approx(61 + 2 / 3)
    assert float(answer["agent_hours"]) == pytest.approx(48 + 1 / 15)
    assert answer["verdict"] == "hold"


def comparison_claim():
    return EvidenceClaims(
        record_id="comparison",
        kind="comparison",
        day=64,
        bundle="B6",
        signed_by="morgan",
        signer_role="decision_owner",
        signed_day=49,
        run_started_day=50,
        verdict="not better",
        label_dependent=True,
        last_payment_day=54,
    )


def test_exercise_06_deployed_and_default():
    check = SOLUTIONS["check_maturity"]
    record = comparison_claim()
    assert check(record, 64, "deployed") == (
        "comparison: labels mature on day 84",
    )
    assert check(record, 64, "companion") == ()
    assert record.verdict == "not better"


def test_exercise_06_boundary_and_missing_day():
    check = SOLUTIONS["check_maturity"]
    record = comparison_claim()
    assert check(record, 84, "deployed") == ()
    absent = record.model_copy(update={"last_payment_day": None})
    assert check(absent, 84, "deployed") == (
        "comparison: last payment day missing",
    )
