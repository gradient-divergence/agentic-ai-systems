"""One observable acceptance failure per starter response, exercises 1–5."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from refund_agent.sources.data_systems_budget import rate_upper, stale_chance

HERE = Path(__file__).resolve().parent
name = os.environ.get("DATA_SYSTEMS_SOLUTION", "data_systems_starter.py")
spec = importlib.util.spec_from_file_location("data_systems_submission", HERE / name)
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def test_exercise_1_contract_failure_is_referral():
    assert submission.exercise_1() == {
        "response": "refer",
        "reason": "contract_vocabulary:shipment.status",
    }


def test_exercise_2_silent_semantics_need_owner():
    assert submission.exercise_2() == {
        "caught_by": "none",
        "responsible": "the carrier integration team",
    }


def test_exercise_3_budget_uses_upper_bound():
    row = submission.exercise_3()
    assert row["estimate"] == 0.015
    assert row["upper"] == pytest.approx(rate_upper(15, 1000))
    assert row["chance"] == pytest.approx(0.003842, abs=1e-6)
    assert row["minutes"] == pytest.approx(13.02, abs=0.01)


def test_exercise_4_zero_is_not_zero_rate():
    row = submission.exercise_4()
    assert row["quiet_upper"] == pytest.approx(rate_upper(0, 6000))
    assert row["quiet_chance"] == pytest.approx(stale_chance(rate_upper(0, 6000), 0.5))
    assert row["daytime_rate"] == 0.04
    assert row["daytime_upper"] == pytest.approx(rate_upper(48, 1200))


def test_exercise_5_later_record_cannot_rebuild_original_read():
    fixture = json.loads((HERE / "lineage-05.json").read_text())
    row = submission.exercise_5(fixture)
    assert row["condition"] == "lineage"
    assert row["evidence"] == "shipment observed after decision"
    assert row["rebuild"] is False
    assert "original fingerprint" in row["reply"]
