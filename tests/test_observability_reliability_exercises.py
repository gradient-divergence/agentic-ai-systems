"""All six chapter 34 starters fail their designed behavior; references pass."""

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "exercises/observability-reliability"


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(EXERCISES))
    for module in ("quotes_06", "detector_06"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    return runpy.run_path(str(EXERCISES / "reference.py"))


def starter(name):
    return runpy.run_path(str(EXERCISES / name))


def test_exercise_1_omitted_records_fail_and_reference_passes(reference):
    fixture = json.loads((EXERCISES / "trace-01.json").read_text())
    assert fixture["audit"] and fixture["spans"]
    roots = [s for s in fixture["spans"] if s["name"] == "run"]
    assert len(roots) == 2 and [s["attempt"] for s in roots] == [1, 2]
    required = {"audit", "spans", "lineage", "event_log", "correction"}
    with pytest.raises(AssertionError):
        assert starter("trace_01.py")["required_records"]() == required
    assert reference["required_records"]() == required


def test_exercise_2_policy_referral_and_budgets(reference):
    rows = json.loads((EXERCISES / "outcomes-02.json").read_text())
    assert all(row["outcomes_final"] is True for row in rows)
    expected = {
        "a": ("good", "not_counted", []),
        "b": ("good", "not_counted", []),
        "c": ("bad", "not_counted", []),
        "d": ("good", "bad", []),
        "e": ("good", "good", ["stale_payment"]),
        "f": ("good", "good", []),
        "g": ("bad", "good", []),
        "h": ("good", "not_counted", []),
    }
    result = reference["classify_outcomes"](rows)
    for key, (timeliness, corrections, breaches) in expected.items():
        assert result[key] == {
            "timeliness": timeliness,
            "corrections": corrections,
            "breaches": breaches,
        }
    with pytest.raises(AssertionError):
        assert starter("outcomes_02.py")["classify_outcomes"](rows) == result


def test_exercise_2_uncorrected_payments_wait_for_explicit_finality(reference):
    rows = json.loads((EXERCISES / "outcomes-02.json").read_text())
    pending = [{**row, "outcomes_final": False} for row in rows]
    result = reference["classify_outcomes"](pending)
    assert result["d"]["corrections"] == "bad"
    for key in ("e", "f", "g"):
        assert result[key]["corrections"] == "pending"
    del pending[-1]["outcomes_final"]
    with pytest.raises(KeyError, match="outcomes_final"):
        reference["classify_outcomes"](pending)


def test_exercise_3_forecast_window_units(reference):
    result = reference["bank_budget"]()
    assert result == {
        "allowance": 2160.0,
        "burn": 20.0,
        "hourly_page_minutes": 43.2,
        "fault_bad": 90.0,
    }
    with pytest.raises(AssertionError):
        assert starter("bank_03.py")["bank_budget"]() == result


def test_exercise_4_population_and_exact_tail(reference):
    result = reference["daily_rule"]()
    assert result["eligible"] == 40 and result["threshold"] == 3
    assert result["power"] == pytest.approx(0.7771918757)
    with pytest.raises(AssertionError):
        assert starter("count_04.py")["daily_rule"]() == result


def test_exercise_5_execution_gap_and_distractors(reference):
    record = json.loads((EXERCISES / "trace-05.json").read_text())
    result = reference["diagnose"](record)
    assert result["age_at_payment_seconds"] == 2250 and result["breach"]
    assert "trusted execution clock" in result["action"]
    with pytest.raises(AssertionError):
        assert starter("trace_05.py")["diagnose"](record)["breach"]


def test_exercise_6_closes_fail_and_hourly_reference_passes(reference):
    from datetime import timedelta

    from quotes_06 import START, market_open

    faulty = starter("detector_06.py")["run_detector"]()
    assert any(not market_open(p["at"]) for p in faulty)
    start = START + timedelta(days=2, hours=11)
    assert not any(start <= p["at"] <= start + timedelta(seconds=20) for p in faulty)
    fixed = reference["run_detector"]()
    assert len(fixed) == 1 and fixed[0]["venue"] == 1
    assert all(market_open(p["at"]) for p in fixed)
    assert start <= fixed[0]["at"] <= start + timedelta(seconds=20)
