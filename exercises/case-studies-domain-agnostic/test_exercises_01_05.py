import importlib.util
import math
import os
from fractions import Fraction as F
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / os.environ.get(
    "CASE_STUDIES_DOMAIN_AGNOSTIC_ANSWERS", "case_studies_domain_agnostic_starter_01_05.py"
)
SPEC = importlib.util.spec_from_file_location("case_studies_domain_agnostic_answers", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_01_form_vs_refund_answers():
    response = MODULE.responses()["01"]
    assert set(response["packet"]) == {"a", "c", "e", "h"}, (
        "the packet contains the general questions and controls"
    )
    assert set(response["refund"]) == {"b", "d", "f", "g"}


def test_02_transfer_changes_the_clinic_decision_unit():
    response = MODULE.responses()["02"]
    assert response["triage"] == ("clock", "number")
    assert response["hospital"] == ("coupling", "unit of decision"), "shared slots need allocation"
    assert response["bank"] == ("rule", "autonomous action")


def test_03_budget_respects_the_prior_tolerance():
    response = MODULE.responses()["03"]
    rate = math.log(2) / 1.5
    chance_at_limit = -math.expm1(-rate * response["budget_seconds"] / 3600)
    assert chance_at_limit == pytest.approx(0.02, abs=1e-12), (
        "copied half-hour limit exceeds the named 2%"
    )
    assert response["chance_30_minutes"] == pytest.approx(1 - 2 ** (-1 / 3))


def test_04_reprice_both_edges_and_the_slower_reversal():
    response = MODULE.responses()["04"]
    assert F(response["clear_edge"]) == F(5, 510), "use this bank's fraud loss and review cost"
    assert F(response["hold_edge"]) == F(15, 105)
    assert response["decisions"] == ["refer", "hold"]
    assert F(response["day_hold_edge"]) == F(360, 450)


def test_05_hold_until_the_insurers_labels_mature():
    response = MODULE.responses()["05"]
    assert response["window_days"] == 90, "the copied 30-day window misses late findings"
    assert response["hidden_fraud_share"] == 0.4
    assert response["decision"] == "hold"
