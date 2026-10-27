"""Exercise 6 JSON acceptance, outside the default tests.

Set STRATEGY_REQUIREMENT to the submission; the default is the reference record.
This checks the exercise's arithmetic and record shape, not canary authorization.
"""

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from demos.data_science_strategy_demo import EvidenceRequirement, check, zero_failure_bound

PATH = Path(os.environ.get("STRATEGY_REQUIREMENT", Path(__file__).with_name("requirement-06.json")))


@pytest.fixture(scope="module")
def requirement():
    return EvidenceRequirement(**json.loads(PATH.read_text()))


def test_fixed_run_has_598_cases_and_their_bound(requirement):
    cases = requirement.cases.lower()
    assert re.search(r"\b598\b", cases)
    assert "fixed" in cases or "prespecified" in cases
    bounds = [float(value) for value in re.findall(r"0\.\d+", cases)]
    assert any(value == 0.0050 or abs(value - 0.00499705869425) < 1e-7 for value in bounds)
    assert zero_failure_bound(598) == pytest.approx(0.00499705869425, abs=1e-12)
    assert zero_failure_bound(597) > 0.005


@pytest.mark.parametrize("bound", ["0.004997", "0.00499705869425", "0.0050"])
def test_fixed_run_accepts_documented_bound_precision(requirement, bound):
    submission = replace(requirement, cases=f"A fixed run of 598 requests; bound {bound}.")
    test_fixed_run_has_598_cases_and_their_bound(submission)


@pytest.mark.parametrize("bound", ["0.0049", "0.0051", "0.0050001"])
def test_fixed_run_rejects_wrong_bounds(requirement, bound):
    submission = replace(requirement, cases=f"A fixed run of 598 requests; bound {bound}.")
    with pytest.raises(AssertionError):
        test_fixed_run_has_598_cases_and_their_bound(submission)


def test_signature_and_threshold_pass_the_pre_run_check(requirement):
    assert check(requirement, 10) == []
    assert requirement.signer.strip()
    assert type(requirement.signed_day) is int and 0 <= requirement.signed_day < 10
    assert "no higher" in requirement.threshold.lower()
    assert any("wrongful denials" in measure.lower() for measure in requirement.measures)
    assert "no more than the baseline's" in requirement.threshold.lower()


@pytest.mark.parametrize("missing", ["measure", "threshold"])
def test_wrongful_denial_requirement_rejects_missing_measure_or_threshold(requirement, missing):
    if missing == "measure":
        submission = replace(
            requirement,
            measures=[
                measure
                for measure in requirement.measures
                if "wrongful denials" not in measure.lower()
            ],
        )
    else:
        submission = replace(
            requirement,
            threshold="No forbidden payment; a share sent to a person no higher than the baseline's",
        )
    with pytest.raises(AssertionError):
        test_signature_and_threshold_pass_the_pre_run_check(submission)


def test_no_field_claims_to_undo_a_paid_refund(requirement):
    for value in vars(requirement).values():
        if isinstance(value, str):
            assert not re.search(
                r"\bundo\b|\bclaw back\b|\bpaid back\b|refunds? reversed", value.lower()
            )
