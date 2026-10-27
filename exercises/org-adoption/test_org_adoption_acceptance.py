"""One designed acceptance failure for each starter, exercises 38.1-5."""

import org_adoption_starter as solution

from demos.org_adoption_fixtures import accounting_fixture
from refund_agent.rollout.org_adoption_desk import EXERCISE_BALANCE


def test_exercise_01_recorded_disagreement_is_governed():
    assert solution.CLASSIFICATION == (
        "governed",
        "workaround",
        "governed",
        "governed",
        "workaround",
    )


def test_exercise_02_named_people_own_actions():
    assert solution.ROLES == {
        "sign": "decision_owner",
        "stop": "release_owner",
        "cause": "reconciler",
    }


def test_exercise_03_intake_is_the_denominator():
    count = solution.count(accounting_fixture(20, console=True))
    assert (
        count.received,
        count.governed,
        len(count.around),
        len(count.missing),
    ) == (20, 18, 1, 1)
    assert count.governed_percent == 90


def test_exercise_04_delayed_posting_keeps_the_gate_blind():
    delayed = solution.amounts(EXERCISE_BALANCE, False)
    assert (
        delayed["balance_read_cents"],
        delayed["agent_paid_cents"],
        delayed["total_refunded_cents"],
        delayed["excess_cents"],
    ) == (9000, 4500, 10500, 1500)
    early = solution.amounts(EXERCISE_BALANCE, True)
    assert (
        early["balance_read_cents"],
        early["agent_paid_cents"],
        early["imported_receipts"],
    ) == (3000, 0, 1)
    assert len(early["count"].around) == len(delayed["count"].around) == 1


def test_exercise_05_phone_request_does_not_disappear():
    assert solution.diagnosis(accounting_fixture(80, console=False)) == {
        "denominator": "intake",
        "share": 98.75,
        "decision": "hold",
    }
