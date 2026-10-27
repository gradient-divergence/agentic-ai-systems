"""Run once against the starter, then with FUTURE_DIRECTIONS_EXERCISE_REFERENCE=1."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("FUTURE_DIRECTIONS_EXERCISE_REFERENCE") == "1":
    from future_directions_reference_06 import check_month
else:
    from watch_06 import check_month


class Referred:
    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.request_id = f"future-directions-{ordinal}"
        self.paid = False
        self.prediction = 0.04
        self.paid_day = None
        self.label_final_day = None

    @property
    def found_false_30d(self):
        raise AssertionError("a referred outcome was read")


class ImmaturePaid:
    def __init__(self, row):
        self.__dict__.update({k: v for k, v in vars(row).items() if k != "found_false_30d"})

    @property
    def found_false_30d(self):
        raise AssertionError("an immature paid outcome was read")


def cohort(first, total, paid, wrong, final):
    result = []
    for i in range(total):
        ordinal = first + i
        result.append(
            SimpleNamespace(
                request_id=f"future-directions-{ordinal}",
                ordinal=ordinal,
                paid=True,
                paid_day=final - 30,
                label_final_day=final,
                prediction=0.04,
                found_false_30d=int(i < wrong),
            )
            if i < paid
            else Referred(ordinal)
        )
    return result


def test_initial_failure_persists_a_separate_confirmation():
    result = check_month(
        cohort(1, 1200, 200, 36, 60), {"status": "clear"}, period="month-01", day=60
    )
    assert result == {"status": "pending", "failed_month": "month-01", "first": 1201, "last": 3200}


@pytest.mark.parametrize("wrong,status", [(20, "clear"), (90, "reverse")])
def test_confirmation_uses_its_matured_paid_subset(wrong, status):
    state = {"status": "pending", "failed_month": "month-01", "first": 1201, "last": 3200}
    result = check_month(
        cohort(1201, 2000, 500, wrong, 91), state, period="confirmation-01", day=91
    )
    assert result["status"] == status


def test_immature_confirmation_remains_pending_without_reading_outcomes():
    state = {"status": "pending", "failed_month": "month-01", "first": 1201, "last": 3200}
    rows = [ImmaturePaid(r) if r.paid else r for r in cohort(1201, 2000, 500, 90, 91)]
    assert check_month(rows, state, period="confirmation-01", day=90) == state


def test_failed_month_cannot_be_reused_as_its_confirmation():
    state = {"status": "pending", "failed_month": "month-01", "first": 1201, "last": 3200}
    with pytest.raises(ValueError, match="cohort"):
        check_month(cohort(1, 2000, 500, 90, 91), state, period="month-01", day=91)
