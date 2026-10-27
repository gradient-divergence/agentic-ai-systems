import os
from datetime import datetime, timezone

import pytest

from demos.human_oversight_risk_demo import exercise_log_rows

if os.environ.get("HUMAN_OVERSIGHT_RISK_REFERENCE") == "1":
    from reference import human_oversight_risk_answers_01_05 as answers
else:
    import human_oversight_risk_answers_01_05 as answers


def test_01_accumulation_has_a_review_queue_sensor():
    assert answers.exercise_1() == {"accumulates": "review_work", "sensor": "backlog_and_item_timestamps"}


def test_02_weekend_due_date_never_authorizes_a_clock_decision():
    arrival = datetime(2026, 10, 2, 18, 20, tzinfo=timezone.utc)
    result = answers.exercise_2(arrival)
    assert result["due"] == datetime(2026, 10, 5, 17, tzinfo=timezone.utc)
    assert result["timeout"] == "escalate_and_notify"


def test_03_the_ward_gets_a_tail_not_just_service_minutes():
    result = answers.exercise_3(10, 12, .5)
    assert result["L"] == 5 and float(result["W"]) * 60 == 30
    assert result["tail"] == pytest.approx(.36787944117)


def test_04_a_good_week_does_not_make_the_senior_monday_feasible():
    result = answers.exercise_4()
    assert result["ordinary_hours"] == 70 and result["senior_hours"] == 28
    assert result["adjusters"] == 3 and not result["senior_monday_feasible"]


def test_05_escalations_are_not_catches_and_busy_pending_work_is_visible():
    records, rows = exercise_log_rows()
    assert records[-1]["report"] == "Planted cases caught: 15 of 20 (75%). Review healthy."
    result = answers.exercise_5(rows)
    assert (result["caught"], result["decided"]) == (9, 14)
    assert result["bands"]["overloaded"]["caught"] == 0
    assert result["bands"]["overloaded"]["unresolved"] == 6
