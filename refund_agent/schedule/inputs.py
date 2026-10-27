"""Constructed desk inputs, as given for the example, in UTC minutes."""

from datetime import date, datetime, time, timezone

INPUTS_VERSION = "review-inputs-v1"
REVIEW_MINUTES = {1: 5, 2: 15, 3: 30}
MINUTES = {"alex": 120, "morgan": 60}
PLAN_TIME = time(9, tzinfo=timezone.utc)
WORKING_DAYS = (0, 1, 2, 3, 4)
START = date(2026, 1, 22)
END = date(2026, 6, 30)
TOL = 1e-7
BUDGET = 10000  # branch nodes across the count and waiting-time solves


def morning(day):
    return datetime.combine(day, PLAN_TIME)
