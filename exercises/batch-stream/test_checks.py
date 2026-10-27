"""Run the starter or the reference solution against exercises 1 through 5."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PATH = Path(
    os.environ.get("BATCH_STREAM_CHECK_SUBMISSION", HERE / "starter_checks.py")
)
SPEC = importlib.util.spec_from_file_location("batch_stream_checks", PATH)
if SPEC is None or SPEC.loader is None:
    raise ValueError("a Python submission with a loader is required")
SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMISSION)


@pytest.mark.parametrize("exercise", range(1, 6))
def test_exercise(exercise):
    if exercise == 1:
        inputs = {
            "at": "2026-01-20T11:40:00+00:00",
            "arrived_at": "2026-01-20T20:30:00+00:00",
            "request_at": "2026-01-20T18:05:00+00:00",
            "observed_at": "2026-01-20T18:03:00+00:00",
        }
        expected = {"delay_seconds": 31800, "age_seconds": 120}
    elif exercise == 2:
        inputs = [
            "pay_absence",
            "return_window",
            "latest_status",
            "pause_canary",
            "clean_and_widen",
        ]
        expected = {
            "pay_absence": "settled",
            "return_window": "delivery event time",
            "latest_status": "fast",
            "pause_canary": "fast; executed correction",
            "clean_and_widen": "settled",
        }
    elif exercise == 3:
        inputs = {
            "day": "2026-01-20T00:00:00+00:00",
            "records": json.loads((HERE / "scans-03.json").read_text()),
        }
        expected = {"fast": 7, "distinct": 6}
    elif exercise == 4:
        inputs = {
            "restored_share": 0.7,
            "fast_delay": 0.04,
            "wait_delay": 0.005,
        }
        expected = {"fast_share": 7 / 82, "wait_share": 7 / 607}
    else:
        inputs = json.loads((HERE / "payment-05.json").read_text())
        expected = {"window_hours": 745, "outside": True}
    result = SUBMISSION.solve(exercise, inputs)
    assert (
        result == pytest.approx(expected)
        if exercise == 4
        else result == expected
    )
