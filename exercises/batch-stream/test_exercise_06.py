"""Behavior checks shared by the reader's starter and reference solution."""

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = Path(os.environ.get("BATCH_STREAM_SUBMISSION", HERE / "starter_06.py"))
SPEC = importlib.util.spec_from_file_location("batch_stream_submission", PATH)
if SPEC is None or SPEC.loader is None:
    raise ValueError("a Python submission with a loader is required")
SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMISSION)
DAY = datetime(2026, 1, 20, tzinfo=timezone.utc)


def records():
    return json.loads((HERE / "scans-03.json").read_text())


def test_copy_counts_once():
    rows = records()[:4]
    assert SUBMISSION.count_day(rows, DAY) == {"count": 3, "corrections": []}


def test_late_scans_stay_in_event_day():
    rows = [r for r in records() if r["event_id"] in ("s7", "s8")]
    assert SUBMISSION.count_day(rows, DAY) == {"count": 2, "corrections": []}


def test_final_cutoff_and_replay():
    rows = records()
    expected = {"count": 8, "corrections": ["s9"]}
    assert SUBMISSION.count_day(rows, DAY) == expected
    assert SUBMISSION.count_day(list(reversed(rows)), DAY) == expected
