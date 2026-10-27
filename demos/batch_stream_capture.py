"""Capture chapter 31's outputs and records from one fresh experiment."""

import json
import os
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from demos import regions as _regions
from refund_agent.release import suite as release_suite

# isort: split
# Initialize the core before importing the selected component's helpers.
from refund_agent.batch_stream.experiment import (
    claim_run,
    count_run,
    exercise_scans,
    figure_run,
    json_default,
    opening_run,
    payment_exercise,
    replay_run,
    stream,
)
from refund_agent.batch_stream.generate import SETTINGS, START
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.view import shipment_as_of
from refund_agent.batch_stream.windows import day_counts

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT
EXERCISES = COMPANION / "exercises/batch-stream"


@lru_cache(maxsize=1)
def candidate_run():
    from refund_agent.batch_stream.component import run_release
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import (
        _CaseRun,
        _run_loop_case,
        run_case,
    )

    suite = extend(release_suite.build_suite())
    candidate = build_candidate(suite)
    record = run_release(candidate, suite)
    positives = [case for case in suite.cases if case.part == "positive"]
    rows = [run_case(case, candidate) for case in positives]
    denied = build_candidate(suite, name="deny-all")
    controls = [run_case(case, denied) for case in positives]
    public_case = next(
        case for case in suite.cases if case.case_id == "public-r-7-0008"
    )
    terminal = run_case(public_case, candidate)

    # Timed-attempt observations stay beside the core's terminal-request grade.
    def attempts(case):
        execution = _CaseRun(case, candidate, "execute")
        _run_loop_case(case, candidate, "execute", execution)
        return execution._artifacts().tools.batch_stream_attempts

    return {
        "release": record,
        "positive_payments": sum(row["payments"] for row in rows),
        "positive_cents": sum(
            row["decision"]["amount_cents"]
            for row in rows
            if row["decision"]["executed"]
        ),
        "positive_100_attempts": attempts(
            next(case for case in positives if case.case_id == "positive-100")
        ),
        "deny_all_failures": sum(bool(row["failed"]) for row in controls),
        "r-7-0008": {
            "attempts": attempts(public_case),
            "terminal_requests": 1,
            "failed": terminal["failed"],
        },
    }


def key_run():
    log = EventLog(exercise_scans())
    times = [START + timedelta(hours=h) for h in (24, 26, 29, 32, 33)]
    rows = [
        dict(
            at=t,
            **day_counts(
                log,
                START,
                t,
                timedelta(hours=6),
                watermark_at=lambda t: t - timedelta(hours=2),
            ),
        )
        for t in times
    ]
    late = sorted(
        {
            e.event_id
            for e in log.entries
            if e.at < e.arrived_at - timedelta(hours=2)
        }
    )
    fractions = [
        Fraction(7, 10) * p / (Fraction(7, 10) * p + Fraction(3, 10))
        for p in (Fraction(4, 100), Fraction(5, 1000))
    ]
    payment = payment_exercise()
    request_at = datetime.fromisoformat(payment["request_at"])
    delivery = next(
        ScanEvent.model_validate_json(json.dumps(e))
        for e in payment["events"]
        if e["order_id"] == payment["order_id"] and e["status"] == "delivered"
    )
    windows = [
        (request_at - time).total_seconds() / 3600
        for time in (delivery.at, delivery.arrived_at)
    ]
    opening_case = opening_run()
    delay = (
        opening_case["arrival"].arrived_at - opening_case["arrival"].at
    ).total_seconds()
    observed = shipment_as_of(
        opening_case["log"],
        opening_case["request"].order_reference,
        opening_case["request"].at,
    ).observed_at
    age = (opening_case["request"].at - observed).total_seconds()
    example = Fraction(4, 5) * Fraction(1, 20)
    example /= example + Fraction(1, 5)
    checked = candidate_run()
    return {
        "delay_seconds": delay,
        "age_seconds": age,
        "example_absence_fraction": str(example),
        "counts": rows,
        "late_ids": late,
        "telecom_fractions": [str(f) for f in fractions],
        "telecom_shares": [float(f) for f in fractions],
        "windows_hours": windows,
        "return_limit_hours": payment["policy_return_days"] * 24,
        "payment_watermark": datetime.fromisoformat(payment["watermark"]),
        "payment_watermark_required": False,
        "candidate": {
            "verdict": checked["release"]["decision"],
            "failed_cases": checked["release"]["failed_cases"],
            "positive_payments": checked["positive_payments"],
            "positive_cents": checked["positive_cents"],
            "positive_100_attempts": checked["positive_100_attempts"],
            "deny_all_failures": checked["deny_all_failures"],
            "r-7-0008": checked["r-7-0008"],
        },
        "correction": {
            "payment_id": payment["receipt_id"],
            "mode": "executed",
            "request_id": payment["request_id"],
            "order_id": payment["order_id"],
            "event_id": delivery.event_id,
            "at": delivery.at,
            "arrived_at": delivery.arrived_at,
            "window_event_hours": windows[0],
            "window_arrival_hours": windows[1],
            "policy_limit_hours": payment["policy_return_days"] * 24,
            "reason": "outside_return_window",
            "payment_stands": True,
        },
    }


def write_data(*, destination=None):
    records = {
        "settings": SETTINGS,
        "stream_fingerprint": stream().fingerprint,
        "events": [e.model_dump(mode="json") for e in stream().log.entries],
        "requests": [r.model_dump(mode="json") for r in stream().requests],
        "counts": count_run(),
        "claims": claim_run(),
        "replay": replay_run(),
        "keys": key_run(),
        "figure": figure_run(),
        "candidate": candidate_run(),
    }
    # Capture replaces generated snapshots outside the component contract.
    root = COMPANION if destination is None else Path(destination)
    exercises = (
        EXERCISES if destination is None else root / "exercises/batch-stream"
    )
    captures = (
        (root / "demos/data/batch-stream-records.json", records, True),
        (
            exercises / "scans-03.json",
            [e.model_dump(mode="json") for e in exercise_scans()],
            False,
        ),
        (exercises / "payment-05.json", payment_exercise(), False),
    )
    for path, value, sorted_keys in captures:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                value, default=json_default, sort_keys=sorted_keys, indent=2
            )
            + "\n"
        )


def capture_chapter(destination):
    destination = Path(destination).resolve()
    extractor = vars(_regions)
    snippets = {}
    for relative in (
        "demos/batch_stream_demo.py",
        "refund_agent/batch_stream/replay.py",
    ):
        path = ROOT / relative
        snippets.update(
            {
                s.snippet_id: s
                for s in extractor["extract_snippets_from_text"](
                    path.read_text(), path
                )
                if s.output
            }
        )
    prior = os.environ.get("BATCH_STREAM_CAPTURE_DIR")
    os.environ["BATCH_STREAM_CAPTURE_DIR"] = str(destination)
    try:
        outputs = extractor["capture_outputs"](
            snippets,
            sys.executable,
            timeout=int(os.environ.get("BOOK_OUTPUT_TIMEOUT_SECONDS", "1800")),
        )
    finally:
        if prior is None:
            os.environ.pop("BATCH_STREAM_CAPTURE_DIR", None)
        else:
            os.environ["BATCH_STREAM_CAPTURE_DIR"] = prior
    out = destination / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out / f"_{name}.txt").write_text(text)
    return outputs
