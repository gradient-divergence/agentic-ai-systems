"""Reference solution checks, exercises 1-5; arithmetic and event-time boundaries."""

import json
from datetime import datetime, timedelta
from fractions import Fraction

from refund_agent.batch_stream.authority import AUTHORITY
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.windows import day_counts


def solve(exercise, inputs):
    if exercise == 1:
        delay = (
            datetime.fromisoformat(inputs["arrived_at"])
            - datetime.fromisoformat(inputs["at"])
        ).total_seconds()
        age = (
            datetime.fromisoformat(inputs["request_at"])
            - datetime.fromisoformat(inputs["observed_at"])
        ).total_seconds()
        return {"delay_seconds": delay, "age_seconds": age}
    if exercise == 2:
        return {name: AUTHORITY[name][0] for name in inputs}
    if exercise == 3:
        day = datetime.fromisoformat(inputs["day"])
        log = EventLog(
            ScanEvent.model_validate_json(json.dumps(r))
            for r in inputs["records"]
        )
        row = day_counts(
            log,
            day,
            day + timedelta(days=1),
            timedelta(hours=6),
            watermark_at=lambda t: t - timedelta(hours=2),
        )
        return {"fast": row["fast"], "distinct": row["distinct"]}
    if exercise == 4:
        base = Fraction(str(inputs["restored_share"]))
        shares = {}
        for name, delay in (
            ("fast_share", "fast_delay"),
            ("wait_share", "wait_delay"),
        ):
            delayed = base * Fraction(str(inputs[delay]))
            shares[name] = float(delayed / (delayed + 1 - base))
        return shares
    if exercise == 5:
        request = datetime.fromisoformat(inputs["request_at"])
        delivery = next(
            r for r in inputs["events"] if r["status"] == "delivered"
        )
        window = (
            request - datetime.fromisoformat(delivery["at"])
        ).total_seconds() / 3600
        return {
            "window_hours": window,
            "outside": window > inputs["policy_return_days"] * 24,
        }
    raise ValueError("exercise must be 1 through 5")
