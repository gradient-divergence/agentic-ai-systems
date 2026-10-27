"""Reference solutions; never imported by the starters."""

from refund_agent.observability_reliability.budget import binom_tail, min_count


def required_records():
    return {"audit", "spans", "lineage", "event_log", "correction"}


def classify_outcomes(rows):
    from datetime import datetime, timedelta

    from refund_agent.observability_reliability.objectives import (
        Observation,
        classify,
    )

    result = {}
    for row in rows:
        final = row["outcomes_final"]
        if type(final) is not bool:
            raise ValueError("outcomes_final requires a boolean")
        received = datetime.fromisoformat(row["received_at"])
        at = received + timedelta(hours=row["answer_hours"])
        observation = Observation(
            request_id=row["id"],
            received_at=received,
            decided_at=at,
            outcome=row["outcome"],
            reason=row["reason"],
            promise_hours=row["promise_hours"],
        )
        classification = classify(observation, now=at, outcomes_final=final).model_dump()
        classification.pop("request_id")
        # This exercise supplies payment facts without inventing receipt evidence.
        classification["corrections"] = (
            "bad"
            if row.get("corrected")
            else "good"
            if row["outcome"] == "approve" and final
            else "pending"
            if row["outcome"] == "approve"
            else "not_counted"
        )
        classification["breaches"] = (
            ["stale_payment"] if row.get("shipment_age_seconds", 0) > 1800 else []
        )
        result[row["id"]] = classification
    return result


def bank_budget():
    allowance = 3000 * 720 * 0.001
    return {
        "allowance": allowance,
        "burn": 0.02 / 0.001,
        "hourly_page_minutes": allowance * 0.02,
        "fault_bad": 3000 * 0.02 * 1.5,
    }


def daily_rule():
    k = min_count(40, 0.01, 0.01)
    return {
        "eligible": 40,
        "threshold": k,
        "power": float(binom_tail(k, 40, 0.1)),
    }


def diagnose(record):
    from datetime import datetime

    spans = [s for s in record["spans"] if s["trace_id"] == record["trace_id"]]
    read = next(s for s in spans if s["name"] == "tool:shipment.status")
    payment = next(s for s in spans if s["name"] == "tool:refund.execute")
    age = (
        record["audit_age_seconds"]
        + (
            datetime.fromisoformat(payment["start"]) - datetime.fromisoformat(read["end"])
        ).total_seconds()
    )
    return {
        "age_at_payment_seconds": age,
        "breach": age > 1800,
        "action": "end authority; trusted execution clock and deadline shorter than remaining evidence life",
    }


def rate_at(at):
    from quotes_06 import market_open

    return 3600 if market_open(at) else 0


def quiet_hours(rate_per_hour, open_hours=325, expected_false=1):
    import math

    if not all(math.isfinite(x) and x>0 for x in (rate_per_hour,open_hours,expected_false)):
        raise ValueError("positive finite rates, hours and tolerance required")
    if expected_false >= rate_per_hour*open_hours:
        raise ValueError("false-page tolerance must be below expected arrivals")
    return math.log(rate_per_hour*open_hours/expected_false)/rate_per_hour


def run_detector():
    from detector_06 import run_detector as run

    return run(rate_at,quiet_hours)
