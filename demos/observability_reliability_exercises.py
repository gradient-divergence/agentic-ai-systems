"""Chapter 34's fixture writer; invoked explicitly, never at import."""

import json
from datetime import timedelta


def write_exercises(experiment, destination):
    built, _, _ = experiment
    trace_id = "trace-" + built["stream"].silence.request_id
    data = {
        "scope": "constructed cohort rehearsal; simulated ledger",
        "spans": [s.model_dump(mode="json") for s in built["rehearsal"]["store"].trace(trace_id)],
        "audit": [
            a.model_dump(mode="json")
            for a in built["rehearsal"]["audits"]
            if a.trace_id == trace_id
        ],
    }
    write_exercise_records(data, destination)


def write_exercise_records(data, destination):
    """Write the same fixtures from the capture function's stored figure data."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trace-01.json").write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    from refund_agent.observability_reliability.fortnight import START

    outcomes = []
    for name, outcome, reason, answer, promise, extras in (
        ("a", "deny", "already_delivered", 1 / 6, 1, {}),
        ("b", "refer", "delivery_unknown", 1 / 6, 1, {}),
        ("c", "refer", "policy_not_in_bundle", 1 / 6, 1, {}),
        ("d", "approve", "eligible", 1 / 6, 1, {"corrected": True}),
        ("e", "approve", "eligible", 1 / 6, 1, {"shipment_age_seconds": 1860}),
        ("f", "approve", "eligible", 40, 48, {}),
        ("g", "approve", "eligible", 52, 48, {}),
        (
            "h",
            "deny",
            "already_delivered",
            1 / 6,
            1,
            {"wrongful_denial": True},
        ),
    ):
        outcomes.append(
            {
                "id": name,
                "outcome": outcome,
                "reason": reason,
                "received_at": START.isoformat(),
                "answer_hours": answer,
                "promise_hours": promise,
                "outcomes_final": True,
                **extras,
            }
        )
    (destination / "outcomes-02.json").write_text(
        json.dumps(outcomes, sort_keys=True, indent=2) + "\n"
    )
    diagnostic = {
        "trace_id": "observability-reliability-diagnostic",
        "audit_age_seconds": 150,
        "spans": [
            {
                "trace_id": "observability-reliability-diagnostic",
                "name": "tool:shipment.status",
                "start": START.isoformat(),
                "end": START.isoformat(),
            },
            {
                "trace_id": "observability-reliability-diagnostic",
                "name": "tool:refund.execute",
                "start": (START + timedelta(minutes=35)).isoformat(),
            },
            {
                "trace_id": "other-request",
                "name": "tool:refund.execute",
                "start": START.isoformat(),
            },
            {
                "trace_id": "policy-referral",
                "name": "run",
                "start": START.isoformat(),
                "reason": "delivery_unknown",
            },
        ],
    }
    (destination / "trace-05.json").write_text(
        json.dumps(diagnostic, sort_keys=True, indent=2) + "\n"
    )


def print_prebatch():
    """Infer the batch from simultaneous arrivals; use only event records."""
    from collections import Counter

    from refund_agent.batch_stream.view import late_events
    from refund_agent.observability_reliability.fortnight import (
        DAYS,
        START,
        make_stream,
    )
    from refund_agent.observability_reliability.silence import region_of

    log = make_stream().log
    unique = {e.event_id: e for e in log.entries}
    groups = Counter((region_of(e.order_id), e.arrived_at) for e in unique.values())
    _, batch = max(groups, key=groups.__getitem__)
    late = {e.event_id for e in late_events(log)}
    before = [e for e in unique.values() if START <= e.at < batch and e.arrived_at < batch]
    whole = [e for e in unique.values() if START <= e.at < START + timedelta(days=DAYS)]
    for name, events in (
        ("pre-batch arrived scans", before),
        ("fortnight", whole),
    ):
        bad = sum(e.event_id in late for e in events)
        print(f"{name}: {bad}/{len(events)} late; {bad / len(events):.6%}")


# region book:fnd.observability-reliability.prebatch-01 output
if __name__ == "__main__":
    print_prebatch()
# endregion book:fnd.observability-reliability.prebatch-01
