"""Constructed runs and arithmetic, separated from stdout capture."""

import json
from dataclasses import asdict
from datetime import timedelta
from functools import lru_cache
from math import ceil, sqrt
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.batch_stream.generate import (
    DAYS,
    START,
    dataset,
    generate,
    opening,
    read_sources,
    scan,
)
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.replay import (
    PaymentRecord,
    Views,
    apply,
    checkpoint,
    corrections,
    replay,
    restore,
)
from refund_agent.batch_stream.tools import EventTimeTools, LogShipmentTools
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    PROCESSING_VERSION,
    complete_through,
    first_complete,
    late_events,
    watermark,
)
from refund_agent.batch_stream.windows import day_counts
from refund_agent.rules import assess, decision, sources

ROOT = Path(__file__).resolve().parents[2]
EXERCISES = ROOT / "exercises/batch-stream"


def wilson(k, n):
    if not 0 <= k <= n or n == 0:
        raise ValueError("nonempty binomial population required")
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (
        0.0 if k == 0 else center - radius,
        1.0 if k == n else center + radius,
    )


@lru_cache(maxsize=1)
def stream():
    return generate()


def opening_run():
    data, log, request = opening()
    fast = LogShipmentTools(data, event_log=log)
    settled = EventTimeTools(data, event_log=log)
    fast_trace = Agent(fast).run(request)
    held = Agent(settled).run(request)
    complete_at = first_complete(log, request.at)
    final_at = first_complete(log, request.at + ALLOWED)
    settled.data = read_sources(data, final_at)
    final = Agent(settled).run(request, at=final_at)
    arrival = next(
        e
        for e in log.entries
        if e.order_id == request.order_reference and e.status == "delivered"
    )
    records = corrections(
        log,
        [PaymentRecord(p, "executed") for p in fast.payments],
        through=arrival.arrived_at,
    )
    return dict(
        data=data,
        log=log,
        request=request,
        fast=fast,
        fast_trace=fast_trace,
        settled=settled,
        held=held,
        final=final,
        arrival=arrival,
        complete_at=complete_at,
        final_at=final_at,
        corrections=records,
    )


def replay_run():
    run = opening_run()
    log, request, fast = run["log"], run["request"], run["fast"]
    original_scan = next(
        e for e in log.entries if e.order_id == request.order_reference
    )
    copy = original_scan.model_copy(
        update={
            "arrived_at": original_scan.arrived_at + timedelta(minutes=5),
        }
    )
    log = EventLog(log.entries + (copy,))
    before = json.dumps(
        {k: v.model_dump(mode="json") for k, v in fast.ledger.items()},
        sort_keys=True,
    )
    through = run["final_at"]
    calls = []
    dispatch = fast._dispatch

    def no_dispatch(*args, **kwargs):
        calls.append(None)
        raise AssertionError("pure replay dispatched a tool")

    fast._dispatch = no_dispatch
    try:
        live = Views()
        for offset, event in log.arrived_by(through):
            live = apply(live, offset, event)
        rebuilt = replay(log, through=through)
        original = replay(log, through=request.at)
        saved = checkpoint(original)
        continued = restore(saved, log, through=through)
        shipments, decisions = {}, {}
        for name, views, at in (
            ("original", original, request.at),
            ("corrected", rebuilt, through),
        ):
            shipment = views.shipment(request.order_reference, at)
            data = read_sources(run["data"], at).model_copy(
                update={"shipments": [shipment]}
            )
            state = c.State(
                request=request, trace_id="replay", execution_at=at
            )
            state.order, _, _ = sources(data, state)
            state.shipment_events = shipment.events
            state.policy = data.policy_at(at)
            chosen = assess(data, state)
            if (
                chosen.outcome == c.Outcome.APPROVE
                and request.reason == "not_delivered"
                and not complete_through(
                    EventLog(views.arrivals), at, MARGIN, request.at + ALLOWED
                )
            ):
                chosen = decision(
                    state, c.Outcome.REFER, "shipment_incomplete"
                )
            shipments[name] = shipment
            decisions[name] = chosen.model_dump(mode="json")
    finally:
        fast._dispatch = dispatch
    calls = len(calls)
    after = json.dumps(
        {k: v.model_dump(mode="json") for k, v in fast.ledger.items()},
        sort_keys=True,
    )
    shadow = corrections(
        log, [PaymentRecord(p, "shadow") for p in fast.payments]
    )
    return {
        "views_equal": live == rebuilt,
        "ledger_unchanged": before == after,
        "checkpoint_equal": continued == rebuilt,
        "tool_calls": calls,
        "copies": len(rebuilt.arrivals) - rebuilt.distinct,
        "cutoff": request.at,
        "processing_version": PROCESSING_VERSION,
        "original": shipments["original"],
        "corrected": shipments["corrected"],
        "original_decision": decisions["original"],
        "corrected_decision": decisions["corrected"],
        "decisions_equal": all(
            decisions[name] == run[recorded].decision.model_dump(mode="json")
            for name, recorded in (
                ("original", "held"),
                ("corrected", "final"),
            )
        ),
        "corrections": run["corrections"] + shadow,
    }


def count_run():
    log = stream().log
    end = START + timedelta(days=1)
    times = (end, first_complete(log, end), first_complete(log, end + ALLOWED))
    rows = [dict(at=t, **day_counts(log, START, t)) for t in times]
    distinct = {
        e.event_id
        for e in log.entries
        if START <= e.at < START + timedelta(days=DAYS)
    }
    late = [e for e in late_events(log) if e.event_id in distinct]
    return {
        "rows": rows,
        "late": len(late),
        "scans": len(distinct),
        "interval": wilson(len(late), len(distinct)),
    }


def _waits(values):
    return {
        "requests": len(values),
        "mean_hours": sum(values) / len(values) if values else None,
        "longest_hours": max(values) if values else None,
    }


@lru_cache(maxsize=1)
def claim_run():
    source = stream()
    rows, populations = [], {}
    finals = {
        r.request_id: first_complete(source.log, r.at + ALLOWED)
        for r in source.requests
    }
    for rule, allowed in (
        ("fast", None),
        ("watermark", timedelta(0)),
        ("final", ALLOWED),
    ):
        payments, held_waits, honest_waits, final_waits, decisions = (
            [],
            [],
            [],
            [],
            [],
        )
        for request in source.requests:
            data = dataset((request,), source.log)
            tools = (
                LogShipmentTools(data, event_log=source.log)
                if allowed is None
                else EventTimeTools(
                    data, event_log=source.log, allowed_lateness=allowed
                )
            )
            initial = trace = Agent(tools).run(request)
            terminal_at = request.at
            if trace.decision.reason == "shipment_incomplete":
                terminal_at = first_complete(source.log, request.at + allowed)
                tools.data = read_sources(data, terminal_at)
                trace = Agent(tools).run(request, at=terminal_at)
                held_waits.append(
                    (terminal_at - request.at).total_seconds() / 3600
                )
            elapsed = (terminal_at - request.at).total_seconds() / 3600
            if request.request_id in source.honest:
                honest_waits.append(elapsed)
            final_waits.append(
                (finals[request.request_id] - request.at).total_seconds()
                / 3600
            )
            payments.extend(
                PaymentRecord(p, "executed") for p in tools.payments
            )
            decisions.append(
                {
                    "request_id": request.request_id,
                    "initial": initial.decision.model_dump(mode="json"),
                    "terminal": trace.decision.model_dump(mode="json"),
                    "terminal_at": terminal_at.isoformat(),
                    "wait_hours": elapsed,
                    "honest": request.request_id in source.honest,
                }
            )
        corrected = corrections(source.log, payments)
        shadow = corrections(
            source.log, [PaymentRecord(p.receipt, "shadow") for p in payments]
        )
        contradicted, paid = len(corrected), len(payments)
        rows.append(
            {
                "rule": rule,
                "claims": len(source.requests),
                "payments": paid,
                "contradicted": contradicted,
                "rho_late": contradicted / paid,
                "interval": wilson(contradicted, paid),
                "awaiting_absence": _waits(held_waits),
                "honest": _waits(honest_waits),
                "evidence_final": _waits(final_waits),
                "corrections": {
                    "executed": len(corrected),
                    "shadow": len(shadow),
                },
            }
        )
        populations[rule] = {
            "decisions": decisions,
            "corrections": [asdict(r) for r in corrected + shadow],
        }
    # Originals only: a copy's arrival delay is not a delivery scan's delay.
    unique = {}
    for event in source.log.entries:
        unique.setdefault(event.event_id, event)
    longest = max(
        (e.arrived_at - e.at).total_seconds() / 3600 for e in unique.values()
    )
    return {
        "rows": rows,
        "populations": populations,
        "longest_delay_hours": longest,
        "longest_delay_at_most_36_hours": longest <= 36,
    }


def exercise_scans():
    times = (
        ("s1", 8, 0, 8, 1),
        ("s2", 9, 30, 9, 31),
        ("s3", 11, 0, 11, 2),
        ("s3", 11, 0, 11, 40),
        ("s4", 13, 15, 16, 45),
        ("s5", 15, 0, 15, 3),
        ("s6", 21, 10, 23, 55),
        ("s7", 23, 30, 25, 30),
        ("s8", 23, 50, 29, 0),
        ("s9", 20, 0, 33, 0),
    )
    return [
        ScanEvent(
            event_id=name,
            order_id=f"parcel-{name}",
            status="delivered",
            at=START + timedelta(hours=eh, minutes=em),
            arrived_at=START + timedelta(hours=ph, minutes=pm),
        )
        for name, eh, em, ph, pm in times
    ]


def payment_exercise():
    request_at = START + timedelta(hours=12)
    made = request_at - timedelta(days=31, hours=1)
    arrived = made + timedelta(days=2)
    observed_at = request_at - timedelta(seconds=1790)
    events = [
        ScanEvent(
            event_id="exercise-05-shipped",
            order_id="exercise-05-parcel",
            status="shipped",
            at=made - timedelta(days=1),
            arrived_at=made - timedelta(days=1),
        ),
        ScanEvent(
            event_id="exercise-05-shipped",
            order_id="exercise-05-parcel",
            status="shipped",
            at=made - timedelta(days=1),
            arrived_at=made - timedelta(days=1) + timedelta(minutes=5),
        ),
        ScanEvent(
            event_id="exercise-05-delivered",
            order_id="exercise-05-parcel",
            status="delivered",
            at=made,
            arrived_at=arrived,
        ),
        scan("exercise-05-background", "shipped", observed_at, observed_at),
    ]
    return {
        "receipt_id": "batch-stream-exercise-05-receipt",
        "amount_cents": 4499,
        "request_id": "batch-stream-exercise-05",
        "order_id": "exercise-05-parcel",
        "reason": "damaged",
        "request_at": request_at.isoformat(),
        "policy_return_days": 30,
        "policy_version": (START - timedelta(days=90)).date().isoformat(),
        "outcome": "approve",
        "executed": True,
        "window_clock_used": "arrived_at",
        "record_age_seconds": 1790,
        "watermark": watermark(EventLog(events), request_at).isoformat(),
        "events": [e.model_dump(mode="json") for e in events],
    }


def figure_run():
    log = stream().log
    late = {e.event_id for e in late_events(log)}
    seen, points = set(), []
    for event in log.entries:
        if START <= event.at < START + timedelta(days=1):
            points.append(
                {
                    "event_id": event.event_id,
                    "event_hours": (event.at - START).total_seconds() / 3600,
                    "arrival_hours": (event.arrived_at - START).total_seconds()
                    / 3600,
                    "late": event.event_id in late,
                    "copy": event.event_id in seen,
                }
            )
        seen.add(event.event_id)
    end = START + timedelta(days=1)
    final_at = first_complete(log, end + ALLOWED)
    last_hour = ceil((final_at - START).total_seconds() / 3600)
    times = [START + timedelta(hours=h) for h in range(last_hour + 1)]
    return {
        "stream_fingerprint": stream().fingerprint,
        "points": points,
        "counts": [
            {
                "processing_hours": (t - START).total_seconds() / 3600,
                **day_counts(log, START, t),
            }
            for t in times
        ],
        "on_time": first_complete(log, end),
        "final": final_at,
    }


def json_default(value):
    if isinstance(value, c.Record):
        return value.model_dump(mode="json")
    return value.isoformat() if hasattr(value, "isoformat") else asdict(value)
