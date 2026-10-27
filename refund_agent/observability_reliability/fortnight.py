"""Constructed tier 1 shadow control and chapter 32's cohort rehearsal.

Both consume the fixed scan stream and chapter 31's source handouts.
Planting depends only on fixed requests/scans, before any decision is executed.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from refund_agent import contracts as c
from refund_agent.batch_stream.generate import (
    COPY_EVERY,
    COPY_MAX_SECONDS,
    DEPOT_HOURS,
    FAST_MEAN_SECONDS,
    FAST_SHARE,
    FILE_HOURS,
    SCANS_PER_DAY,
    read_sources,
    scan,
)
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.replay import PaymentRecord, corrections
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    first_complete,
    shipment_as_of,
    watermark,
)
from refund_agent.canary.cloud_native_deployment_component import (
    read_contracts,
)
from refund_agent.canary.cloud_native_deployment_component import (
    runtime_paths as canary_paths,
)
from refund_agent.canary.cloud_native_deployment_plan import (
    CanaryPlan,
    sign_plan,
)
from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal
from refund_agent.canary.cloud_native_deployment_start import _budget
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import AppendStore
from refund_agent.data.live import LiveStream, generate_live
from refund_agent.observability_reliability.budget import signed_budget
from refund_agent.observability_reliability.monitor import Monitor
from refund_agent.observability_reliability.objectives import (
    Observation,
    payment_evidence,
)
from refund_agent.observability_reliability.silence import (
    region_logs,
    region_of,
    replay_detectors,
)
from refund_agent.observability_reliability.spans import SpanStore

# Candidate/suite construction is the demo's responsibility, outside the
# component's resource scope. Execution uses the core's full selected guard.
from refund_agent.release.candidate import build_tools, run_request
from refund_agent.release.extensions import resource_guard
from refund_agent.selection import fingerprint

START = datetime(2026, 10, 1, tzinfo=timezone.utc)
SEED = 34
DAYS = 14
CANARY_ID = "observability-reliability-fortnight"


@dataclass(frozen=True)
class Silence:
    region: int
    start: datetime
    end: datetime
    request_id: str
    event_id: str


@dataclass(frozen=True)
class Fortnight:
    live: LiveStream
    requests: tuple[c.Request, ...]
    log: EventLog
    silence: Silence
    fingerprint: str


def planting_cohort(requester_id: str) -> bool:
    """The brief's fixed hash, used only to plant; this never binds a payment."""
    digest = sha256(f"{CANARY_ID}:{requester_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10000 < 500


def plant(requests, log: EventLog) -> Silence:
    delivered = {e.order_id: e for e in log.entries if e.status == "delivered"}
    for day in range(2, 11):
        for request in sorted(requests, key=lambda r: (r.at, r.request_id)):
            if (
                request.at.date() != (START + timedelta(days=day)).date()
                or request.reason != "not_delivered"
                or not planting_cohort(request.requester_id)
            ):
                continue
            event = delivered.get(request.order_reference)
            if event and timedelta(0) <= request.at - event.at <= timedelta(
                hours=10
            ):
                start = event.at - timedelta(hours=1)
                return Silence(
                    region_of(event.order_id),
                    start,
                    start + timedelta(hours=48),
                    request.request_id,
                    event.event_id,
                )
    raise ValueError("no qualifying cohort claim; fixed seed is not changed")


def make_stream() -> Fortnight:
    # The seven-amount cycle has TWO tier 1 amounts. 234 total/day gives
    # 936 tier 1 requests in fourteen days, ~2000/month; no seed search.
    live = generate_live(SEED, START, DAYS, 234)
    requests = tuple(
        sorted(
            (r for r in live.data.requests if r.amount_cents < 5000),
            key=lambda r: (r.at, r.request_id),
        )
    )
    shipments = {s.order_id: s for s in live.data.shipments}
    assigned = {}
    rng = random.Random(SEED)
    for index, request in enumerate(requests):
        if request.reason == "not_delivered":
            made = request.at - timedelta(hours=rng.uniform(0, 24))
            rows = (
                (("shipped", made - timedelta(days=2)), ("delivered", made))
                if index % 5 != 4
                else (("shipped", request.at - timedelta(days=2)),)
            )
        else:
            # Other reasons retain chapter 29's scan times and policy timing.
            rows = tuple(
                (e.status, e.at)
                for e in shipments[request.order_reference].events
            )
        for status, made in rows:
            assigned.setdefault(made.date(), []).append(
                (request.order_reference, status, made)
            )
    for day in range(-4, DAYS + 3):
        at = START + timedelta(days=day)
        rows = assigned.setdefault(at.date(), [])
        remaining = SCANS_PER_DAY - len(rows)
        if remaining < 0:
            raise ValueError(
                "assigned scans exceed chapter 31's daily scan volume"
            )
        rows.extend(
            (
                f"observability-reliability-feed-{day}-{j}",
                "shipped",
                at + timedelta(seconds=(j + 0.5) * 86400 / remaining),
            )
            for j in range(remaining)
        )
    events = []
    for day in sorted(assigned):
        for index, (order_id, status, made) in enumerate(
            sorted(assigned[day], key=lambda row: row[2])
        ):
            draw = rng.random()
            delay = (
                rng.expovariate(1 / FAST_MEAN_SECONDS)
                if draw < FAST_SHARE
                else rng.uniform(*DEPOT_HOURS) * 3600
                if draw < 0.99
                else rng.uniform(*FILE_HOURS) * 3600
            )
            event = scan(
                order_id, status, made, made + timedelta(seconds=delay)
            )
            events.append(event)
            if index % COPY_EVERY == 0:
                events.append(
                    event.model_copy(
                        update={
                            "arrived_at": event.arrived_at
                            + timedelta(
                                seconds=rng.uniform(0, COPY_MAX_SECONDS)
                            )
                        }
                    )
                )
    original = EventLog(events)
    silence = plant(requests, original)
    log = EventLog(
        e.model_copy(update={"arrived_at": silence.end})
        if region_of(e.order_id) == silence.region
        and silence.start <= e.arrived_at < silence.end
        else e
        for e in original.entries
    )
    identity = fingerprint(
        {
            "seed": SEED,
            "live": live.fingerprint,
            "events": [e.model_dump(mode="json") for e in log.entries],
            "plant": {
                "request": silence.request_id,
                "event": silence.event_id,
                "region": silence.region,
                "start": silence.start.isoformat(),
                "end": silence.end.isoformat(),
            },
        }
    )
    return Fortnight(live, requests, log, silence, identity)


def _request_data(stream: Fortnight, request):
    original = stream.live.data
    order = next(
        o for o in original.orders if o.order_id == request.order_reference
    )
    shipment = next(
        s for s in original.shipments if s.order_id == order.order_id
    )
    retrieval = next(
        s for s in original.retrievals if s.request_id == request.request_id
    )
    return original.model_copy(
        update={
            "requests": [request],
            "orders": [order],
            "shipments": [shipment],
            "retrievals": [retrieval],
            "correct": [],
        }
    )


@resource_guard(0)
def run_shadow_fortnight(candidate, stream: Fortnight, *, fix: bool) -> dict:
    selected = next(
        r
        for r in candidate.tools_composition["components"]
        if r["name"] == "observability-reliability"
    )
    if selected["settings"]["region_finality"] != fix:
        raise ValueError("fortnight finality differs from selected candidate")
    store = SpanStore()
    traces, lineage, audits, payments, observations = [], [], [], [], []
    inputs = []
    regions = region_logs(stream.log)
    for request in stream.requests:
        data = source_snapshot(
            _request_data(stream, request), stream.log, request.at
        )
        tools = build_tools(
            data, candidate, "shadow", event_log=stream.log, span_store=store
        )
        trace = run_request(candidate, tools, request)
        traces.append(trace)
        held = trace.decision.reason == "shipment_incomplete"
        if held:
            progress = (
                regions[region_of(request.order_reference)]
                if fix
                else stream.log
            )
            at = first_complete(progress, request.at + ALLOWED, MARGIN)
            event = next(
                e
                for e in progress.entries
                if e.at >= request.at + ALLOWED + MARGIN
            )
            advance = store.causal(
                request, at, "watermark_advance", reads=(event.event_id,)
            )
            tools._trace_links = ((advance.span_id, "released_by"),)
            tools.data = source_snapshot(data, stream.log, at)
            trace = run_request(candidate, tools, request, at=at)
            traces.append(trace)
        answered_at = trace.steps[-1].state.execution_at or request.at
        observations.append(
            Observation(
                request_id=request.request_id,
                received_at=request.at,
                decided_at=answered_at,
                outcome=trace.decision.outcome.value,
                reason=trace.decision.reason,
                promise_hours=48 if held else 1,
            )
        )
        lineage.extend(tools.lineage)
        inputs.append(tools.record_store.export())
        audits.extend(tools.audits.values())
        payments.extend(
            PaymentRecord(p, "shadow") for p in tools.shadow_payments
        )
        if tools.payments:
            raise ValueError(
                "chapter 34's explicit shadow experiment moved money"
            )
    changes = corrections(stream.log, payments)
    request_by_id = {r.request_id: r for r in stream.requests}
    for correction in changes:
        payment_span = next(
            s
            for s in store.trace(f"trace-{correction.request_id}")
            if s.name == "store:ledger" and correction.payment_id in s.writes
        )
        store.causal(
            request_by_id[correction.request_id],
            correction.arrived_at,
            "correction",
            reads=(correction.event_id, correction.payment_id),
            writes=(f"correction:{correction.mode}:{correction.payment_id}",),
            links=((payment_span.span_id, "corrects"),),
        )
    budget_record = signed_budget(CANARY_ID, candidate.fingerprint, START)
    monitor = Monitor(budget_record)
    for observation in observations:
        monitor.observe(observation)
    for payment in payments:
        monitor.record_payment(payment)
    for correction in sorted(changes, key=lambda r: r.arrived_at):
        monitor.record_correction(correction)
    monitor.tick(START + timedelta(days=28), outcomes_final=True)
    alerts = monitor.close_windows(monitor.now)
    pages = replay_detectors(stream.log, START, START + timedelta(days=DAYS))
    return {
        "stream": stream,
        "candidate": candidate,
        "store": store,
        "traces": tuple(traces),
        "lineage": tuple(lineage),
        "audits": tuple(audits),
        "payments": tuple(payments),
        "observations": tuple(observations),
        "inputs": inputs,
        "corrections": changes,
        "monitor": monitor,
        "alerts": alerts,
        "pages": pages,
        "page": pages[0]["alert"] if pages else None,
        "fix": fix,
        "watermark_at_batch": watermark(stream.log, stream.silence.end),
    }


def source_snapshot(data, log, at):
    """Explicit fixture provider: each source handout matches its as-of log.

    The original event-time wrapper restores its backing dataset before an audit
    call. Supplying the actual shipment snapshot keeps chapter 30's denial check
    and lineage consistent. No tool manufactures observation timestamps here.
    """
    return read_sources(data, at).model_copy(
        update={
            "shipments": [
                shipment_as_of(log, order.order_id, at)
                for order in data.orders
            ]
        }
    )


@resource_guard(0)
def run_canary_rehearsal(candidate, target, stream: Fortnight, *, fix: bool):
    """Use the real cohort binding; an incident is not rollback admission."""
    handouts = tuple(
        source_snapshot(_request_data(stream, r), stream.log, r.at)
        for r in stream.requests
    )
    data = stream.live.data.model_copy(
        update={
            "requests": list(stream.requests),
            "orders": [d.orders[0] for d in handouts],
            "shipments": [d.shipments[0] for d in handouts],
            "retrievals": [d.retrievals[0] for d in handouts],
            "correct": [],
        }
    )
    live = LiveStream(
        data,
        {
            **stream.live.manifest,
            "scope": "observability-reliability tier 1 incident rehearsal",
            "parent_stream": stream.fingerprint,
            "source_handouts": "chapter 31 read_sources and as-of shipments",
        },
    )
    label = "fixed" if fix else "as-built"
    plan = sign_plan(
        CanaryPlan(
            canary_id=CANARY_ID,
            candidate_fingerprint=candidate.fingerprint,
            suite_fingerprint=candidate.suite["fingerprint"],
            rollback_target_fingerprint=target.fingerprint,
            requirement_ref="observability-reliability-rehearsal-only",
            rehearsal_ref="observability-reliability-rehearsal-" + label,
            error_budget_ref="observability-reliability-budget-" + label,
        ),
        "morgan",
        START - timedelta(seconds=1),
    )
    store = SpanStore()
    budget_record = signed_budget(CANARY_ID, candidate.fingerprint, START)
    record = run_rehearsal(
        plan,
        live,
        DAYS,
        [stream.silence],
        "4h",
        target,
        candidate=candidate,
        event_log=stream.log,
        context={
            "tools_context": {
                "span_store": store,
                "observability_reliability_budget": budget_record,
            }
        },
    )
    path = canary_paths(candidate)["store"]
    rows = AppendStore(path, "observability-reliability-reader").rows(
        "rehearsal",
        all_namespaces=True,
    )
    written = next(
        row
        for row in reversed(rows)
        if row["key"] == record.rehearsal_id
        and row["value"] == record.model_dump(mode="json")
    )
    records = AppendStore(path, written["namespace"])
    receipts = tuple(
        p
        for p in records.records("ledger", read_contracts(candidate)).values()
        if p.request is not None
    )
    payments = tuple(PaymentRecord(p, "executed") for p in receipts)
    changes = corrections(stream.log, payments)
    for change in changes:
        payment = next(
            p for p in receipts if p.receipt_id == change.payment_id
        )
        paid_span = next(
            s
            for s in store.trace(payment_evidence(payment).trace_id)
            if s.name == "store:ledger" and payment.receipt_id in s.writes
        )
        store.causal(
            payment.request,
            change.arrived_at,
            "correction",
            reads=(change.event_id, change.payment_id),
            writes=(f"correction:executed:{change.payment_id}",),
            links=((paid_span.span_id, "corrects"),),
        )
    session = CanarySession(plan, records)
    admission = records.latest("budget", plan.error_budget_ref)
    _budget(admission, plan, START)
    monitor = Monitor.for_canary(session, candidate)
    for change in changes:
        monitor.record_correction(change)
    return {
        "record": record,
        "plan": plan,
        "candidate": candidate,
        "store": store,
        "payments": payments,
        "corrections": changes,
        "stops": session.stops,
        "audits": tuple(
            records.records("audits", read_contracts(candidate)).values()
        ),
        "budget": budget_record,
        "budget_admission": admission,
        "monitor": monitor,
    }
