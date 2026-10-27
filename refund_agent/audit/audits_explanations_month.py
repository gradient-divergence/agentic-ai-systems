"""Constructed source records, separate hidden truth, and actual candidate runs.

Regions must arrive from chapter 34 before outcomes are generated. This module
does not implement region_of or substitute for event-time, promotion or scope.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from pydantic import Field

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_rates import (
    AuditObservation,
    Period,
)
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    SegmentAssignments,
)
from refund_agent.audit.audits_explanations_stage6 import ExplanationAttachment
from refund_agent.data.generate import Dataset, number
from refund_agent.memory.register import clause_text
from refund_agent.release.agent_evaluation_resources import (
    ConstructedDataset,
    run_constructed_request,
)
from refund_agent.release.candidate import Candidate

START = datetime(2026, 5, 1, tzinfo=timezone.utc)


class RegionSettings(c.Record):
    region: str
    missing_return_scan: float = 0.05
    expired_return: float = 0.10
    false_return_claim: float = 0.02
    arrived_not_delivered_claim: float
    missing_arrived_scan: float = 0.02
    false_delivery_scan: float


SETTINGS = (
    RegionSettings(
        region="A", arrived_not_delivered_claim=0.80, false_delivery_scan=0.02
    ),
    RegionSettings(
        region="B", arrived_not_delivered_claim=0.90, false_delivery_scan=0.02
    ),
    RegionSettings(
        region="C", arrived_not_delivered_claim=0.80, false_delivery_scan=0.50
    ),
)


class Truth(c.Record):
    request_id: str
    received: bool
    genuine: bool
    owed: bool
    found_false_if_paid: bool


class Month(c.Record):
    seed: int
    period: Period
    settings: tuple[RegionSettings, ...]
    segments: SegmentAssignments
    data: Dataset
    truth: tuple[Truth, ...]
    history: dict[str, tuple[EvaluationRecord, ...]]
    evaluation_times: dict[str, datetime]

    def tool_data(self) -> Dataset:
        return self.data.model_copy(deep=True, update={"correct": []})


def _draw(seed, i, name, probability):
    return number(seed, i, "audits-explanations-audit-" + name, 10000) < int(
        probability * 10000
    )


def constructed_month(
    segments: SegmentAssignments, seed: int = 7, n: int = 2000
) -> Month:
    if type(seed) is not int or type(n) is not int or n <= 0:
        raise ValueError(
            "month needs an integer seed and positive request count"
        )
    segments = SegmentAssignments.model_validate(segments)
    if any(g not in {"A", "B", "C"} for g in segments.by_order.values()):
        raise ValueError("chapter 34 region names must be A, B and C")
    settings = {r.region: r for r in SETTINGS}
    effective = START - timedelta(days=30)
    policy = c.Policy(
        version=effective.date().isoformat(),
        effective_at=effective,
        return_days=30,
        cap_cents=100000,
        text=f"refund_cap_v4@{effective.date().isoformat()}: physical goods. "
        f"#eligibility: {clause_text('eligibility', 30)} "
        f"#cap: {clause_text('cap', 100000)}",
    )
    orders, requests, shipments, retrievals, labels, truths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    histories, times = {}, {}
    for i in range(n):
        oid = f"audits-explanations-o-{i:04d}"
        setting = settings[segments.region_of(oid)]
        at = START + timedelta(
            days=number(seed, i, "audits-explanations-audit-day", 30),
            seconds=number(seed, i, "audits-explanations-audit-second", 86400),
        )
        reason = ("damaged", "changed_mind", "not_delivered")[i % 3]
        amount = 1 + number(seed, i, "audits-explanations-audit-amount", 4999)
        if reason == "not_delivered":
            received = _draw(
                seed, i, "arrived", setting.arrived_not_delivered_claim
            )
            scanned = (
                not _draw(
                    seed, i, "missing-arrived", setting.missing_arrived_scan
                )
                if received
                else (
                    _draw(
                        seed, i, "false-delivery", setting.false_delivery_scan
                    )
                )
            )
            expired, genuine, owed = False, True, not received
        else:
            received = True
            scanned = not _draw(
                seed, i, "missing-return", setting.missing_return_scan
            )
            expired = _draw(seed, i, "expired", setting.expired_return)
            genuine = not _draw(
                seed, i, "false-return", setting.false_return_claim
            )
            owed = genuine and not expired
        delivered_at = at - timedelta(days=32 if expired else 3)
        purchased = delivered_at - timedelta(days=5)
        request = c.Request(
            request_id=f"audits-explanations-r-{seed}-{i:04d}",
            requester_id=f"audits-explanations-customer-{seed}-{i:04d}",
            order_reference=oid,
            at=at - timedelta(hours=36),
            amount_cents=amount,
            reason=reason,
            text="Please review this physical-goods refund request.",
        )
        order = c.Order(
            order_id=oid,
            reference=oid,
            purchased_at=purchased,
            observed_at=at - timedelta(seconds=120),
            items=[
                c.Item(
                    name="physical desk lamp", quantity=1, unit_cents=100000
                )
            ],
            paid_cents=100000,
            payment_id=f"audits-explanations-payment-{i:04d}",
        )
        events = [
            c.ShipmentEvent(at=purchased + timedelta(days=1), status="shipped")
        ]
        if scanned:
            events.append(c.ShipmentEvent(at=delivered_at, status="delivered"))
        shipment = c.Shipment(
            order_id=oid,
            observed_at=at - timedelta(seconds=120),
            events=events,
        )
        snapshot = c.PolicySnapshot(
            request_id=request.request_id,
            version=policy.version,
            observed_at=at - timedelta(hours=1),
        )
        if expired and scanned:
            boundary = delivered_at + timedelta(days=30)
            histories[request.request_id] = (
                EvaluationRecord(
                    request=request,
                    order=order.model_copy(
                        update={
                            "observed_at": boundary - timedelta(seconds=120)
                        }
                    ),
                    shipment=shipment.model_copy(
                        update={
                            "observed_at": boundary - timedelta(seconds=120)
                        }
                    ),
                    policy=policy,
                    policy_observed_at=boundary - timedelta(hours=1),
                    at=boundary,
                    refunded_cents=0,
                ),
            )
        truth = Truth(
            request_id=request.request_id,
            received=received,
            genuine=genuine,
            owed=owed,
            found_false_if_paid=not owed,
        )
        labels.append(
            c.Decision(
                request_id=request.request_id,
                outcome=c.Outcome.APPROVE if owed else c.Outcome.DENY,
                amount_cents=amount,
                tier=c.Tier.AGENT,
                reason="constructed_owed" if owed else "constructed_not_owed",
                policy_citation=f"refund_cap_v4@{policy.version}#eligibility",
            )
        )
        orders.append(order)
        requests.append(request)
        shipments.append(shipment)
        retrievals.append(snapshot)
        truths.append(truth)
        times[request.request_id] = at
    return Month(
        seed=seed,
        period=Period(start=START, end=START + timedelta(days=30)),
        settings=SETTINGS,
        segments=segments,
        data=Dataset(
            orders=orders,
            requests=requests,
            shipments=shipments,
            policies=[policy],
            retrievals=retrievals,
            correct=labels,
            reviewers=[
                c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
                c.Reviewer(
                    reviewer_id="morgan", name="Morgan", role="named_approver"
                ),
            ],
        ),
        truth=tuple(truths),
        history=histories,
        evaluation_times=times,
    )


class MonthRun(c.Record):
    candidate: dict
    traces: tuple[c.Trace, ...]
    explanations: tuple[ExplanationAttachment, ...]
    observations: tuple[AuditObservation, ...]
    paid_cents: int = Field(ge=0)


def run_month(
    month: Month,
    candidate: Candidate,
    context_for: Callable[[Dataset, c.Request], dict],
) -> MonthRun:
    """One real run per order, with caller-provided integrated tools/context.

    Orders are disjoint with no balance carry-over. Mounting one per run avoids
    quadratic corpus copying while retaining the core's payment/audit stores.
    """
    data = month.tool_data()
    orders = {o.order_id: o for o in data.orders}
    shipments = {s.order_id: s for s in data.shipments}
    retrievals = {s.request_id: s for s in data.retrievals}
    truths = {t.request_id: t for t in month.truth}
    traces, attached, observations = [], [], []
    cents = 0
    for request in data.requests:
        oid = request.order_reference
        local = Dataset(
            orders=[orders[oid]],
            requests=[request],
            shipments=[shipments[oid]],
            policies=data.policies,
            retrievals=[retrievals[request.request_id]],
            correct=[],
            reviewers=data.reviewers,
        )
        result = run_constructed_request(
            ConstructedDataset(local, source="generated"),
            candidate,
            request,
            at=month.evaluation_times[request.request_id],
            **context_for(local, request),
        )
        trace = result.trace
        traces.append(trace)
        explanations = result.explanations
        if request.request_id in explanations:
            attached.append(explanations[request.request_id])
        held = request.request_id in result.held_requests
        cents += sum(p.amount_cents for p in result.payments)
        # Only the audit joins the isolated truth to a completed decision.
        truth = truths[request.request_id]
        observations.append(
            AuditObservation(
                request_id=request.request_id,
                order_id=oid,
                segment=month.segments.region_of(oid),
                at=month.evaluation_times[request.request_id],
                outcome=trace.decision.outcome.value,
                reason=(
                    explanations[request.request_id].proposed_decision.reason
                    if held else trace.decision.reason
                ),
                held_proposal=held,
                owed=truth.owed,
                found_false_if_paid=truth.found_false_if_paid,
                label_mature_at=month.evaluation_times[request.request_id]
                + timedelta(days=30),
            )
        )
    return MonthRun(
        candidate=candidate.record(),
        traces=tuple(traces),
        explanations=tuple(attached),
        observations=tuple(observations),
        paid_cents=cents,
    )
