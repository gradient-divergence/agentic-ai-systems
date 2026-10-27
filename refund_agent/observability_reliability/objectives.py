"""Chapter 34 classifies policy-valid answers separately from safe behavior."""

import math
from datetime import datetime, timedelta
from typing import Literal

from pydantic import model_validator

from refund_agent import contracts as c
from refund_agent.batch_stream.replay import CorrectionRecord, PaymentRecord
from refund_agent.observability_reliability.reasons import reason_class
from refund_agent.rules import draft, freshness_problem, risk_score


def payment_request(receipt: c.Receipt) -> c.Request:
    if receipt.request is None:
        raise ValueError("payment observation requires its recorded request")
    return receipt.request


def payment_evidence(receipt: c.Receipt) -> c.Evidence:
    if receipt.evidence is None:
        raise ValueError("payment observation requires its recorded evidence")
    return receipt.evidence


class Observation(c.Record):
    request_id: str
    received_at: datetime
    decided_at: datetime | None
    outcome: Literal["approve", "deny", "refer"] | None
    reason: str
    promise_hours: int
    tier: int = 1

    @model_validator(mode="after")
    def timing(self):
        if self.promise_hours not in (1, 48) or self.tier not in (1, 2, 3):
            raise ValueError("unknown promise or tier")
        if (self.outcome is None) != (self.decided_at is None):
            raise ValueError("answer and time must be present together")
        if self.decided_at and self.decided_at < self.received_at:
            raise ValueError("answer before request")
        return self

    @property
    def deadline(self):
        return self.received_at + timedelta(hours=self.promise_hours)


class Classification(c.Record):
    request_id: str
    timeliness: Literal["good", "bad", "pending", "not_counted"]
    corrections: Literal["good", "bad", "pending", "not_counted"]
    breaches: tuple[str, ...]


def receipt_breaches(payment: PaymentRecord, payments=()) -> tuple[str, ...]:
    receipt = payment.receipt
    evidence = payment_evidence(receipt)
    request = payment_request(receipt)
    state = c.State(
        request=request,
        trace_id=evidence.trace_id,
        order=evidence.order,
        shipment_events=evidence.shipment_events,
        policy=evidence.policy,
        freshness=evidence.freshness,
        execution_at=receipt.at,
    )
    state.refunded_cents = sum(
        p.receipt.amount_cents
        for p in payments
        if p.mode == payment.mode
        and p.receipt.order_id == receipt.order_id
        and p.receipt.at <= receipt.at
        and p.receipt.receipt_id != receipt.receipt_id
    )
    prohibited = (
        draft(state).outcome != c.Outcome.APPROVE or risk_score(state) >= 0.2
    )
    prohibited |= receipt.amount_cents != request.amount_cents
    result = ["forbidden_payment"] if prohibited else []
    if freshness_problem(evidence.freshness):
        result.append("stale_payment")
    return tuple(result)


def classify(
    observation: Observation,
    payments: tuple[PaymentRecord, ...] = (),
    corrections: tuple[CorrectionRecord, ...] = (),
    *,
    now: datetime,
    outcomes_final=False,
    extra_reasons=None,
) -> Classification:
    now = c._utc_time(now)
    if observation.tier != 1:
        return Classification(
            request_id=observation.request_id,
            timeliness="not_counted",
            corrections="not_counted",
            breaches=(),
        )
    kind = (
        reason_class(observation.reason, extra=extra_reasons)
        if observation.reason
        else "hold"
    )
    answered = (
        observation.decided_at is not None
        and observation.decided_at <= now
        and kind != "hold"
    )
    timely = "pending"
    if answered:
        timely = (
            "good"
            if kind == "policy"
            and observation.decided_at <= observation.deadline
            else "bad"
        )
    elif now > observation.deadline:
        timely = "bad"
    own = tuple(
        p
        for p in payments
        if payment_request(p.receipt).request_id == observation.request_id
        and p.receipt.at <= now
    )
    contradicted = {
        (r.mode, r.payment_id) for r in corrections if r.arrived_at <= now
    }
    corrected = any(
        (p.mode, p.receipt.receipt_id) in contradicted for p in own
    )
    correction_status = (
        ("bad" if corrected else "good" if outcomes_final else "pending")
        if own
        else "not_counted"
    )
    breaches = tuple(
        sorted({b for p in own for b in receipt_breaches(p, payments)})
    )
    return Classification(
        request_id=observation.request_id,
        timeliness=timely,
        corrections=correction_status,
        breaches=breaches,
    )


class DenialPair(c.Record):
    request_id: str
    agent_denied: bool
    baseline_denied: bool
    person_paid_at: datetime | None
    found_false: bool
    label_mature_at: datetime

    @model_validator(mode="after")
    def mature_label_window(self):
        if (
            self.person_paid_at is not None
            and self.label_mature_at < self.person_paid_at + timedelta(days=30)
        ):
            raise ValueError("D6 label matures before its paid outcome window")
        return self


def paired_denials(rows: tuple[DenialPair, ...], *, now: datetime) -> dict:
    if len({r.request_id for r in rows}) != len(rows):
        raise ValueError("D6 requires the same distinct requests in both arms")
    mature = tuple(r for r in rows if now >= r.label_mature_at)
    agent = sum(
        r.person_paid_at is not None and r.agent_denied and not r.found_false
        for r in mature
    )
    baseline = sum(
        r.person_paid_at is not None
        and r.baseline_denied
        and not r.found_false
        for r in mature
    )
    return {
        "request_ids": [r.request_id for r in rows],
        "mature": len(mature),
        "pending": len(rows) - len(mature),
        "agent": agent,
        "baseline": baseline,
        "pass": agent <= baseline if len(mature) == len(rows) else None,
        "population": "paired canary-period requests; separate from the signed 299 start window",
    }


def nearest_rank(values, q):
    values = tuple(values)
    if not values:
        raise ValueError("empty measurement")
    if not 0 < q <= 1 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError(
            "finite nonnegative durations and quantile in (0, 1] required"
        )
    return sorted(values)[math.ceil(len(values) * q) - 1]


def share_within(durations, deadline):
    values = tuple(durations)
    nearest_rank(values, 1)
    if not math.isfinite(deadline) or deadline < 0:
        raise ValueError("finite nonnegative deadline required")
    return sum(v <= deadline for v in values) / len(values)
