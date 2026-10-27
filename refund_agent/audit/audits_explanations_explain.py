"""Rule-record explanations. Counterfactuals clear eligibility, grant no authority.

Historical evidence is an explicit input. Moving an evaluation clock never
pretends that a snapshot observed afterwards was available at the earlier time.
"""

import json
from datetime import datetime, timedelta
from typing import Literal

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ExplanationContext,
    ReviewRoute,
)
from refund_agent.memory.register import current_records
from refund_agent.rules import draft, tier_for

GENERATOR_VERSION = "audits-explanations-explain-1"
DENIAL_REASONS = (
    "amount_over_policy",
    "already_delivered",
    "outside_return_window",
)
LIMITS = {
    "already_delivered": "A delivery scan is the carrier's record of delivery. "
    "It does not show who received the parcel.",
    "outside_return_window": "The delivery time is the carrier's scan time.",
    "amount_over_policy": "The balance is read from the payment ledger.",
}
NEXT = {
    "already_delivered": "Ask for a person to review if the parcel did not arrive.",
    "outside_return_window": "Ask for a person to review if the delivery date is wrong.",
    "amount_over_policy": "A request up to the payable amount is assessed again. "
    "Ask for a person to review this denial.",
}
DESCRIPTIONS = {
    "amount_over_policy": "The amount exceeds the policy cap or remaining payment balance.",
    "already_delivered": "The carrier recorded a delivery.",
    "outside_return_window": "The evaluation occurred after the return window closed.",
}
CHANGE_LABELS = {
    "request.amount_cents": "Requested amount in cents",
    "evaluation.at": "Evaluation time",
    "shipment.deliveries": "Recorded delivery scans",
}


class Failure(c.Record):
    reason: str
    section: Literal["cap", "eligibility"]
    fields: tuple[str, ...]


class Clause(c.Record):
    reason: str
    citation: str
    text: str
    version: str


class EvidenceItem(c.Record):
    field: str
    value: str
    source: str
    record_id: str
    order_id: str | None
    observed_at: datetime
    age_seconds: float


class Change(c.Record):
    field: str
    before: str
    after: str


class Counterfactual(c.Record):
    changes: tuple[Change, ...]
    evaluation: EvaluationRecord | None
    outcome: Literal["eligible", "still_ineligible", "no_admissible_change"]
    approval_required: str
    approval_granted: bool = False


class DenialExplanation(c.Record):
    request_id: str
    generator_version: str = GENERATOR_VERSION
    proposed: bool
    reasons: tuple[str, ...]
    clauses: tuple[Clause, ...]
    evidence: tuple[EvidenceItem, ...]
    counterfactual: Counterfactual
    limits: tuple[str, ...]
    next_step: ReviewRoute
    rendered: str


def failing_clauses(state: c.State, refunded: int) -> tuple[Failure, ...]:
    if state.order is None or state.policy is None:
        raise ValueError("a denial explanation requires complete evidence")
    if type(refunded) is not int or refunded < 0:
        raise ValueError("refunded must be nonnegative integer cents")
    request, order, policy = state.request, state.order, state.policy
    failures = []
    if request.amount_cents > min(
        policy.cap_cents, order.paid_cents - refunded
    ):
        failures.append(
            Failure(
                reason="amount_over_policy",
                section="cap",
                fields=(
                    "request.amount_cents",
                    "policy.cap_cents",
                    "order.paid_cents",
                    "ledger.refunded_cents",
                ),
            )
        )
    delivered = [
        e.at for e in state.shipment_events if e.status == "delivered"
    ]
    if request.reason == "not_delivered" and delivered:
        failures.append(
            Failure(
                reason="already_delivered",
                section="eligibility",
                fields=("request.reason", "shipment.deliveries"),
            )
        )
    if (
        request.reason in ("damaged", "changed_mind")
        and delivered
        and (
            (state.execution_at or request.at) - max(delivered)
        ).total_seconds()
        > policy.return_days * 86400
    ):
        failures.append(
            Failure(
                reason="outside_return_window",
                section="eligibility",
                fields=(
                    "request.reason",
                    "shipment.deliveries",
                    "evaluation.at",
                    "policy.return_days",
                ),
            )
        )
    return tuple(failures)


def _policy_valid(
    record: EvaluationRecord, context: ExplanationContext
) -> bool:
    clauses = [
        r
        for r in current_records(context.register.records, record.at)
        if r.valid_from <= record.at
        and (r.valid_to is None or record.at < r.valid_to)
    ]
    return len(clauses) == 2 and all(
        r.version == record.policy.version
        and r.value
        == (
            record.policy.cap_cents
            if r.section == "cap"
            else record.policy.return_days
        )
        and r.named_approver == record.policy.named_approver
        for r in clauses
    )


def _bound_context(
    state: c.State, refunded: int, context: ExplanationContext
) -> None:
    r = EvaluationRecord.model_validate(context.current)
    if not (
        r.request == state.request
        and r.order == state.order
        and r.shipment.events == state.shipment_events
        and r.policy == state.policy
        and r.at == (state.execution_at or state.request.at)
        and r.refunded_cents == refunded
        and r.state().freshness == state.freshness
    ):
        raise ValueError(
            "explanation context differs from effective evaluation evidence"
        )
    if not _policy_valid(r, context) or context.register.at < r.at:
        raise ValueError("clause version is not in force at evaluation time")
    if context.review_route.request_id != state.request.request_id:
        raise ValueError("review route names another request")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fields(record: EvaluationRecord) -> dict[str, EvidenceItem]:
    r, o, s, p, at = (
        record.request,
        record.order,
        record.shipment,
        record.policy,
        record.at,
    )
    fields = {
        "request.amount_cents": (
            r.amount_cents,
            "request",
            r.request_id,
            o.order_id,
            min(r.at, at),
        ),
        "request.reason": (
            r.reason,
            "request",
            r.request_id,
            o.order_id,
            min(r.at, at),
        ),
        "policy.cap_cents": (
            p.cap_cents,
            "policy.retrieve",
            p.version,
            None,
            record.policy_observed_at,
        ),
        "policy.return_days": (
            p.return_days,
            "policy.retrieve",
            p.version,
            None,
            record.policy_observed_at,
        ),
        "order.paid_cents": (
            o.paid_cents,
            "order.lookup",
            o.order_id,
            o.order_id,
            o.observed_at,
        ),
        "ledger.refunded_cents": (
            record.refunded_cents,
            "ledger",
            o.order_id,
            o.order_id,
            at,
        ),
        "shipment.deliveries": (
            [e.at.isoformat() for e in s.events if e.status == "delivered"],
            "shipment.status",
            s.order_id,
            o.order_id,
            s.observed_at,
        ),
        "evaluation.at": (
            at.isoformat(),
            "evaluation_clock",
            r.request_id,
            o.order_id,
            at,
        ),
    }
    return {
        name: EvidenceItem(
            field=name,
            value=_json(value),
            source=source,
            record_id=rid,
            order_id=oid,
            observed_at=observed,
            age_seconds=(at - observed).total_seconds(),
        )
        for name, (value, source, rid, oid, observed) in fields.items()
    }


def _evidence(failures, record):
    names = tuple(dict.fromkeys(name for f in failures for name in f.fields))
    values = _fields(record)
    return tuple(values[name] for name in names)


def _clauses(failures, context):
    record = context.current
    versions = [
        r
        for r in current_records(context.register.records, record.at)
        if r.valid_from <= record.at
        and (r.valid_to is None or record.at < r.valid_to)
    ]
    result = []
    for failure in failures:
        clause = next(r for r in versions if r.section == failure.section)
        text = context.register.texts.get(clause.text_hash)
        if text is None:
            raise ValueError(
                "policy clause text is not available in the register"
            )
        result.append(
            Clause(
                reason=failure.reason,
                citation=clause.citation,
                text=text,
                version=clause.version,
            )
        )
    return tuple(result)


def _changed(source, result):
    values = (
        (
            "request.amount_cents",
            source.request.amount_cents,
            result.request.amount_cents,
        ),
        ("evaluation.at", source.at.isoformat(), result.at.isoformat()),
        (
            "shipment.deliveries",
            [
                e.at.isoformat()
                for e in source.shipment.events
                if e.status == "delivered"
            ],
            [
                e.at.isoformat()
                for e in result.shipment.events
                if e.status == "delivered"
            ],
        ),
    )
    return tuple(
        Change(field=f, before=_json(a), after=_json(b))
        for f, a, b in values
        if a != b
    )


def counterfactual(state, failures, *, context):
    reasons = {f.reason for f in failures}
    current = context.current
    if "outside_return_window" in reasons:
        candidates = []
        for record in context.history:
            record = EvaluationRecord.model_validate(record)
            delivered = [
                e.at for e in record.shipment.events if e.status == "delivered"
            ]
            if (
                record.request == current.request
                and record.order.order_id == current.order.order_id
                and record.at <= current.at
                and delivered
                and _policy_valid(record, context)
            ):
                end = max(delivered) + timedelta(
                    days=record.policy.return_days
                )
                policy_ends = [
                    r.valid_to
                    for r in current_records(
                        context.register.records, current.at
                    )
                    if r.version == record.policy.version
                    and r.valid_to is not None
                ]
                if policy_ends:
                    end = min(
                        end, min(policy_ends) - timedelta(microseconds=1)
                    )
                if record.at == end:
                    candidates.append(record)
        if not candidates:
            raise ValueError(
                "historical evidence at the last eligible evaluation time is required"
            )
    else:
        candidates = [current]
    results = []
    for source in candidates:
        request, shipment = source.request, source.shipment
        if "amount_over_policy" in reasons or request.amount_cents > min(
            source.policy.cap_cents,
            source.order.paid_cents - source.refunded_cents,
        ):
            payable = min(
                source.policy.cap_cents,
                source.order.paid_cents - source.refunded_cents,
            )
            if payable <= 0:
                continue
            request = request.model_copy(
                update={"amount_cents": min(request.amount_cents, payable)}
            )
        if "already_delivered" in reasons:
            shipment = shipment.model_copy(
                update={
                    "events": [
                        e for e in shipment.events if e.status != "delivered"
                    ]
                }
            )
        changed = EvaluationRecord.model_validate(
            {**source.model_dump(), "request": request, "shipment": shipment}
        )
        result = draft(changed.state())
        results.append(
            Counterfactual(
                changes=_changed(current, changed),
                evaluation=changed,
                outcome="eligible"
                if result.reason == "eligible"
                else "still_ineligible",
                approval_required=context.approval_required,
            )
        )
    if not results:
        return Counterfactual(
            changes=(),
            evaluation=None,
            outcome="no_admissible_change",
            approval_required=context.approval_required,
        )
    eligible = [r for r in results if r.outcome == "eligible"]
    return min(
        eligible or results,
        key=lambda r: (len(r.changes), -r.evaluation.at.timestamp()),
    )


def _render(reasons, clauses, cf, route, proposed):
    sentences = [
        "The agent proposes to deny your request. A person must decide."
        if proposed
        else "Your request was denied."
    ]
    for reason, clause in zip(reasons, clauses):
        sentences.append(f"{DESCRIPTIONS[reason]} Policy: {clause.citation}.")
        sentences.append(LIMITS[reason])
        sentences.append(NEXT[reason])
    if cf.outcome == "no_admissible_change":
        sentences.append(
            "No admissible change of evidence makes this request policy-eligible."
        )
        sentences.append("Ask for a person to review this denial.")
    else:
        sentences.append(
            "The proposed changes still fail eligibility."
            if cf.outcome == "still_ineligible"
            else "The changed evidence establishes policy eligibility. No payment is authorized."
        )
        for change in cf.changes:
            sentences.append(
                f"{CHANGE_LABELS[change.field]}: {change.before} becomes {change.after}."
            )
        sentences.append(f"Required approval remains: {cf.approval_required}.")
    sentences.append(
        f"Review contact: {route.contact}. Ticket: {route.ticket_id}."
    )
    sentences.append(f"Review deadline: {route.deadline.isoformat()}.")
    return "\n".join(sentences)


# region book:fnd.audits-explanations.explain-core-01
def explain(state, decision, refunded, *, context):
    _bound_context(state, refunded, context)  # (1)
    failures = failing_clauses(state, refunded)  # (2)
    _check_denial(decision, state, failures)  # (3)
    clauses = _clauses(failures, context)  # (4)
    changed = counterfactual(state, failures, context=context)  # (5)
    if changed.outcome == "still_ineligible":  # (6)
        raise ValueError(
            "counterfactual does not clear every eligibility clause"
        )
    reasons = tuple(f.reason for f in failures)
    return DenialExplanation(
        request_id=decision.request_id,
        proposed=decision.outcome == c.Outcome.REFER,
        reasons=reasons,
        clauses=clauses,
        evidence=_evidence(failures, context.current),
        counterfactual=changed,
        limits=tuple(LIMITS[r] for r in reasons),
        next_step=context.review_route,
        rendered=_render(  # (7)
            reasons,
            clauses,
            changed,
            context.review_route,
            decision.outcome == c.Outcome.REFER,
        ),
    )


def replay(explanation, state, *, context):
    _bound_context(state, context.current.refunded_cents, context)
    failures = failing_clauses(state, context.current.refunded_cents)
    clauses = _clauses(failures, context)
    return _replay_checks(explanation, state, failures, clauses, context)  # (8)


# endregion book:fnd.audits-explanations.explain-core-01


def _check_denial(decision, state, failures):
    if (
        decision.outcome not in (c.Outcome.DENY, c.Outcome.REFER)
        or not failures
        or decision.reason != failures[0].reason
        or decision.request_id != state.request.request_id
        or decision.amount_cents != state.request.amount_cents
        or decision.tier != tier_for(state.request.amount_cents)
        or decision.executed
    ):
        raise ValueError(
            "explain requires a denial or referral retaining its denial reason"
        )
    citation = f"refund_cap_v4@{state.policy.version}#{failures[0].section}"
    if decision.policy_citation != citation:
        raise ValueError("decision clause version or anchor is not in force")


def _replay_checks(e, state, failures, clauses, context):
    evidence = _evidence(failures, context.current)
    cf = counterfactual(state, failures, context=context)
    reasons = tuple(f.reason for f in failures)
    decision = draft(state, context.current.refunded_cents)
    cf_state = (
        e.counterfactual.evaluation.state()
        if e.counterfactual.evaluation
        else None
    )
    cf_valid = (
        e.counterfactual == cf
        and not e.counterfactual.approval_granted
        and e.counterfactual.approval_required == context.approval_required
        and (
            cf.outcome == "no_admissible_change"
            or cf_state is not None
            and draft(cf_state).reason == "eligible"
        )
    )
    visible = context.may_act(
        state.request.requester_id, context.current.order.order_id
    )
    return {
        "denial": bool(reasons)
        and decision.outcome == c.Outcome.DENY
        and e.request_id == state.request.request_id
        and e.reasons[:1] == reasons[:1],
        "counterfactual": cf_valid,
        "evidence": all(item in evidence for item in e.evidence),
        "disclosure": visible
        and all(
            item.order_id in (None, context.current.order.order_id)
            for item in e.evidence
        )
        and e.rendered
        == _render(
            e.reasons, e.clauses, e.counterfactual, e.next_step, e.proposed
        )
        and e.next_step == context.review_route
        and e.limits == tuple(LIMITS[r] for r in e.reasons),
        "version": all(clause in clauses for clause in e.clauses)
        and e.generator_version == GENERATOR_VERSION,
        "completeness": e.reasons == reasons
        and e.clauses == clauses
        and e.evidence == evidence
        and (
            state.decision is None
            or e.proposed == (state.decision.outcome == c.Outcome.REFER)
        ),
    }
