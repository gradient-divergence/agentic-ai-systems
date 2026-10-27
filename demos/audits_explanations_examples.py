"""Explicit chapter-36 demonstration records, not credentials or partner components."""

import json
from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ExplanationContext,
    ReviewRoute,
)
from refund_agent.memory.register import PolicyRegister, clause_text

AT = datetime(2026, 5, 1, tzinfo=timezone.utc)


def example_records(kind="two-clause") -> dict:
    choices = {
        "two-clause": (5200, 4000, 0, "damaged", 31),
        "exhausted": (2499, 4000, 4000, "damaged", 10),
        "tier2": (5200, 100000, 0, "not_delivered", 10),
        "tier3": (65000, 100000, 0, "not_delivered", 10),
    }
    amount, paid, refunded, reason, day = choices[kind]
    at = AT + timedelta(days=day)
    policy = c.Policy(
        version=AT.date().isoformat(),
        effective_at=AT,
        return_days=30,
        cap_cents=100000,
        text=f"refund_cap_v4@{AT.date().isoformat()}: physical goods. "
        f"#eligibility: {clause_text('eligibility', 30)} "
        f"#cap: {clause_text('cap', 100000)}",
    )
    request = c.Request(
        request_id=f"audits-explanations-{kind}",
        requester_id="audit-customer",
        order_reference="audits-explanations-order",
        at=at - timedelta(days=2),
        amount_cents=amount,
        reason=reason,
    )
    order = c.Order(
        order_id=request.order_reference,
        reference=request.order_reference,
        purchased_at=AT - timedelta(days=5),
        observed_at=at - timedelta(seconds=120),
        items=[c.Item(name="physical desk lamp", quantity=1, unit_cents=paid)],
        paid_cents=paid,
        payment_id="audits-explanations-payment",
    )
    shipment = c.Shipment(
        order_id=order.order_id,
        observed_at=at - timedelta(seconds=120),
        events=[
            c.ShipmentEvent(at=AT - timedelta(days=4), status="shipped"),
            c.ShipmentEvent(at=AT, status="delivered"),
        ],
    )
    current = EvaluationRecord(
        request=request,
        order=order,
        shipment=shipment,
        policy=policy,
        policy_observed_at=at - timedelta(hours=1),
        at=at,
        refunded_cents=refunded,
    )
    history = ()
    if day == 31:
        end = AT + timedelta(days=30)
        history = (
            EvaluationRecord(
                request=request,
                order=order.model_copy(
                    update={"observed_at": end - timedelta(seconds=120)}
                ),
                shipment=shipment.model_copy(
                    update={"observed_at": end - timedelta(seconds=120)}
                ),
                policy=policy,
                policy_observed_at=end - timedelta(hours=1),
                at=end,
                refunded_cents=refunded,
            ),
        )
    route = ReviewRoute(
        request_id=request.request_id,
        ticket_id=f"review-{request.request_id}",
        deadline=at + timedelta(days=2),
    )
    return {
        "current": current.model_dump(mode="json"),
        "history": [r.model_dump(mode="json") for r in history],
        "review_route": route.model_dump(mode="json"),
    }


def context_from_records(records, *, may_act):
    current = EvaluationRecord.model_validate_json(
        json.dumps(records["current"])
    )
    history = tuple(
        EvaluationRecord.model_validate_json(json.dumps(row))
        for row in records["history"]
    )
    register = PolicyRegister()
    register.announce(current.policy, current.policy.effective_at)
    route = ReviewRoute.model_validate_json(
        json.dumps(records["review_route"])
    )
    return ExplanationContext(
        current, register.snapshot(current.at), history, may_act, route
    )
