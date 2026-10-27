"""Compute the rule's facts before spending the remaining context on ranked records."""

from dataclasses import dataclass
from datetime import timedelta

from refund_agent import contracts as c
from refund_agent import rules


def required_facts(state):
    request, order, policy = state.request, state.order, state.policy
    if order is None or policy is None:
        raise ValueError("missing_evidence")
    at = state.execution_at or request.at
    delivered = [event.at for event in state.shipment_events if event.status == "delivered"]
    latest = max(delivered) if delivered else None
    return {
        "request_id": request.request_id,
        "at": at,
        "known_at": at,
        "amount_cents": request.amount_cents,
        "reason": request.reason,
        "balance_cents": order.paid_cents - state.refunded_cents,
        "purchased_at": order.purchased_at,
        "delivered_at": latest,
        "delivery_age_days": (at - latest).total_seconds() / 86400 if latest else None,
        "future_evidence": order.purchased_at > at or any(e.at > at for e in state.shipment_events),
        "policy_version": policy.version,
        "return_days": policy.return_days,
        "cap_cents": policy.cap_cents,
        "named_approver": policy.named_approver,
        "snapshot_at": at - timedelta(seconds=state.freshness.get("policy", 0)),
    }


def fact_lines(facts):
    return tuple(
        f"{key}: {value.isoformat() if hasattr(value, 'isoformat') else value}"
        for key, value in facts.items()
    )


def draft_from_facts(facts):
    """The same structured eligibility decision, without reading the cut records."""
    amount, reason = facts["amount_cents"], facts["reason"]
    section, outcome, why = "eligibility", c.Outcome.APPROVE, "eligible"
    if amount > min(facts["cap_cents"], facts["balance_cents"]):
        section, outcome, why = "cap", c.Outcome.DENY, "amount_over_policy"
    elif reason not in ("damaged", "changed_mind", "not_delivered"):
        outcome, why = c.Outcome.REFER, "unknown_reason"
    elif facts["future_evidence"]:
        outcome, why = c.Outcome.REFER, "future_evidence"
    elif reason == "not_delivered" and facts["delivered_at"] is not None:
        outcome, why = c.Outcome.DENY, "already_delivered"
    elif reason != "not_delivered" and facts["delivered_at"] is None:
        outcome, why = c.Outcome.REFER, "delivery_unknown"
    elif reason != "not_delivered" and facts["delivery_age_days"] > facts["return_days"]:
        outcome, why = c.Outcome.DENY, "outside_return_window"
    return c.Decision(
        request_id=facts["request_id"],
        amount_cents=amount,
        tier=rules.tier_for(amount),
        outcome=outcome,
        reason=why,
        policy_citation=f"refund_cap_v4@{facts['policy_version']}#{section}",
    )


@dataclass(frozen=True)
class Assembly:
    facts: dict
    lines: tuple[str, ...]
    kept: tuple[str, ...]
    cut: tuple[str, ...]
    words: int
    refusal: str | None = None


def assemble(state, records, budget):
    """Budget includes fixed sections and required lines; draft reservation stays outside."""
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    facts = required_facts(state)
    lines = fact_lines(facts)
    used = sum(len(line.split()) for line in lines)
    if used > budget:
        return Assembly(
            {}, (), (), tuple(r["id"] for r in records), used, "required_facts_exceed_budget"
        )
    kept, cut = [], []
    for record in records:
        size = len(record["text"].split())
        if used + size <= budget:
            kept.append(record["id"])
            lines += (record["text"],)
            used += size
        else:
            cut.append(record["id"])
    return Assembly(facts, lines, tuple(kept), tuple(cut), used)


def last_scans_summary(events, n):
    if type(n) is not int or n < 0:
        raise ValueError("n must be nonnegative")
    return sorted(events, key=lambda e: e.at)[-n:] if n else []
