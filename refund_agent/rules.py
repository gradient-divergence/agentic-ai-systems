"""Policy and tier checks; none of these is a model-callable tool."""

import math
from datetime import datetime, timedelta
from types import MappingProxyType as _ReadOnly

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset

_FRESHNESS_LIMITS = _ReadOnly({"policy": 30 * 86400, "order": 30 * 60, "shipment": 30 * 60})
_RISK_THRESHOLD = 0.2  # Fictional stage-1 threshold, not a calibrated estimate.


def tier_for(amount_cents: int) -> c.Tier:
    if type(amount_cents) is not int or amount_cents <= 0:
        raise ValueError("amount must be positive integer cents")
    if amount_cents < 5000:
        return c.Tier.AGENT
    return c.Tier.REVIEWER if amount_cents <= 50000 else c.Tier.NAMED_APPROVER


def risk_score(state: c.State) -> float:
    # ponytail: fixed text rule; replace with a measured, calibrated score in the risk stage.
    return 0.9 if "ignore" in state.request.text.casefold() else 0.1


def decision(
    state: c.State,
    outcome: c.Outcome,
    reason: str,
    section: str = "eligibility",
    executed: bool = False,
) -> c.Decision:
    citation = None if state.policy is None else f"refund_cap_v4@{state.policy.version}#{section}"
    return c.Decision(
        request_id=state.request.request_id,
        outcome=outcome,
        amount_cents=state.request.amount_cents,
        tier=tier_for(state.request.amount_cents),
        reason=reason,
        policy_citation=citation,
        executed=executed,
    )


def draft(state: c.State, refunded_cents: int | None = None) -> c.Decision:
    """Eligibility from observed evidence, before execution authorization."""
    request, order, policy = state.request, state.order, state.policy
    if order is None or policy is None:
        return decision(state, c.Outcome.REFER, "missing_evidence")
    at = state.execution_at or request.at
    refunded = state.refunded_cents if refunded_cents is None else refunded_cents
    if request.amount_cents > min(policy.cap_cents, order.paid_cents - refunded):
        return decision(state, c.Outcome.DENY, "amount_over_policy", "cap")
    if request.reason not in ("damaged", "changed_mind", "not_delivered"):
        return decision(state, c.Outcome.REFER, "unknown_reason")
    if order.purchased_at > at or any(event.at > at for event in state.shipment_events):
        return decision(state, c.Outcome.REFER, "future_evidence")
    delivered = [e.at for e in state.shipment_events if e.status == "delivered"]
    if request.reason == "not_delivered":
        if delivered:
            return decision(state, c.Outcome.DENY, "already_delivered")
    elif not delivered:
        return decision(state, c.Outcome.REFER, "delivery_unknown")
    elif (at - max(delivered)).total_seconds() > policy.return_days * 86400:
        return decision(state, c.Outcome.DENY, "outside_return_window")
    return decision(state, c.Outcome.APPROVE, "eligible")


def sources(
    data: Dataset, state: c.State
) -> tuple[c.Order | None, c.Shipment | None, c.PolicySnapshot | None]:
    orders = [o for o in data.orders if state.request.order_reference in (o.order_id, o.reference)]
    order = orders[0] if len(orders) == 1 else None
    shipment = next((s for s in data.shipments if order and s.order_id == order.order_id), None)
    snapshot = next((s for s in data.retrievals if s.request_id == state.request.request_id), None)
    return order, shipment, snapshot


def refresh_summaries(data: Dataset, state: c.State) -> None:
    """Summaries are display fields; timestamps and the scoring rule are authoritative."""
    order, shipment, snapshot = sources(data, state)
    observations = {"order": order, "shipment": shipment, "policy": snapshot}
    state.freshness = {
        name: ((state.execution_at or state.request.at) - source.observed_at).total_seconds()
        for name, source in observations.items()
        if source is not None
    }
    state.policy_version = state.policy.version if state.policy else None
    state.risk_score = risk_score(state)


def freshness_problem(freshness: dict[str, float]) -> str | None:
    for source, limit in _FRESHNESS_LIMITS.items():
        age = freshness.get(source)
        if age is None:
            return f"missing_{source}_freshness"
        if not math.isfinite(age):
            return f"invalid_{source}_freshness"
        if age < 0:
            return f"future_{source}"
        if age > limit:
            return f"stale_{source}"
    return None


def evidence_problem(data: Dataset, state: c.State) -> str | None:
    if state.order is None or state.policy is None or not state.shipment_events:
        return "missing_evidence"
    order, shipment, snapshot = sources(data, state)
    if order is None or shipment is None or snapshot is None:
        return "missing_evidence"
    if state.order != order or state.shipment_events != shipment.events:
        return "evidence_mismatch"
    try:
        current = data.policy_at(state.execution_at or state.request.at)
    except ValueError:
        return "non_current_policy"
    if state.policy != current or snapshot.version != current.version:
        return "non_current_policy"
    refresh_summaries(data, state)
    problem = freshness_problem(state.freshness)
    if problem:
        return problem
    return "suspicious_request" if risk_score(state) >= _RISK_THRESHOLD else None


def assess(data: Dataset, state: c.State, refunded_cents: int | None = None) -> c.Decision:
    problem = evidence_problem(data, state)
    return decision(state, c.Outcome.REFER, problem) if problem else draft(state, refunded_cents)


def execution_time(state: c.State, approval: c.Approval | None) -> datetime:
    if state.execution_at is not None:
        return state.execution_at
    # The offline clock advances past any recorded approval, without reading wall time.
    latest = max(state.request.at, approval.approved_at) if approval else state.request.at
    return latest + timedelta(microseconds=1)


# region book:fnd.action-guardrails.approval-01
def proposal_problem(
    state: c.State, proposal: c.Proposal | None, amount: int
) -> str | None:
    if proposal is None or state.order is None or state.policy is None:
        return "proposal_required"
    fields = (
        ("request", proposal.request != state.request),
        ("order", proposal.evidence.order.order_id != state.order.order_id),
        ("amount", proposal.amount_cents != amount),
        ("reason", proposal.reason != state.request.reason),
        (
            "policy_version",
            proposal.evidence.policy.version != state.policy.version,
        ),
    )
    changed = [name for name, differs in fields if differs]
    return "proposal_changed:" + ",".join(changed) if changed else None


def approval_problem(
    data: Dataset,
    state: c.State,
    proposal: c.Proposal,
    approval: c.Approval | None,
) -> str | None:
    if approval is None:
        return "approval_required"
    assert state.order is not None and state.policy is not None
    person = next(
        (r for r in data.reviewers if r.reviewer_id == approval.reviewer), None
    )
    if (
        type(approval.amount_cents) is not int
        or approval.request_id != state.request.request_id
        or approval.order_id != state.order.order_id
        or approval.proposal_id != proposal.proposal_id
        or approval.amount_cents != proposal.amount_cents
        or approval.reason != proposal.reason
        or approval.policy_version != state.policy.version
        or person is None
        or person.role != approval.role
        or approval.role not in ("reviewer", "named_approver")
        or approval.reviewer
        in (
            "refund-agent",
            state.request.preparer_id,
            state.request.requester_id,
        )
        or approval.approved_at
        < max(
            proposal.created_at,
            state.order.purchased_at,
            state.policy.effective_at,
        )
        or (
            tier_for(proposal.amount_cents) == c.Tier.NAMED_APPROVER
            and (
                approval.role != "named_approver"
                or approval.reviewer != state.policy.named_approver
            )
        )
    ):
        return "invalid_approval"
    return None
# endregion book:fnd.action-guardrails.approval-01


# region book:fnd.action-guardrails.gate-01
def gate(
    data: Dataset,
    state: c.State,
    amount_cents: int,
    proposal: c.Proposal | None,
    approval: c.Approval | None,
    refunded_cents: int = 0,
) -> str | None:
    """All new payments pass this gate, including direct tool calls."""
    if (
        type(amount_cents) is not int
        or amount_cents <= 0
        or amount_cents != state.request.amount_cents
    ):
        return "amount_mismatch"
    problem = evidence_problem(data, state) or proposal_problem(
        state, proposal, amount_cents
    )
    if problem:
        return problem
    assert proposal is not None
    assessed = draft(state, refunded_cents)
    if assessed.outcome != c.Outcome.APPROVE:
        return assessed.reason
    if execution_time(state, approval) < proposal.created_at:
        return "execution_before_proposal"
    if tier_for(amount_cents) == c.Tier.AGENT and approval is None:
        return None
    if approval and approval.approved_at >= execution_time(state, approval):
        return "invalid_approval"
    return approval_problem(data, state, proposal, approval)
# endregion book:fnd.action-guardrails.gate-01
