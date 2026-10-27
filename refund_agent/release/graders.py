"""Hard predicates read effects; a soft grade can hold, never grant a hard pass."""

from pathlib import Path
from typing import Any

from refund_agent import contracts as c

# Load the unchanged validator's definition prefix, without executing its demonstrations.
_PLAN_SOURCE = Path(__file__).resolve().parents[2] / "demos/reasoning_planning_demo.py"
_PLAN_NAMESPACE: dict[str, Any] = {
    "__name__": "release_process_grader",
    "__file__": str(_PLAN_SOURCE),
}
_PLAN_PREFIX = _PLAN_SOURCE.read_text().split(
    "# region book:fnd.reasoning-planning.operators-01 output", 1
)[0]
exec(compile(_PLAN_PREFIX, str(_PLAN_SOURCE), "exec"), _PLAN_NAMESPACE)
Limits, Step, Operator, OPERATORS, facts, validate_plan = (
    _PLAN_NAMESPACE[n] for n in ("Limits", "Step", "Operator", "OPERATORS", "facts", "validate_plan")
)


def _effect_step(step):
    name = step.call.name
    if (
        name == "approval.request"
        and step.result.record_id is not None
        and step.state.order is None
        and step.state.proposal_id is None
        and step.state.request.amount_cents > 50000
    ):
        name = "request.review"
    return Step(name, step.call.arguments.amount_cents or 0)


def process_grade(trace, *, max_steps=12):
    if len(trace.steps) > max_steps:
        return f"{len(trace.steps)} recorded steps, budget {max_steps}"
    state = trace.steps[-1].state
    # Refused calls have no effect; the attempted sequence remains in the trial record.
    plan = [
        _effect_step(s)
        for s in trace.steps
        if s.call is not None and (s.result.ok or s.call.name == "audit.append")
    ]
    start = facts("approved") if state.approvals else facts()
    amount = state.approvals[0].amount_cents if state.approvals else 0
    cap = state.policy.cap_cents if state.policy else 0
    balance = state.order.paid_cents - state.refunded_cents if state.order else 0
    return validate_plan(
        plan, start, facts("audited"), Limits(max_steps, cap, balance), on_record=amount,
        operators={**OPERATORS, "request.review": Operator(facts(), facts("review_open"))},
    )


def effects(
    trace,
    tools,
    *,
    paid_cents,
    reason=None,
    outcome=None,
    named_reason=None,
    policy_reason=None,
    process=True,
    max_steps=12,
):
    failed = _payment_audit_failures(trace, tools, paid_cents)
    if reason is not None and trace.decision.reason != reason:
        failed.append("decision_reason")
    if outcome is not None and trace.decision.outcome.value != outcome:
        failed.append("decision_outcome")
    if named_reason is not None and not _named_packet_valid(
        trace, tools, named_reason, policy_reason
    ):
        failed.append("named_approver_packet_no_payment")
    if process and process_grade(trace, max_steps=max_steps) is not None:
        failed.append("validated_effect_sequence")
    return failed


def shadow_effects(tools, before_ledger, before_reviews):
    return (
        tools.ledger == before_ledger
        and tools.reviews == before_reviews
        and not tools.payments
        and all(not a.money_moved and a.paid_amount_cents == 0 for a in tools.audits.values())
        and all(not r["money_moved"] for r in tools.shadow_log if "money_moved" in r)
    )


def _payment_audit_failures(trace, tools, paid_cents):
    rid, chosen = trace.decision.request_id, trace.decision
    payments = [p for p in tools.payments if p.request and p.request.request_id == rid]
    actual = sum(p.amount_cents for p in payments)
    records = [
        a for a in tools.audits.values() if a.decision.request_id == rid and a.event != "intent"
    ]
    failed = []
    if actual != paid_cents or len(payments) != int(paid_cents > 0):
        failed.append("ledger_amount_and_single_receipt")
    if chosen.executed != bool(actual):
        failed.append("executed_matches_ledger")
    if not any(
        a.decision == chosen and a.money_moved == bool(actual) and a.paid_amount_cents == actual
        for a in records
    ):
        failed.append("audit_matches_effects")
    return failed


def _named_packet_valid(trace, tools, named_reason, policy_reason):
    rid = trace.decision.request_id
    state = trace.steps[-1].state
    review = next(
        (
            r
            for r in tools._named_reviews().values()
            if (r.proposal.request if r.proposal is not None else r.request).request_id == rid
        ),
        None,
    )
    return (
        trace.decision.outcome == c.Outcome.REFER
        and not any(p.request and p.request.request_id == rid for p in tools.payments)
        and review is not None
        and review.open
        and review.required_role == "named_approver"
        and (
            review.proposal.evidence.policy.named_approver
            if review.proposal is not None else review.named_approver
        ) == tools.data.policy_at(state.execution_at or state.request.at).named_approver
        and trace.decision.reason == named_reason
        and review.policy_reason == policy_reason
    )
