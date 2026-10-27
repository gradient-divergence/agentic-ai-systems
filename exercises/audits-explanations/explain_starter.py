"""Exercise 6: complete clause collection and a joint admissible counterfactual.

The worked amount-only case constructs a valid record but omits other failures.
Complete collect_failures and joint_counterfactual; retain the source-bound
record, replay checks and required approval without granting that approval.
"""

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_explain import (
    LIMITS,
    Counterfactual,
    DenialExplanation,
    Failure,
    _bound_context,
    _changed,
    _check_denial,
    _clauses,
    _evidence,
    _render,
)
from refund_agent.audit.audits_explanations_records import EvaluationRecord
from refund_agent.rules import draft


def collect_failures(state, refunded):
    """Collect every cap, recorded-delivery and return-window violation.

    The amount check is supplied as a worked case. Add the independent delivery
    and time checks, preserving all failures when more than one clause fails.
    """
    if state.order is None or state.policy is None:
        raise ValueError("a denial explanation requires complete evidence")
    failures = []
    payable = min(state.policy.cap_cents, state.order.paid_cents - refunded)
    if state.request.amount_cents > payable:
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
    return tuple(failures)


def joint_counterfactual(state, failures, *, context):
    """Complete the joint change, retaining admissible historical evidence.

    This worked case changes only the requested amount at the current clock.
    Use context.history for a historical evaluation, handle recorded delivery,
    and clear every collected failure together. A new clock cannot invent a
    source read. If no admissible change exists, report that outcome. Keep the
    required approval separate from eligibility and never grant it here.
    """
    current = context.current
    payable = min(current.policy.cap_cents, current.order.paid_cents - current.refunded_cents)
    if payable <= 0:
        return Counterfactual(
            changes=(),
            evaluation=None,
            outcome="no_admissible_change",
            approval_required=context.approval_required,
        )
    request = current.request.model_copy(
        update={"amount_cents": min(current.request.amount_cents, payable)}
    )
    changed = EvaluationRecord.model_validate({**current.model_dump(), "request": request})
    outcome = "eligible" if draft(changed.state()).reason == "eligible" else "still_ineligible"
    return Counterfactual(
        changes=_changed(current, changed),
        evaluation=changed,
        outcome=outcome,
        approval_required=context.approval_required,
    )


def explain(state, decision, refunded, *, context):
    _bound_context(state, refunded, context)
    failures = collect_failures(state, refunded)
    _check_denial(decision, state, failures)
    clauses = _clauses(failures, context)
    changed = joint_counterfactual(state, failures, context=context)
    reasons = tuple(f.reason for f in failures)
    proposed = decision.outcome == c.Outcome.REFER
    return DenialExplanation(
        request_id=decision.request_id,
        proposed=proposed,
        reasons=reasons,
        clauses=clauses,
        evidence=_evidence(failures, context.current),
        counterfactual=changed,
        limits=tuple(LIMITS[r] for r in reasons),
        next_step=context.review_route,
        rendered=_render(reasons, clauses, changed, context.review_route, proposed),
    )
