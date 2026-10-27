"""Six-check fidelity includes eligibility, time, approvals and complete evidence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.audit.audits_explanations_explain import (
    counterfactual,
    explain,
    failing_clauses,
    replay,
)
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ExplanationContext,
    ReviewRoute,
)
from refund_agent.data.generate import Dataset
from refund_agent.memory.register import PolicyRegister, clause_text
from refund_agent.rules import draft
from refund_agent.tools import Tools

AT = datetime(2026, 5, 1, tzinfo=timezone.utc)


def context(
    *,
    amount=5200,
    paid=4000,
    refunded=0,
    reason="damaged",
    days=31,
    submitted=29,
    delivery=True,
):
    policy = c.Policy(
        version="2026-05-01",
        effective_at=AT,
        return_days=30,
        cap_cents=100000,
        text=f"refund_cap_v4@2026-05-01: physical goods. "
        f"#eligibility: {clause_text('eligibility', 30)} "
        f"#cap: {clause_text('cap', 100000)}",
    )
    request = c.Request(
        request_id="audits-explanations-two-clause",
        requester_id="audits-explanations-customer",
        order_reference="audit-o",
        at=AT + timedelta(days=submitted),
        amount_cents=amount,
        reason=reason,
        text="Model text never belongs in an explanation.",
    )
    order = c.Order(
        order_id="audit-o",
        reference="audit-o",
        purchased_at=AT - timedelta(days=5),
        observed_at=AT + timedelta(days=days, seconds=-120),
        items=[c.Item(name="lamp", quantity=1, unit_cents=paid)],
        paid_cents=paid,
        payment_id="audit-payment",
    )
    events = [c.ShipmentEvent(at=AT - timedelta(days=4), status="shipped")]
    if delivery:
        events.append(c.ShipmentEvent(at=AT, status="delivered"))
    shipment = c.Shipment(
        order_id=order.order_id, observed_at=order.observed_at, events=events
    )
    current = EvaluationRecord(
        request=request,
        order=order,
        shipment=shipment,
        policy=policy,
        policy_observed_at=AT + timedelta(days=days, hours=-1),
        at=AT + timedelta(days=days),
        refunded_cents=refunded,
    )
    historical = EvaluationRecord.model_validate(
        {
            **current.model_dump(),
            "at": AT + timedelta(days=30),
            "order": order.model_copy(
                update={"observed_at": AT + timedelta(days=30, seconds=-120)}
            ),
            "shipment": shipment.model_copy(
                update={"observed_at": AT + timedelta(days=30, seconds=-120)}
            ),
            "policy_observed_at": AT + timedelta(days=30, hours=-1),
        }
    )
    register = PolicyRegister()
    register.announce(policy, AT)
    return ExplanationContext(
        current,
        register.snapshot(AT + timedelta(days=60)),
        (historical,),
        lambda subject, order_id: subject == request.requester_id
        and order_id == order.order_id,
        ReviewRoute(
            request_id=request.request_id,
            ticket_id="audit-review",
            deadline=AT + timedelta(days=33),
        ),
    )


def generate(ctx):
    state = ctx.current.state()
    return explain(
        state, draft(state), state.refunded_cents, context=ctx
    ), state


def test_two_clause_case_clears_every_failure_and_preserves_approval_b1_b2():
    ctx = context()
    explanation, state = generate(ctx)
    assert explanation.reasons == (
        "amount_over_policy",
        "outside_return_window",
    )
    assert all(replay(explanation, state, context=ctx).values())
    cf = explanation.counterfactual
    assert cf.outcome == "eligible" and cf.approval_required == "reviewer"
    assert cf.approval_granted is False
    assert len(cf.changes) == 2 and cf.evaluation.request.amount_cents == 4000
    assert cf.evaluation.at == AT + timedelta(days=30)
    amount_only = state.model_copy(deep=True)
    amount_only.request = amount_only.request.model_copy(
        update={"amount_cents": 4000}
    )
    assert draft(amount_only).reason == "outside_return_window"


def test_request_clock_alone_cannot_clear_day_31_execution_b2():
    ctx = context(amount=2499, paid=9000, submitted=29)
    explanation, state = generate(ctx)
    moved = state.model_copy(deep=True)
    moved.request = moved.request.model_copy(
        update={"at": AT + timedelta(days=28)}
    )
    assert draft(moved).reason == "outside_return_window"
    assert explanation.counterfactual.evaluation.at == AT + timedelta(days=30)
    assert all(replay(explanation, state, context=ctx).values())


@pytest.mark.parametrize(
    "amount,role", [(5200, "reviewer"), (65000, "named_approver:morgan")]
)
def test_unapproved_policy_eligibility_never_asserts_execution(amount, role):
    ctx = context(
        amount=amount,
        paid=100000,
        days=10,
        submitted=9,
        reason="not_delivered",
    )
    explanation, state = generate(ctx)
    assert explanation.counterfactual.approval_required == role
    assert not explanation.counterfactual.approval_granted
    eligible = draft(explanation.counterfactual.evaluation.state())
    assert eligible.reason == "eligible" and not eligible.executed
    assert state.approvals == []
    assert all(replay(explanation, state, context=ctx).values())
    # Prove eligibility is still unpaid through the actual loop and core gate.
    evaluated = explanation.counterfactual.evaluation
    data = Dataset(
        requests=[evaluated.request],
        orders=[evaluated.order],
        shipments=[evaluated.shipment],
        policies=[evaluated.policy],
        correct=[],
        retrievals=[
            c.PolicySnapshot(
                request_id=evaluated.request.request_id,
                version=evaluated.policy.version,
                observed_at=evaluated.policy_observed_at,
            )
        ],
    )
    tools = Tools(data)
    trace = Agent(tools).run(evaluated.request, at=evaluated.at)
    assert trace.decision.outcome == c.Outcome.REFER and not tools.payments
    assert any(
        r.required_role == ("named_approver" if amount > 50000 else "reviewer")
        for r in tools.reviews.values()
    )


def test_exhausted_balance_has_no_admissible_positive_amount():
    ctx = context(amount=2499, paid=4000, refunded=4000, days=10, submitted=9)
    explanation, state = generate(ctx)
    assert explanation.counterfactual.outcome == "no_admissible_change"
    assert explanation.counterfactual.evaluation is None
    assert "person" in explanation.rendered
    assert all(replay(explanation, state, context=ctx).values())


def test_missing_historical_evidence_is_refused_without_retiming_current_sources():
    ctx = replace(context(), history=())
    with pytest.raises(ValueError, match="historical evidence"):
        generate(ctx)


def test_wrong_version_and_other_order_scan_fail_replay():
    ctx = context(reason="not_delivered", amount=2499, days=10, submitted=9)
    explanation, state = generate(ctx)
    wrong = explanation.model_copy(
        update={
            "clauses": (
                explanation.clauses[0].model_copy(
                    update={"version": "2026-07-01"}
                ),
            )
        }
    )
    assert replay(wrong, state, context=ctx)["version"] is False
    evidence = list(explanation.evidence)
    evidence[0] = evidence[0].model_copy(
        update={"order_id": "another-customer-order"}
    )
    wrong = explanation.model_copy(update={"evidence": tuple(evidence)})
    assert replay(wrong, state, context=ctx)["disclosure"] is False
    assert (
        replay(
            explanation, state, context=replace(ctx, may_act=lambda *_: False)
        )["disclosure"]
        is False
    )


def test_rendered_payload_cannot_smuggle_text_or_other_customer_identifier():
    ctx = context()
    explanation, state = generate(ctx)
    for injected in (state.request.text, "risk_score=0.9", "customer-7-9999"):
        wrong = explanation.model_copy(
            update={"rendered": explanation.rendered + injected}
        )
        assert not replay(wrong, state, context=ctx)["disclosure"]
    assert (
        ";" not in explanation.rendered
        and state.request.text not in explanation.rendered
    )


def test_first_clause_only_fails_completeness_and_counterfactual():
    ctx = context()
    explanation, state = generate(ctx)
    failures = failing_clauses(state, state.refunded_cents)
    cf = counterfactual(state, failures[:1], context=ctx)
    wrong = explanation.model_copy(
        update={
            "reasons": explanation.reasons[:1],
            "clauses": explanation.clauses[:1],
            "counterfactual": cf,
        }
    )
    result = replay(wrong, state, context=ctx)
    assert (
        result["completeness"] is False and result["counterfactual"] is False
    )


def test_denial_sources_include_carrier_observation_age_and_person_route():
    ctx = context(reason="not_delivered", amount=2499, days=10, submitted=9)
    explanation, state = generate(ctx)
    delivered = next(
        e for e in explanation.evidence if e.field == "shipment.deliveries"
    )
    assert (
        delivered.age_seconds == 120
        and delivered.observed_at == ctx.current.shipment.observed_at
    )
    assert explanation.next_step.ticket_id == ctx.review_route.ticket_id
    assert explanation.next_step.deadline == ctx.review_route.deadline
    assert "does not show who received" in explanation.rendered


def test_payment_non_denial_referral_and_stale_clause_version_refused():
    ctx = context()
    state = ctx.current.state()
    for outcome, reason in (
        (c.Outcome.APPROVE, "eligible"),
        (c.Outcome.REFER, "missing_evidence"),
    ):
        with pytest.raises(ValueError, match="denial"):
            explain(
                state,
                draft(state).model_copy(
                    update={"outcome": outcome, "reason": reason}
                ),
                state.refunded_cents,
                context=ctx,
            )
    stale = draft(state).model_copy(
        update={"policy_citation": "refund_cap_v4@2026-07-01#cap"}
    )
    with pytest.raises(ValueError, match="version"):
        explain(state, stale, state.refunded_cents, context=ctx)


def test_proposed_named_approver_denial_remains_proposed():
    ctx = context(amount=65000, paid=4000, days=10, submitted=9)
    state = ctx.current.state()
    proposed = draft(state).model_copy(update={"outcome": c.Outcome.REFER})
    explanation = explain(state, proposed, 0, context=ctx)
    assert explanation.proposed
    assert "proposes to deny" in explanation.rendered
    assert "Your request was denied" not in explanation.rendered
    assert (
        explanation.counterfactual.approval_required == "named_approver:morgan"
    )
    assert all(replay(explanation, state, context=ctx).values())


def test_clock_move_uses_historical_policy_and_evidence_across_a_version_change():
    ctx = context(amount=2499)
    old = ctx.current.policy
    effective = AT + timedelta(days=25)
    new = c.Policy(
        version=effective.date().isoformat(),
        effective_at=effective,
        return_days=20,
        cap_cents=100000,
        text=f"refund_cap_v4@{effective.date().isoformat()}: physical goods. "
        f"#eligibility: {clause_text('eligibility', 20)} "
        f"#cap: {clause_text('cap', 100000)}",
    )
    register = PolicyRegister()
    register.announce(old, AT)
    register.announce(new, effective)
    boundary = effective - timedelta(microseconds=1)
    row = ctx.history[0]
    historical = EvaluationRecord.model_validate(
        {
            **row.model_dump(),
            "at": boundary,
            "order": row.order.model_copy(
                update={"observed_at": boundary - timedelta(seconds=120)}
            ),
            "shipment": row.shipment.model_copy(
                update={"observed_at": boundary - timedelta(seconds=120)}
            ),
            "policy_observed_at": boundary - timedelta(hours=1),
        }
    )
    ctx = replace(
        ctx,
        current=ctx.current.model_copy(update={"policy": new}),
        history=(historical,),
        register=register.snapshot(ctx.current.at),
    )
    e, state = generate(ctx)
    assert e.clauses[0].version == new.version
    assert e.counterfactual.evaluation.policy.version == old.version
    assert e.counterfactual.evaluation.at == boundary
    assert e.counterfactual.evaluation.shipment.observed_at < boundary
    assert all(replay(e, state, context=ctx).values())
