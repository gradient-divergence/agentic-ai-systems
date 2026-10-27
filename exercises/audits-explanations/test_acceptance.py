"""Exercise references pass, starters fail for their stated behavioral reasons."""

import json
import os
import runpy
from datetime import timedelta
from pathlib import Path

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite(trials=1))

# isort: split
from demos.audits_explanations_examples import AT, context_from_records  # noqa: E402
from refund_agent import contracts as c  # noqa: E402
from refund_agent.audit.audits_explanations_explain import replay  # noqa: E402
from refund_agent.audit.audits_explanations_rates import (  # noqa: E402
    AuditObservation,
    AuditRule,
    Period,
)
from refund_agent.audit.audits_explanations_records import (  # noqa: E402
    EvaluationRecord,
    ExplanationContext,
    ReviewRoute,
)
from refund_agent.memory.register import PolicyRegister, clause_text  # noqa: E402
from refund_agent.rules import draft  # noqa: E402

HERE = Path(__file__).resolve().parent


def fixture():
    records = json.loads((HERE / "two_clause_06.json").read_text())
    subject = records["current"]["request"]["requester_id"]
    order = records["current"]["order"]["order_id"]
    return context_from_records(
        records, may_act=lambda s, o: s == subject and o == order
    )


def solution():
    name = os.environ.get("AUDITS_EXPLANATIONS_SOLUTION", "explain_starter")
    if name not in {"reference", "explain_starter"}:
        raise ValueError("choose reference or explain_starter")
    return runpy.run_path(str(HERE / f"{name}.py"))["explain"]


def check_solution(explain, ctx, *, state=None, decision=None):
    state = state or ctx.current.state()
    e = explain(state, decision or draft(state), state.refunded_cents, context=ctx)
    checks = replay(e, state, context=ctx)
    assert checks["completeness"], (
        "the explanation omits a failing clause or required evidence"
    )
    assert checks["counterfactual"], (
        "the counterfactual still fails eligibility"
    )
    assert all(checks.values())
    return e


def test_solution_completes_both_clauses_and_eligibility():
    check_solution(solution(), fixture())


def constructed_case(kind):
    """Independent inputs, with values distinct from two_clause_06.json."""
    days = 17 if kind == "window" else 5
    at = AT + timedelta(days=days)
    amount = 3400 if kind == "cap" else 700
    policy = c.Policy(
        version="2026-05-01",
        effective_at=AT,
        return_days=14,
        cap_cents=2500,
        text="refund_cap_v4@2026-05-01: physical goods. "
        f"#eligibility: {clause_text('eligibility', 14)} "
        f"#cap: {clause_text('cap', 2500)}",
    )
    request = c.Request(
        request_id=f"audits-explanations-independent-{kind}",
        requester_id=f"independent-customer-{kind}",
        order_reference=f"independent-order-{kind}",
        at=AT + timedelta(days=13) if kind == "window" else at,
        amount_cents=amount,
        reason="not_delivered" if kind == "delivered" else "damaged",
    )
    order = c.Order(
        order_id=request.order_reference,
        reference=request.order_reference,
        purchased_at=AT - timedelta(days=2),
        observed_at=at - timedelta(seconds=73),
        items=[c.Item(name="physical lamp", quantity=1, unit_cents=2800)],
        paid_cents=2800,
        payment_id=f"independent-payment-{kind}",
    )
    shipment = c.Shipment(
        order_id=order.order_id,
        observed_at=at - timedelta(seconds=73),
        events=[
            c.ShipmentEvent(at=AT - timedelta(days=1), status="shipped"),
            c.ShipmentEvent(at=AT, status="delivered"),
        ],
    )
    current = EvaluationRecord(
        request=request,
        order=order,
        shipment=shipment,
        policy=policy,
        policy_observed_at=at - timedelta(minutes=41),
        at=at,
        refunded_cents=2800 if kind == "balance" else 0,
    )
    history = ()
    if kind == "window":
        end = AT + timedelta(days=14)
        history = (
            EvaluationRecord(
                request=request,
                order=order.model_copy(update={"observed_at": end - timedelta(seconds=73)}),
                shipment=shipment.model_copy(update={"observed_at": end - timedelta(seconds=73)}),
                policy=policy,
                policy_observed_at=end - timedelta(minutes=41),
                at=end,
                refunded_cents=0,
            ),
        )
    register = PolicyRegister()
    register.announce(policy, AT)
    return ExplanationContext(
        current,
        register.snapshot(at),
        history,
        lambda s, o: s == request.requester_id and o == order.order_id,
        ReviewRoute(
            request_id=request.request_id,
            ticket_id=f"independent-review-{kind}",
            deadline=at + timedelta(days=1),
        ),
    )


@pytest.mark.parametrize("kind", ("delivered", "window", "cap", "balance"))
def test_solution_handles_independent_denial_reasons(kind):
    ctx = constructed_case(kind)
    e = check_solution(solution(), ctx)
    expected = {
        "delivered": "already_delivered",
        "window": "outside_return_window",
        "cap": "amount_over_policy",
        "balance": "amount_over_policy",
    }
    assert e.reasons == (expected[kind],)
    if kind in {"delivered", "window"}:
        scan = next(i for i in e.evidence if i.field == "shipment.deliveries")
        assert json.loads(scan.value) == [AT.isoformat()]
        assert scan.observed_at == ctx.current.shipment.observed_at
        assert scan.age_seconds == 73
    if kind == "window":
        assert e.counterfactual.evaluation.at == AT + timedelta(days=14)
        assert e.counterfactual.evaluation.request.at == ctx.current.request.at
    if kind == "balance":
        assert e.counterfactual.outcome == "no_admissible_change"
        assert e.counterfactual.evaluation is None
        assert e.counterfactual.changes == ()


def test_fixture_only_solution_is_rejected_by_independent_case():
    reference = runpy.run_path(str(HERE / "reference.py"))["explain"]

    def fixture_only(state, decision, refunded, *, context):
        e = reference(state, decision, refunded, context=context)
        return (
            e
            if state.request.request_id == fixture().current.request.request_id
            else e.model_copy(update={"evidence": ()})
        )

    check_solution(fixture_only, fixture())
    with pytest.raises(AssertionError, match="failing clause or required evidence"):
        check_solution(fixture_only, constructed_case("delivered"))


def test_solution_replays_every_canonical_seed7_denial():
    from demos.audits_explanations_integrated_demo import RUN_PATH, _read_run

    if not RUN_PATH.is_file():
        pytest.skip("canonical seed-7 run absent; chapters 33-35 integration pending")
    record = _read_run(RUN_PATH)
    register = PolicyRegister()
    for policy in record.month.data.policies:
        register.announce(policy, policy.effective_at)
    snapshot = register.snapshot(record.audit.at)
    attachments = {a.request_id: a for a in record.run.explanations}
    denials = [t for t in record.run.traces if t.decision.outcome == c.Outcome.DENY]
    assert denials, "canonical seed-7 run has no denials to replay"
    explain = solution()
    for trace in denials:
        rid = trace.decision.request_id
        attachment = attachments[rid]
        current = attachment.evaluation
        assert current is not None, "the rule's retained source snapshot is required"
        ctx = ExplanationContext(
            current,
            snapshot,
            record.month.history.get(rid, ()),
            lambda s, o: s == current.request.requester_id and o == current.order.order_id,
            attachment.explanation.next_step,
        )
        check_solution(explain, ctx, state=trace.steps[-1].state, decision=trace.decision)


def test_starter_has_the_two_designed_failures():
    solution = runpy.run_path(str(HERE / "explain_starter.py"))["explain"]
    ctx = fixture()
    state = ctx.current.state()
    e = solution(state, draft(state), state.refunded_cents, context=ctx)
    checks = replay(e, state, context=ctx)
    assert checks["denial"] and checks["evidence"] and checks["disclosure"]
    assert (
        checks["completeness"] is False and checks["counterfactual"] is False
    )


def test_blind_reading_reference_rejects_appeals_substitution():
    # Constructed test records only. The canonical seed-11 month awaits regions.
    observations = tuple(
        AuditObservation(
            request_id=f"audits-explanations-ex5-{g}-{i}",
            order_id=f"audit-o-{g}-{i}",
            segment=g,
            at=AT + timedelta(days=1),
            outcome="deny" if i < 50 else "approve",
            reason="already_delivered" if i < 50 else "eligible",
            owed=(g == "C" and i < 5) if i < 50 else True,
            found_false_if_paid=None if i < 50 else False,
            label_mature_at=AT + timedelta(days=32),
        ).model_dump(mode="json")
        for g in ("A", "B", "C")
        for i in range(150)
    )
    report = {
        "seed": 11,
        "observations": observations,
        "rule": AuditRule(signed_at=AT - timedelta(days=1)).model_dump(
            mode="json"
        ),
        "period": Period(start=AT, end=AT + timedelta(days=28)).model_dump(
            mode="json"
        ),
        "at": (AT + timedelta(days=33)).isoformat(),
        "appeals": {"A": 1, "B": 5, "C": 3},
        "denials": {"A": 50, "B": 50, "C": 50},
    }
    ref = runpy.run_path(str(HERE / "reference.py"))["analyze_report"](report)
    starter = runpy.run_path(str(HERE / "audit_starter.py"))["analyze_report"](
        report
    )
    assert ref["holds"] == {"C": ["already_delivered"]}
    assert starter["holds"] == {"B": ["already_delivered"]}



def test_starter_constructs_its_worked_example_without_complete_generators(monkeypatch):
    from refund_agent.audit import audits_explanations_explain as complete

    def forbidden(*args, **kwargs):
        raise AssertionError("the starter must construct clauses and changes locally")

    for name in ("explain", "failing_clauses", "counterfactual"):
        monkeypatch.setattr(complete, name, forbidden)
    explain = runpy.run_path(str(HERE / "explain_starter.py"))["explain"]
    ctx = fixture()
    state = ctx.current.state()
    result = explain(state, draft(state), state.refunded_cents, context=ctx)
    assert result.reasons == ("amount_over_policy",)
    assert result.counterfactual.outcome == "still_ineligible"
    assert not result.counterfactual.approval_granted
