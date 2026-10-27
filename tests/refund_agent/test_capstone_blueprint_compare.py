"""Hand-counted three-arm accounting and real rules-client runs."""

import pytest
from capstone_blueprint_fixtures import AT, ROLES

from refund_agent.agent import Agent
from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
from refund_agent.launch.capstone_blueprint_compare import (
    ArmDecision,
    ComparisonInputs,
    LabelledRequest,
    MeasuredArm,
    ReviewWork,
    measure,
)
from refund_agent.launch.capstone_blueprint_requirement import (
    refund_requirement,
)
from refund_agent.launch.capstone_blueprint_rules import RulesClient
from refund_agent.tools import Tools


def comparison_fixture():
    req = refund_requirement("B6")
    requests = tuple(
        LabelledRequest(
            request_id=f"r{i}",
            tier=1 if i < 14 else 2 if i < 18 else 3,
            amount_cents=4999 if i < 14 else 5000 if i < 18 else 50001,
            requested_day=50 + i / 24,
            promised_hours=1,
            person_outcome="deny" if i == 0 else "approve",
            person_paid=i != 0,
            found_false_30d=i == 1,
            label_day=50 + i / 24 + 30,
        )
        for i in range(20)
    )
    choices = tuple(
        ArmDecision(
            request_id=r.request_id,
            outcome=r.person_outcome,
            answer_day=r.requested_day + 1 / 48,
            paid_cents=r.amount_cents if r.person_paid else 0,
            timeliness_good=True,
        )
        for r in requests
    )
    reviews = []
    for kind, count, minutes in (
        ("tier2", 4, 15),
        ("tier3", 2, 30),
        ("referral", 2, 5),
        ("recheck", 2, 5),
        ("blind", 1, 15),
        ("planted", 1, 15),
    ):
        reviews.extend(
            ReviewWork(
                review_id=f"{kind}-{i}",
                request_id=None,
                work_class=kind,
                minutes=minutes,
            )
            for i in range(count)
        )
    parts = {
        kind: "same-" + kind
        for kind in ("model", "prompt", "tools", "policy", "index")
    }
    a = MeasuredArm(
        name="agent",
        bundle="B6",
        components=parts,
        model_identity="ScriptedModel",
        decisions=choices,
        reviews=tuple(reviews),
    )
    c = MeasuredArm(
        name="rules",
        bundle="rules-B6",
        components={**parts, "model": "rules-model"},
        model_identity="RulesClient",
        decisions=choices,
        reviews=tuple(reviews),
    )
    return req, ComparisonInputs(
        period=req.comparison_period,
        generated_labels=True,
        requests=requests,
        agent=a,
        rules=c,
        manual_answer_days={
            r.request_id: r.requested_day + 1 for r in requests
        },
        manual_queue_record="constructed-queue-result",
    )


def test_twenty_request_accounting_and_no_fixed_rules_line():
    req, inputs = comparison_fixture()
    result = measure(inputs, req, ROLES)
    a, manual, rules = result.arms
    assert a.final_decisions == manual.final_decisions == rules.final_decisions
    assert a.review_minutes == {
        "tier2": 60,
        "tier3": 60,
        "referral": 10,
        "recheck": 10,
        "blind": 15,
        "planted": 15,
        "modification": 0,
        "refusal_followup": 0,
        "held_denial": 0,
        "adjudication": 0,
    }
    assert a.hours_per_week == pytest.approx(170 / 120)
    assert manual.hours_per_week == pytest.approx(190 / 120)
    assert (
        a.cost_cents == rules.cost_cents == 4800 and manual.cost_cents == 8000
    )
    assert (
        a.wrongful_denials
        == rules.wrongful_denials
        == manual.wrongful_denials
        == 0
    )
    assert a.wrong_payment_cents == manual.wrong_payment_cents == 0
    assert (
        a.payments_later_false
        == rules.payments_later_false
        == manual.payments_later_false
        == 1
    )
    assert (a.tier1_answered_in_time, manual.tier1_answered_in_time) == (14, 0)
    assert (
        result.against_manual == "better"
        and result.against_rules == "not better"
    )
    assert (
        result.verdict == "not better"
        and result.manual_reviews_reviewers_work is False
    )
    # Measurement is not inferred from ScriptedModel: an extra measured recheck changes hours.
    extra = ReviewWork(
        review_id="extra-recheck",
        request_id="r2",
        work_class="recheck",
        minutes=5,
    )
    altered = inputs.model_copy(
        update={
            "rules": inputs.rules.model_copy(
                update={"reviews": inputs.rules.reviews + (extra,)}
            )
        }
    )
    assert (
        measure(altered, req, ROLES).arms[2].hours_per_week
        > rules.hours_per_week
    )


def test_a_fast_system_referral_is_bad_and_policy_referral_uses_its_classification():
    req, inputs = comparison_fixture()
    choices = tuple(
        row.model_copy(update={"timeliness_good": False})
        if row.request_id == "r2"
        else row
        for row in inputs.agent.decisions
    )
    changed = inputs.model_copy(
        update={
            "agent": inputs.agent.model_copy(update={"decisions": choices})
        }
    )
    assert measure(changed, req, ROLES).arms[0].tier1_answered_in_time == 13
    # Chapter 34 may classify an early policy referral as good before the person
    # finishes: the final answer clock must not replace that objective record.
    choices = tuple(
        row.model_copy(update={"answer_day": row.answer_day + 1})
        if row.request_id == "r2"
        else row
        for row in inputs.agent.decisions
    )
    changed = inputs.model_copy(
        update={
            "agent": inputs.agent.model_copy(update={"decisions": choices})
        }
    )
    assert measure(changed, req, ROLES).arms[0].tier1_answered_in_time == 14


def test_better_model_can_clear_the_comparison_without_changing_the_rule():
    req, inputs = comparison_fixture()
    choices = tuple(
        r.model_copy(update={"outcome": "deny", "paid_cents": 0})
        if r.request_id == "r2"
        else r
        for r in inputs.rules.decisions
    )
    changed = inputs.model_copy(
        update={
            "agent": inputs.agent.model_copy(
                update={"model_identity": "constructed-live-model"}
            ),
            "rules": inputs.rules.model_copy(update={"decisions": choices}),
        }
    )
    assert measure(changed, req, ROLES).against_rules == "better"


def test_scripted_mismatch_is_a_stop_rule():
    req, inputs = comparison_fixture()
    choices = tuple(
        r.model_copy(update={"outcome": "deny", "paid_cents": 0})
        if r.request_id == "r2"
        else r
        for r in inputs.rules.decisions
    )
    changed = inputs.model_copy(
        update={
            "rules": inputs.rules.model_copy(update={"decisions": choices})
        }
    )
    with pytest.raises(ValueError, match="stop the code job"):
        measure(changed, req, ROLES)


@pytest.mark.parametrize("arm", ["agent", "rules", "manual"])
@pytest.mark.parametrize("outcome", ["approve", "deny"])
def test_future_decisions_and_payments_cannot_count(arm, outcome):
    req, inputs = comparison_fixture()
    assert req.gate_day == 64
    request_id = "r0" if outcome == "deny" else "r2"
    if arm == "manual":
        inputs = inputs.model_copy(
            update={
                "manual_answer_days": {
                    **inputs.manual_answer_days,
                    request_id: 74,
                },
            }
        )
    else:
        measured = getattr(inputs, arm)
        decisions = tuple(
            row.model_copy(
                update={
                    "answer_day": 74,
                    "outcome": outcome,
                    "paid_cents": 0 if outcome == "deny" else row.paid_cents,
                }
            )
            if row.request_id == request_id
            else row
            for row in measured.decisions
        )
        inputs = inputs.model_copy(
            update={
                arm: measured.model_copy(update={"decisions": decisions}),
            }
        )
    with pytest.raises(ValueError, match="after evaluation clock"):
        measure(inputs, req, ROLES)


@pytest.mark.parametrize("fault", ["tools", "requests", "manual", "labels"])
def test_unfair_or_incomplete_comparison_is_refused(fault):
    req, inputs = comparison_fixture()
    if fault == "tools":
        inputs = inputs.model_copy(
            update={
                "rules": inputs.rules.model_copy(
                    update={
                        "components": {
                            **inputs.rules.components,
                            "tools": "different",
                        }
                    }
                )
            }
        )
    elif fault == "requests":
        inputs = inputs.model_copy(
            update={
                "agent": inputs.agent.model_copy(
                    update={"decisions": inputs.agent.decisions[:-1]}
                )
            }
        )
    elif fault == "manual":
        inputs = inputs.model_copy(update={"manual_answer_days": {}})
    else:
        inputs = inputs.model_copy(update={"generated_labels": False})
    with pytest.raises(ValueError):
        measure(inputs, req, ROLES)


def test_registered_rules_client_runs_inside_candidate_scope():
    from refund_agent.launch.capstone_blueprint_rules import SELECTOR
    from refund_agent.release.candidate import (
        build_candidate,
        build_tools,
        run_request,
    )
    from refund_agent.release.suite import build_suite

    suite = build_suite(components=(SELECTOR,))
    candidate = build_candidate(suite)
    component = candidate.tools_composition["components"][0]
    assert component["name"] == "capstone-blueprint.rules"
    assert len(component["fingerprint"]) == 64
    data, request = fixture("capstone-blueprint", "selected-rules", AT)
    trace = run_request(
        candidate, build_tools(data, candidate, "execute"), request
    )
    assert trace.decision.executed and trace.receipt.amount_cents == 4999


def test_rules_client_preserves_the_composed_risk_component():
    from refund_agent import contracts as c

    class RiskTools(Tools):
        def assess(self, state):
            chosen = super().assess(state)
            return chosen.model_copy(
                update={
                    "outcome": c.Outcome.REFER,
                    "reason": "constructed_risk_referral",
                }
            )

    data, request = fixture("capstone-blueprint", "risk", AT)
    trace = Agent(Tools(data)).run(request)
    state = next(s.state for s in trace.steps if s.phase == c.Phase.ASSESS)
    state = state.model_copy(update={"phase": c.Phase.ASSESS})
    reply = RulesClient(RiskTools(data)).reply(state)
    assert reply.outcome == c.Outcome.REFER
    assert reply.reason == "constructed_risk_referral"


@pytest.mark.parametrize("amount", [4999, 5000, 50001])
def test_real_rules_client_keeps_core_gate_and_named_approver(amount):
    data, request = fixture(
        "capstone-blueprint", f"rules-{amount}", AT, amount=amount
    )
    agent = Agent(Tools(data)).run(request)
    tools = Tools(data)
    rules = Agent(tools, RulesClient(tools)).run(request)
    assert (
        agent.decision.outcome,
        agent.decision.reason,
        bool(agent.receipt),
    ) == (rules.decision.outcome, rules.decision.reason, bool(rules.receipt))
