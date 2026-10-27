"""Real Agent/build_tools path, with chapter-36 constructed partner records."""

from datetime import datetime, timedelta, timezone

import pytest

from refund_agent.release.candidate import (
    build_candidate,
    build_tools,
    run_request,
)
from refund_agent.release.suite import build_suite

# isort: split
# Load chapter 29's core before importing selected component helpers.
from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_component import (
    STORE_PATH,
    selector,
)
from refund_agent.audit.audits_explanations_holds import AuditHolds
from refund_agent.audit.audits_explanations_rates import (
    AuditObservation,
    AuditRule,
    Period,
    audit_period,
)
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ReviewRoute,
    SegmentAssignments,
)
from refund_agent.audit.audits_explanations_stage6 import AuditInputs
from refund_agent.batch_stream.component import selector as event_time_selector
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.data.generate import generate
from refund_agent.memory.register import PolicyRegister
from refund_agent.model import ScriptedModel
from refund_agent.release import extensions
from refund_agent.release.extensions import contributions

AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
RULE = AuditRule(signed_at=AT - timedelta(days=1))


def inputs(data, *, segment="A", visible=True, with_history=True, evaluation_at=None):
    register = PolicyRegister()
    for policy in data.policies:
        register.announce(policy, policy.effective_at)
    histories = {}
    if with_history:
        for request in data.requests:
            order = next(
                (
                    o
                    for o in data.orders
                    if o.order_id == request.order_reference
                ),
                None,
            )
            if order is None:
                continue
            shipment = next(
                s for s in data.shipments if s.order_id == order.order_id
            )
            delivered = [
                e.at for e in shipment.events if e.status == "delivered"
            ]
            if not delivered:
                continue
            dates = sorted(p.effective_at for p in data.policies)
            for policy in data.policies:
                at = max(delivered) + timedelta(days=policy.return_days)
                ends = [d for d in dates if d > policy.effective_at]
                if ends:
                    at = min(at, min(ends) - timedelta(microseconds=1))
                if (
                    policy.effective_at <= at <= (evaluation_at or request.at)
                    and data.policy_at(at) == policy
                ):
                    historical = EvaluationRecord(
                        request=request,
                        order=order.model_copy(
                            update={"observed_at": at - timedelta(seconds=120)}
                        ),
                        shipment=shipment.model_copy(
                            update={"observed_at": at - timedelta(seconds=120)}
                        ),
                        policy=policy,
                        policy_observed_at=at - timedelta(hours=1),
                        at=at,
                        refunded_cents=sum(
                            p.amount_cents
                            for p in order.prior_refunds
                            if p.at <= at
                        ),
                    )
                    histories.setdefault(request.request_id, []).append(
                        historical
                    )
    return AuditInputs(
        RULE,
        SegmentAssignments(
            by_order={o.order_id: segment for o in data.orders},
            version="constructed-chapter36-test-records",
        ),
        register.snapshot(
            max(r.at for r in data.requests) + timedelta(days=40)
        ),
        {k: tuple(v) for k, v in histories.items()},
        {
            r.request_id: ReviewRoute(
                request_id=r.request_id,
                ticket_id=f"audit-review-{r.request_id}",
                deadline=r.at + timedelta(days=2),
            )
            for r in data.requests
        },
        lambda subject, order: visible
        and any(
            r.requester_id == subject and r.order_reference == order
            for r in data.requests
        ),
        AuditHolds(STORE_PATH),
    )


def candidate():
    return build_candidate(build_suite(1, components=(selector(RULE),)))


def test_public_denials_use_real_agent_and_each_carries_passing_replay():
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    selected = candidate()
    tools = build_tools(
        data, selected, "execute", audits_explanations=inputs(data)
    )
    traces = [run_request(selected, tools, r) for r in data.requests]
    denials = [t for t in traces if t.decision.outcome == c.Outcome.DENY]
    assert len(denials) == 8
    for trace in denials:
        attached = tools.explanations[trace.decision.request_id]
        assert attached.explanation is not None and all(
            attached.checks.values()
        )
        assert (
            tools.audits[trace.steps[-1].result.record_id].decision
            == trace.decision
        )
    assert tools.data.correct == []


def test_missing_partner_inputs_close_the_composition():
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    with pytest.raises(ValueError, match="inputs required"):
        build_tools(data, candidate(), "execute")


def test_disclosure_replay_failure_is_referred_and_logged_before_effect():
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    selected = candidate()
    tools = build_tools(
        data,
        selected,
        "execute",
        audits_explanations=inputs(data, visible=False),
    )
    request = next(r for r in data.requests if r.request_id == "r-7-0014")
    trace = run_request(selected, tools, request)
    assert (
        trace.decision.outcome == c.Outcome.REFER
        and trace.decision.reason == "explanation_failed"
    )
    attached = tools.explanations[request.request_id]
    assert (
        attached.failure == "replay_failed"
        and not attached.checks["disclosure"]
    )
    assert trace.failure_detail == "audits-explanations:replay_failed"
    assert not tools.payments


def test_hold_refers_only_its_segment_and_reason_with_proposed_explanation():
    rows = tuple(
        AuditObservation(
            request_id=f"audits-explanations-hold-{i}",
            order_id=f"hold-o-{i}",
            segment="C",
            at=AT + timedelta(days=1),
            outcome="deny" if i < 8 else "approve",
            reason="already_delivered" if i < 8 else "eligible",
            owed=True,
            found_false_if_paid=None if i < 8 else False,
            label_mature_at=AT + timedelta(days=32),
        )
        for i in range(20)
    )
    AuditHolds(STORE_PATH).apply(
        audit_period(
            rows,
            Period(start=AT, end=AT + timedelta(days=28)),
            RULE,
            7,
            at=AT + timedelta(days=33),
        )
    )
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    request = next(r for r in data.requests if r.request_id == "r-7-0029")
    selected = candidate()
    for segment, outcome in (("C", c.Outcome.REFER), ("A", c.Outcome.DENY)):
        tools = build_tools(
            data,
            selected,
            "execute",
            audits_explanations=inputs(data, segment=segment),
        )
        trace = run_request(selected, tools, request)
        assert (
            trace.decision.outcome == outcome
            and trace.decision.reason == "already_delivered"
        )
        assert tools.explanations[request.request_id].explanation.proposed == (
            segment == "C"
        )
        assert all(tools.explanations[request.request_id].checks.values())


def test_declared_identity_store_and_partner_contributions():
    selected = candidate()
    record = selected.tools_composition["components"][0]
    assert record["version"] == "audits-explanations-stage6-1"
    assert any("audits_explanations_explain" in m for m in record["modules"])
    assert record["stores"][0]["mode"] == "append"
    assert record["stores"][0]["owner"] == "morgan"
    bundle = contributions(selected, "bundle_items")[record["name"]]
    assert (
        bundle["tools"]["audits-explanations.generator"]
        == "audits-explanations-explain-1"
    )
    requirements = contributions(selected, "promotion_requirements")[
        record["name"]
    ]
    assert requirements[0]["must_pass"]
    assert contributions(selected, "reasons")[record["name"]] == {
        "explanation_failed": "referral"
    }
    default = build_candidate(build_suite(1))
    assert "components" not in default.tools_composition


def test_month_runner_uses_real_tools_with_truth_stripped_and_mature_audit():
    from refund_agent.audit.audits_explanations_month import (
        constructed_month,
        run_month,
    )

    segments = SegmentAssignments(
        by_order={f"audits-explanations-o-{i:04d}": "A" for i in range(18)},
        version="constructed-chapter36-test-records",
    )
    month = constructed_month(segments, n=18)
    selected = candidate()

    def context_for(data, request):
        assert data.correct == [] and not hasattr(data, "truth")
        return {"audits_explanations": inputs(data)}

    run = run_month(month, selected, context_for)
    assert len(run.traces) == len(run.observations) == 18
    assert run.paid_cents == sum(
        t.receipt.amount_cents for t in run.traces if t.receipt
    )
    assert all(
        a.failure is None and all(a.checks.values()) for a in run.explanations
    )
    audit = audit_period(
        run.observations,
        month.period,
        RULE,
        7,
        at=month.period.end + timedelta(days=30),
    )
    assert audit.segments[0].status in {"assessed", "no_denials"}


def test_held_failed_explanation_retains_proposed_reason_in_month_audit():
    from refund_agent.audit.audits_explanations_month import (
        SETTINGS,
        Month,
        Truth,
        run_month,
    )

    prior = AuditObservation(
        request_id="audits-explanations-prior-hold",
        order_id="audits-explanations-prior-order",
        segment="C",
        at=AT + timedelta(days=1),
        outcome="deny",
        reason="already_delivered",
        owed=True,
        found_false_if_paid=None,
        label_mature_at=AT + timedelta(days=32),
    )
    AuditHolds(STORE_PATH).apply(
        audit_period(
            (prior,),
            Period(start=AT, end=AT + timedelta(days=28)),
            RULE,
            7,
            at=AT + timedelta(days=33),
        )
    )
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    request = next(r for r in data.requests if r.request_id == "r-7-0029")
    audit_inputs = inputs(data, segment="C", visible=False)
    month = Month(
        seed=7,
        period=Period(start=request.at, end=request.at + timedelta(days=30)),
        settings=SETTINGS,
        segments=audit_inputs.segments,
        data=data.model_copy(update={"requests": [request]}),
        truth=(
            Truth(
                request_id=request.request_id,
                received=False,
                genuine=True,
                owed=True,
                found_false_if_paid=False,
            ),
        ),
        history={},
        evaluation_times={request.request_id: request.at},
    )
    run = run_month(
        month, candidate(), lambda *_: {"audits_explanations": audit_inputs}
    )
    (trace,) = run.traces
    (attachment,) = run.explanations
    (observation,) = run.observations
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.reason == "explanation_failed"
    assert trace.failure_detail == "audits-explanations:replay_failed"
    assert attachment.failure == "replay_failed"
    assert attachment.checks["disclosure"] is False
    assert attachment.proposed_decision.reason == "already_delivered"
    assert observation.held_proposal and observation.outcome == "refer"
    assert observation.reason == attachment.proposed_decision.reason
    assert run.paid_cents == 0
    audit = audit_period(
        run.observations,
        month.period,
        RULE,
        7,
        at=month.period.end + timedelta(days=30),
    )
    segment = audit.segments[2]
    assert segment.denials == segment.held_proposals == segment.wrongful == 1
    assert segment.sample_ids == (request.request_id,)
    assert segment.breach and segment.held_reasons == ("already_delivered",)


def event_time_denial(*, later_scan=False, heartbeat=False):
    data = generate(7, orders=40, requests=40).model_copy(
        update={"correct": []}
    )
    request = next(r for r in data.requests if r.request_id == "r-7-0029")
    shipment = next(s for s in data.shipments if s.order_id == request.order_reference)
    observed = request.at - timedelta(seconds=60)
    scans = [
        ScanEvent(
            event_id=f"audits-explanations-scan-{i}",
            order_id=shipment.order_id,
            status=e.status,
            at=e.at,
            arrived_at=observed,
        )
        for i, e in enumerate(shipment.events)
    ]
    if later_scan:
        later = c.ShipmentEvent(status="delivered", at=request.at - timedelta(minutes=30))
        scans.append(
            ScanEvent(
                event_id="audits-explanations-later-scan",
                order_id=shipment.order_id,
                status="delivered",
                at=later.at,
                arrived_at=request.at + timedelta(seconds=60),
            )
        )
        shipment = shipment.model_copy(update={"events": [*shipment.events, later]})
        data = data.model_copy(
            update={
                "shipments": [
                    shipment if s.order_id == shipment.order_id else s for s in data.shipments
                ]
            }
        )
    selected = build_candidate(build_suite(1, components=(event_time_selector(), selector(RULE))))
    log = EventLog(scans)
    tools = build_tools(
        data,
        selected,
        "execute",
        event_log=log,
        audits_explanations=inputs(data, with_history=False),
    )

    class AssessmentModel(ScriptedModel):
        def reply(self, state):
            if heartbeat and state.phase == c.Phase.ASSESS:
                log.append(
                    ScanEvent(
                        event_id="audits-explanations-heartbeat",
                        order_id="audits-explanations-other-order",
                        status="shipped",
                        at=request.at - timedelta(minutes=2),
                        arrived_at=request.at - timedelta(seconds=30),
                    )
                )
            return super().reply(state)

    return request, run_request(selected, tools, request, model=AssessmentModel()), tools


def test_retained_snapshot_is_taken_at_rule_evaluation_not_the_prior_read():
    request, trace, tools = event_time_denial(heartbeat=True)
    shipment_read = next(s for s in trace.steps if s.phase == c.Phase.SHIPMENT)
    assert shipment_read.result.age_seconds == 60
    attachment = tools.explanations[request.request_id]
    assert attachment.evaluation.shipment.observed_at == request.at - timedelta(seconds=30)
    assert all(attachment.checks.values())


def test_composed_explanation_cites_the_shipment_age_the_rule_read():
    request, trace, tools = event_time_denial()
    assert trace.decision.reason == "already_delivered"
    assert trace.steps[-1].state.freshness["shipment"] == 60
    attachment = tools.explanations[request.request_id]
    evidence = next(e for e in attachment.explanation.evidence if e.field == "shipment.deliveries")
    assert evidence.age_seconds == 60
    assert evidence.observed_at == request.at - timedelta(seconds=60)
    assert all(attachment.checks.values())
    assert attachment.evaluation.shipment.observed_at == evidence.observed_at


def test_replay_independently_rejects_a_rebuilt_observation_time():
    from refund_agent.audit.audits_explanations_explain import replay

    request, trace, tools = event_time_denial()
    state = trace.steps[-1].state
    attachment = tools.explanations[request.request_id]
    e = attachment.explanation
    rebuilt = e.model_copy(
        update={
            "evidence": tuple(
                item.model_copy(
                    update={
                        "observed_at": request.at - timedelta(seconds=120),
                        "age_seconds": 120,
                    }
                )
                if item.field == "shipment.deliveries"
                else item
                for item in e.evidence
            )
        }
    )
    checks = replay(rebuilt, state, context=tools._explanation_context(state))
    assert checks["evidence"] is False
    assert checks["completeness"] is False


def test_later_arriving_scan_does_not_refer_a_correctly_explained_denial():
    request, trace, tools = event_time_denial(later_scan=True)
    assert trace.decision.outcome == c.Outcome.DENY
    assert trace.decision.reason == "already_delivered"
    attachment = tools.explanations[request.request_id]
    assert attachment.failure is None
    assert all(attachment.checks.values())


def partner_records(data, *, evaluation_at=None):
    records = inputs(data, evaluation_at=evaluation_at)
    return {
        "audits_explanations_segments": records.segments,
        "audits_explanations_register": records.register,
        "audits_explanations_history": records.history,
        "audits_explanations_disclosure": records.may_act,
        "audits_explanations_review_routes": records.review_routes,
    }


def test_registered_case_schedules_through_core_with_partner_records(
    monkeypatch,
):
    from refund_agent.release.suite import run_case

    suite = build_suite(1, components=(selector(RULE),))
    selected = build_candidate(suite)
    case = next(c for c in suite.cases if c.handler == "audits-explanations-replay")
    apply = extensions.apply

    def supply_records(candidate, hook, *, value, **context):
        if hook == "fixture_context":
            # Construct partner records, not substitute partner components.
            at = context["data"].orders[0].observed_at + timedelta(seconds=120)
            value = {**value, **partner_records(context["data"], evaluation_at=at)}
        return apply(candidate, hook, value=value, **context)

    monkeypatch.setattr(extensions, "apply", supply_records)
    result = run_case(case, selected)
    assert result["failed"] == [], result["decision"]
    assert result["decision"]["outcome"] == "deny"
    assert result["decision"]["reason"] == "amount_over_policy"
    assert result["payments"] == 0
    assert result["tool_calls"]


@pytest.mark.parametrize(
    "missing, owner",
    (
        ("audits_explanations_segments", "chapter 34"),
        ("audits_explanations_disclosure", "chapter 35"),
        ("audits_explanations_review_routes", "chapter 37"),
        ("audits_explanations_register", "policy register"),
        ("audits_explanations_history", "historical source"),
    ),
)
def test_fixture_context_refuses_each_absent_record(missing, owner):
    data = generate(7, orders=5, requests=5)
    value = partner_records(data)
    value.pop(missing)
    with pytest.raises(ValueError, match=owner):
        extensions.apply(
            candidate(),
            "fixture_context",
            value=value,
            data=data,
            mode="execute",
            purpose="loop-case",
        )
