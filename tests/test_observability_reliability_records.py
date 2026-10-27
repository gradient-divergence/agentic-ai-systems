# ruff: noqa: E402 -- initialize the fixed core before component imports.
"""Chapter 34 joins actual records and preserves the core's decision semantics."""

from datetime import timedelta

import pytest

from refund_agent.release.candidate import (
    build_candidate,
    build_tools,
    run_request,
)
from refund_agent.release.suite import build_suite, run_case

CORE = build_candidate(build_suite())

# isort: split
from refund_agent.agent import Agent
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.batch_stream.generate import opening, read_sources
from refund_agent.batch_stream.replay import CorrectionRecord, PaymentRecord
from refund_agent.batch_stream.view import ALLOWED, first_complete
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan
from refund_agent.canary.cloud_native_deployment_start import _budget
from refund_agent.observability_reliability.budget import signed_budget
from refund_agent.observability_reliability.component import selector
from refund_agent.observability_reliability.monitor import Monitor
from refund_agent.observability_reliability.objectives import (
    Observation,
    classify,
)
from refund_agent.observability_reliability.silence import (
    region_logs,
    region_of,
)
from refund_agent.observability_reliability.spans import audit_coverage
from refund_agent.sources.data_systems_release import SELECTOR
from refund_agent.tools import Tools


def test_signed_budget_supplies_actual_canary_start_reader_fields():
    from refund_agent.observability_reliability.fortnight import START

    record = signed_budget("observability-reliability-start", "candidate", START)
    plan = CanaryPlan(
        canary_id=record.canary_id,
        candidate_fingerprint=record.candidate_fingerprint,
        suite_fingerprint="suite",
        rollback_target_fingerprint="target",
        requirement_ref="observability-reliability-window",
        rehearsal_ref="observability-reliability-rehearsal",
        error_budget_ref="observability-reliability-budget",
    )
    _budget(record.canary_start_record(), plan, START)
    assert record.objectives[1].allowance == "112/3"
    import json

    from refund_agent.observability_reliability.budget import ErrorBudgetRecord

    supplied = record.canary_start_record()
    reloaded = ErrorBudgetRecord.model_validate_json(json.dumps(supplied["definition"]))
    assert reloaded == record
    assert supplied["definition_fingerprint"] == reloaded.fingerprint


def test_selected_identity_is_independent_of_demo_imports(tmp_path):
    import subprocess
    import sys

    program = """
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
build_candidate(build_suite())
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.observability_reliability.component import selector
from refund_agent.sources.data_systems_release import SELECTOR
def fingerprints():
    return tuple(build_candidate(build_suite(trials=1, components=[
        SELECTOR, event_selector(), selector(region_finality=fix)
    ])).fingerprint for fix in (False, True))
before = fingerprints()
import demos.observability_reliability_demo
after = fingerprints()
assert before == after, (before, after)
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def candidate(*, traced=True, fix=False):
    specs = [SELECTOR, event_selector()]
    if traced:
        specs.append(selector(region_finality=fix))
    suite = build_suite(trials=1, components=specs)
    return suite, build_candidate(suite)


@pytest.mark.parametrize(
    "fix,reason,cents",
    [(False, "eligible", 4999), (True, "already_delivered", 0)],
)
def test_silence_case_schedules_core_execution(fix, reason, cents):
    suite, selected = candidate(fix=fix)
    case = next(case for case in suite.cases if case.handler == "observability-reliability-silence")
    result = run_case(case, selected)
    assert result["failed"] == []
    assert result["decision"]["reason"] == reason
    assert (
        sum(p["amount_cents"] for p in result.get("receipts", ())) == cents
        if "receipts" in result
        else result["payments"] == bool(cents)
    )


def test_tracing_preserves_forty_public_cases():
    base_suite, base = candidate(traced=False)
    suite, observed = candidate()
    public = {case.case_id: case for case in base_suite.cases if case.part == "public"}
    cases = [case for case in suite.cases if case.part == "public"]
    assert len(cases) == 40
    for case in cases:
        plain = run_case(public[case.case_id], base)
        traced = run_case(case, observed)
        assert traced["decision"] == plain["decision"], case.case_id
        assert traced["payments"] == plain["payments"], case.case_id
        assert traced["review_items"] == plain["review_items"], case.case_id


def test_run_step_store_coverage_and_shadow_ledger_stays_empty():
    _, selected = candidate()
    data, log, request = opening(delivered=False)
    tools = build_tools(data, selected, "shadow", event_log=log)
    first = run_request(selected, tools, request)
    at = first_complete(log, request.at + ALLOWED)
    advance = tools.span_store.causal(request, at, "watermark_advance", reads=("event-progress",))
    tools._trace_links = ((advance.span_id, "released_by"),)
    tools.data = read_sources(data, at)
    second = run_request(selected, tools, request, at=at)
    spans = tools.span_store.entries
    roots = [s for s in spans if s.name == "run"]
    assert [s.attempt for s in roots] == [1, 2]
    assert roots[1].links == ((advance.span_id, "released_by"),)
    assert len([s for s in spans if s.name == "step"]) == len(first.steps) + len(second.steps)
    ids = {s.span_id for s in spans}
    assert all(s.parent_id is None or s.parent_id in ids for s in spans)
    assert all(s.start <= s.end for s in spans)
    assert audit_coverage(spans, tuple(tools.audits.values()))["pass"]
    assert len(tools.shadow_payments) == 1 and tools.payments == []
    assert all(s.tokens is None for s in spans)
    assert not any(
        "Please" in s.model_dump_json() or '"paid_cents"' in s.model_dump_json() for s in spans
    )


def payment(i=0, mode="shadow"):
    data, _, request = opening(delivered=False)
    request = request.model_copy(update={"request_id": f"observability-reliability-receipt-{i}"})
    data = data.model_copy(
        update={
            "requests": [request],
            "retrievals": [
                data.retrievals[0].model_copy(update={"request_id": request.request_id})
            ],
        }
    )
    trace = Agent(Tools(data)).run(request)
    assert trace.receipt is not None
    return PaymentRecord(trace.receipt, mode)


def test_steps_execute_before_audit_and_parent_their_tools():
    _, selected = candidate()
    data, log, request = opening(delivered=False)
    tools = build_tools(data, selected, "shadow", event_log=log)
    trace = run_request(selected, tools, request)
    spans = tools.span_store.entries
    steps = sorted((s for s in spans if s.name == "step"), key=lambda s: s.start)
    audit = next(s for s in spans if s.name == "tool:audit.append")
    assert len(steps) == len(trace.steps)
    assert all(s.end <= audit.start for s in steps[:-1])
    assert all(a.end <= b.start for a, b in zip(steps, steps[1:]))
    for step, executed in zip(steps, trace.steps):
        if executed.call is not None:
            tool = next(s for s in spans if s.name == "tool:" + executed.call.name)
            assert tool.parent_id == step.span_id
            assert step.start <= tool.start <= tool.end <= step.end


def test_closed_payment_window_keeps_late_corrections_across_reload(tmp_path):
    from pathlib import Path

    from refund_agent.observability_reliability import monitor as module
    from refund_agent.release import agent_evaluation_resources as resources

    payments = tuple(payment(i) for i in range(40))
    record = signed_budget("closed-corrections", "fingerprint", payments[0].receipt.at)
    path = tmp_path / "observability-reliability-budget.jsonl"
    allowed = resources.ACCESS[(module.__name__, Path(module.__file__).resolve())][0]
    with resources.scope(allowed, {path.resolve(): "append"}):
        monitor = Monitor(record, path)
        for p in payments:
            monitor.record_payment(p)
        assert monitor.close_windows(record.start + timedelta(hours=24)) == ()
        monitor = Monitor(record, path)
        for p in payments[:3]:
            monitor.record_correction(correction(p, record.start + timedelta(hours=54)))
        page = next((p for p in monitor.alerts if p.rule == "corrections"), None)
        assert page is not None, "closed boundary discarded late corrections"
        assert (page.count, page.threshold) == (3, 3)
        restarted = Monitor(record, path)
        assert restarted.close_windows(record.start + timedelta(hours=60)) == ()


def correction(p, at):
    return CorrectionRecord(
        p.receipt.receipt_id,
        p.mode,
        p.receipt.request.request_id,
        p.receipt.order_id,
        "observability-reliability-late-scan",
        p.receipt.request.at - timedelta(hours=1),
        at,
        "shipped",
        "delivered",
    )


def bundle_selection():
    from demos.observability_reliability_demo import bundle_selector

    return bundle_selector()


def test_spans_and_causal_records_read_the_decision_bundle_stamp():
    from refund_agent.observability_reliability.component import constructed_case
    from refund_agent.promotion.mlops_mle_bundle import from_candidate

    suite = build_suite(
        trials=1,
        components=[SELECTOR, event_selector(), bundle_selection(), selector()],
    )
    selected = build_candidate(suite)
    data, log, request = constructed_case()
    tools = build_tools(data, selected, "shadow", event_log=log)
    first = run_request(selected, tools, request)
    assert first.decision.reason == "shipment_incomplete"
    causal = tools.span_store.causal(request, request.at, "watermark_advance")
    identity = from_candidate(selected).fingerprint
    assert all(row.bundle == identity for row in tools.lineage)
    assert all(span.bundle == identity for span in tools.span_store.entries)
    assert causal.bundle == identity


def test_selected_tracing_keeps_registry_transport_usable_after_run(tmp_path):
    from refund_agent.observability_reliability.component import constructed_case
    from refund_agent.promotion.mlops_mle_bundle import from_candidate
    from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH
    from refund_agent.promotion.mlops_mle_registry import Registry

    suite = build_suite(
        trials=1,
        components=[SELECTOR, event_selector(), bundle_selection(), selector()],
    )
    selected = build_candidate(suite)
    data, log, request = constructed_case()
    tools = build_tools(data, selected, "shadow", event_log=log)
    run_request(selected, tools, request)
    bundle = from_candidate(selected)
    registry = Registry(REGISTRY_PATH, "observability-reliability-transport-" + str(tmp_path))
    registry.append(bundle)
    assert registry.read()[0]["value"]["fingerprint"] == bundle.fingerprint


@pytest.fixture
def canary_budget_session(tmp_path, request):
    from pathlib import Path

    from refund_agent.canary.cloud_native_deployment_component import (
        selector as canary_selector,
    )
    from refund_agent.canary.cloud_native_deployment_plan import sign_plan
    from refund_agent.canary.cloud_native_deployment_rehearsal import (
        rehearsal_session,
    )
    from refund_agent.canary.cloud_native_deployment_store import AppendStore
    from refund_agent.observability_reliability import monitor as module
    from refund_agent.observability_reliability.fortnight import START
    from refund_agent.release import agent_evaluation_resources as resources
    from refund_agent.release.extensions import RuntimeStore

    path = tmp_path / "observability-reliability-canary.jsonl"
    specs = [
        SELECTOR,
        event_selector(),
        canary_selector(store=RuntimeStore(str(path), "morgan", "append")),
        selector(),
    ]
    if getattr(request, "param", False):
        specs.append(bundle_selection())
    suite = build_suite(
        trials=1,
        components=specs,
    )
    selected = build_candidate(suite)
    start = START
    record = signed_budget("observability-reliability-lifecycle", selected.fingerprint, start)
    plan = sign_plan(
        CanaryPlan(
            canary_id=record.canary_id,
            candidate_fingerprint=selected.fingerprint,
            suite_fingerprint=suite.fingerprint,
            rollback_target_fingerprint="target",
            requirement_ref="rehearsal-only",
            rehearsal_ref="rehearsal-only",
            error_budget_ref="observability-reliability-budget",
        ),
        "morgan",
        record.signed_at,
    )
    allowed = resources.ACCESS[(module.__name__, Path(module.__file__).resolve())][0]
    stores = {path.resolve(): "append"}
    if getattr(request, "param", False):
        from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH

        stores[REGISTRY_PATH] = "append"
    with resources.scope(allowed, stores):
        store = AppendStore(
            path, "observability-reliability-" + tmp_path.parent.name + "-" + tmp_path.name
        )
        session = rehearsal_session(plan, store, start)
        store.append("budget", plan.error_budget_ref, record.canary_start_record())
        yield session, selected, record


def admitted_registry(session, selected, at):
    from refund_agent.promotion.mlops_mle_bundle import from_candidate
    from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH
    from refund_agent.promotion.mlops_mle_record import PromotionRecord, sign
    from refund_agent.promotion.mlops_mle_registry import Registry

    bundle = from_candidate(selected)
    registry = Registry(REGISTRY_PATH, session.store.namespace)
    registry.append(bundle)
    registry.append(
        sign(
            PromotionRecord(
                candidate_fingerprint=bundle.fingerprint,
                incumbent_fingerprint=None,
                from_stage="shadow",
                to_stage="canary",
                changes=(),
                requirements={},
                status="promoted",
                unmet=(),
                rollback_target=None,
                refs={"canary_id": session.plan.canary_id},
            ),
            "morgan",
            at,
        )
    )
    return registry, bundle


@pytest.mark.parametrize("canary_budget_session", [True], indirect=True)
def test_budget_end_demotes_but_pause_does_not_and_restart_deduplicates(
    canary_budget_session,
):
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    session, selected, record = canary_budget_session
    registry, bundle = admitted_registry(session, selected, record.start)
    StopRule(session).trigger("owner", at=record.start, owner="morgan")
    monitor = Monitor.for_canary(session, selected)
    assert registry.admitted(bundle.fingerprint, "canary")
    for i in range(38):
        monitor.observe(
            Observation(
                request_id=f"demote-{i}",
                received_at=record.start,
                decided_at=None,
                outcome=None,
                promise_hours=1,
                reason="shipment_incomplete",
            )
        )
    monitor.tick(record.start + timedelta(hours=2))
    assert session.stops[-1].outcome == "ended"
    assert not registry.admitted(bundle.fingerprint, "canary")
    assert registry.admitted(bundle.fingerprint, "shadow")
    demotion = registry.transitions()[-1]
    assert demotion.kind == "demotion"
    assert demotion.refs["stop"] == session.stops[-1].stop_id
    before = registry.transitions()
    Monitor.for_canary(session, selected)
    assert registry.transitions() == before


@pytest.mark.parametrize("canary_budget_session", [True], indirect=True)
@pytest.mark.parametrize("trigger", ["defect", "time_box"])
def test_reload_reconciles_an_ended_release_with_its_registry(
    canary_budget_session,
    trigger,
):
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    session, selected, record = canary_budget_session
    registry, bundle = admitted_registry(session, selected, record.start)
    stop = StopRule(session).end(trigger, at=record.start + timedelta(hours=2))
    Monitor.for_canary(session, selected)
    assert not registry.admitted(bundle.fingerprint, "canary")
    assert registry.transitions()[-1].refs["stop"] == stop.stop_id


@pytest.mark.parametrize("canary_budget_session", [True], indirect=True)
@pytest.mark.parametrize("trigger", ["defect", "time_box"])
def test_guarded_resume_demotes_when_core_resume_ends_the_release(
    canary_budget_session,
    trigger,
):
    from refund_agent.canary.cloud_native_deployment_stop import SettledEvidence, StopRule
    from refund_agent.observability_reliability.monitor import resume_canary

    session, selected, record = canary_budget_session
    registry, bundle = admitted_registry(session, selected, record.start)
    StopRule(session).trigger("owner", at=record.start, owner="morgan")
    at = record.end if trigger == "time_box" else record.start + timedelta(hours=2)
    settled = SettledEvidence(final=True, allowed_on_read=trigger != "defect")
    with pytest.raises(ValueError, match="expired|defect"):
        resume_canary(session, selected, "morgan", settled, at=at)
    assert session.stops[-1].trigger == trigger and session.outcome == "ended"
    assert not registry.admitted(bundle.fingerprint, "canary")
    assert registry.transitions()[-1].refs["stop"] == session.stops[-1].stop_id


def test_d6_report_is_unavailable_until_complete_mature_pairs_are_recorded(
    canary_budget_session,
):
    from refund_agent.observability_reliability.objectives import DenialPair

    session, selected, record = canary_budget_session
    monitor = Monitor.for_canary(session, selected)
    assert monitor.status()["d6"]["available"] is False
    for i in range(2):
        monitor.observe(
            Observation(
                request_id=f"d6-{i}",
                received_at=record.start,
                decided_at=record.start,
                outcome="deny",
                reason="already_delivered",
                promise_hours=1,
            )
        )
    pair = DenialPair(
        request_id="d6-0",
        agent_denied=True,
        baseline_denied=True,
        person_paid_at=record.start,
        found_false=False,
        label_mature_at=record.start + timedelta(days=30),
    )
    with pytest.raises(ValueError, match="pair"):
        monitor.record_denial_pair(pair.model_copy(update={"agent_denied": False}))
    monitor.record_denial_pair(pair)
    monitor.tick(record.start + timedelta(days=31))
    incomplete = monitor.status()["d6"]
    assert not incomplete["available"] and incomplete["pass"] is None
    assert incomplete["missing_pairs"] == 1
    monitor.record_denial_pair(
        pair.model_copy(
            update={
                "request_id": "d6-1",
                "label_mature_at": record.start + timedelta(days=32),
            }
        )
    )
    pending = monitor.status()["d6"]
    assert pending["pending"] == 1 and not pending["available"] and pending["pass"] is None
    monitor.tick(record.start + timedelta(days=33))
    report = monitor.status()["d6"]
    assert report["available"] and report["pass"]
    assert report["agent"] == report["baseline"] == report["mature"] == 2
    assert monitor.status()["objectives"]["timeliness"]["bad"] == 0
    assert Monitor.for_canary(session, selected).status()["d6"] == report
    with pytest.raises(ValueError, match="pair|request"):
        monitor.record_denial_pair(pair.model_copy(update={"request_id": "other"}))


@pytest.mark.parametrize("paused", [False, True])
def test_live_budget_exhaustion_ends_running_or_paused_grant(canary_budget_session, paused):
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    session, selected, record = canary_budget_session
    if paused:
        StopRule(session).trigger("owner", at=record.start, owner="morgan")
    monitor = Monitor.for_canary(session, selected)
    for i in range(38):
        monitor.observe(
            Observation(
                request_id=f"held-{i}",
                received_at=record.start,
                decided_at=None,
                outcome=None,
                promise_hours=1,
                reason="shipment_incomplete",
            )
        )
    monitor.tick(record.start + timedelta(hours=1, microseconds=1))
    assert session.outcome == "ended" and session.grant.budget_spent
    assert session.stops[-1].trigger == "budget_spent"
    before = session.stops
    restarted = Monitor.for_canary(session, selected)
    assert restarted.status()["spent"]
    assert session.stops == before


def test_already_spent_restart_ends_before_any_new_payment(
    canary_budget_session,
):
    session, selected, record = canary_budget_session
    # A crash persisted spend before delivering the end callback.
    monitor = Monitor(record, store=session.store)
    for i in range(38):
        monitor.observe(
            Observation(
                request_id=f"crashed-{i}",
                received_at=record.start,
                decided_at=None,
                outcome=None,
                promise_hours=1,
                reason="shipment_incomplete",
            )
        )
    monitor.tick(record.start + timedelta(hours=2))
    assert session.outcome == "running"
    restarted = Monitor.for_canary(session, selected)
    assert session.outcome == "ended" and session.grant.budget_spent
    assert restarted.status()["spent"]
    assert len(session.stops) == 1
    assert session.store.records("ledger") == {}


def test_combined_correction_exhaustion_refuses_resume_irreversibly(
    canary_budget_session,
):
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.canary.cloud_native_deployment_stop import (
        SettledEvidence,
        StopRule,
    )

    session, selected, record = canary_budget_session
    monitor = Monitor.for_canary(session, selected)
    for i in range(12):
        data, request = fixture(session.plan.canary_id, f"correction-{i}", record.start)
        receipt = Agent(Tools(data)).run(request).receipt
        assert receipt is not None
        p = PaymentRecord(receipt, "shadow" if i % 2 else "executed")
        monitor.record_payment(p)
        monitor.record_payment(p)
        monitor.record_correction(correction(p, record.start + timedelta(days=2)))
    assert monitor.status()["objectives"]["corrections"]["bad"] == 12
    assert session.outcome == "ended" and session.grant.budget_spent
    with pytest.raises(ValueError, match="only a paused, unspent grant"):
        StopRule(session).resume(
            "morgan",
            SettledEvidence(final=True, allowed_on_read=True),
            at=record.start + timedelta(days=3),
        )
    assert Monitor.for_canary(session, selected).status()["spent"]


def test_canary_monitor_rejects_changed_admission_projection(
    canary_budget_session,
):
    session, selected, record = canary_budget_session
    admission = record.canary_start_record()
    admission["objectives"][0]["allowance"] = 999
    session.store.append("budget", session.plan.error_budget_ref, admission)
    with pytest.raises(ValueError, match="admission"):
        Monitor.for_canary(session, selected)


def test_start_check_refuses_missing_session_authority(canary_budget_session):
    from refund_agent.observability_reliability import component

    session, selected, record = canary_budget_session
    checks = component.start_checks(
        candidate=selected,
        settings={},
        plan=session.plan,
        at=record.start,
        records={"error_budget": record.canary_start_record()},
    )
    assert checks[0]["passed"] is False


def test_resume_reloads_exhaustion_before_restoring_authority(
    canary_budget_session,
):
    from refund_agent.canary.cloud_native_deployment_stop import (
        SettledEvidence,
        StopRule,
    )
    from refund_agent.observability_reliability.monitor import resume_canary

    session, selected, record = canary_budget_session
    StopRule(session).trigger("owner", at=record.start, owner="morgan")
    monitor = Monitor(record, store=session.store)
    for i in range(38):
        monitor.observe(
            Observation(
                request_id=f"resume-{i}",
                received_at=record.start,
                decided_at=None,
                outcome=None,
                promise_hours=1,
                reason="shipment_incomplete",
            )
        )
    monitor.tick(record.start + timedelta(hours=2))
    assert session.outcome == "paused"
    with pytest.raises(ValueError, match="only a paused, unspent grant"):
        resume_canary(
            session,
            selected,
            "morgan",
            SettledEvidence(final=True, allowed_on_read=True),
            at=record.start + timedelta(hours=3),
        )
    assert session.outcome == "ended" and session.grant.budget_spent


def test_selected_canary_tools_persist_shadow_and_executed_spend(
    canary_budget_session,
):
    from refund_agent.batch_stream.generate import snapshot_log
    from refund_agent.canary.cloud_native_deployment_plan import in_cohort
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture

    session, selected, record = canary_budget_session
    for cohort in (False, True):
        data, request = fixture(session.plan.canary_id, f"budget-{cohort}", record.start)
        requester = next(
            f"budget-person-{i}"
            for i in range(1000)
            if in_cohort(session.plan.canary_id, f"budget-person-{i}") == cohort
        )
        request = request.model_copy(update={"requester_id": requester})
        data = data.model_copy(update={"requests": [request]})
        tools = build_tools(
            data,
            selected,
            "execute",
            event_log=snapshot_log(data),
            canary_session=session,
        )
        trace = run_request(selected, tools, request, at=record.start)
        assert len(tools.payments if cohort else tools.shadow_payments) == 1, trace.decision
        assert (trace.decision.outcome.value, trace.decision.reason) == (
            ("approve", "eligible") if cohort else ("refer", "tool_error")
        )
        observed = tools.budget_monitor.observations[request.request_id]
        assert (observed.outcome, observed.reason) == (
            trace.decision.outcome.value, trace.decision.reason
        )
    monitor = Monitor.for_canary(session, selected)
    assert {p.mode for p in monitor.payments.values()} == {
        "shadow",
        "executed",
    }
    assert monitor.status()["objectives"]["corrections"]["eligible"] == 2
    assert len(monitor.observations) == 2
    assert monitor.status()["objectives"]["timeliness"]["bad"] == 1


def test_shadow_receipts_do_not_hide_returned_runtime_failures(canary_budget_session):
    from refund_agent.batch_stream.generate import snapshot_log
    from refund_agent.canary.cloud_native_deployment_plan import in_cohort
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.runtimes.faults import FaultTools

    session, selected, record = canary_budget_session
    requester = next(
        f"failure-person-{i}"
        for i in range(1000)
        if not in_cohort(session.plan.canary_id, f"failure-person-{i}")
    )
    traces = []
    for i in range(38):
        data, request = fixture(session.plan.canary_id, f"failure-{i}", record.start)
        request = request.model_copy(update={"requester_id": requester})
        data = data.model_copy(update={"requests": [request]})
        tools = build_tools(
            data, selected, "execute", event_log=snapshot_log(data), canary_session=session
        )
        trace = run_request(
            selected, FaultTools(tools, tool="refund.execute", after=True), request, at=record.start
        )
        traces.append(trace)
        assert len(tools.shadow_payments) == 1 and tools.payments == []
        root = next(s for s in tools.span_store.entries if s.name == "run")
        assert root.reason == trace.decision.reason == "tool_error"
    assert [(t.decision.outcome.value, t.decision.reason) for t in traces] == [
        ("refer", "tool_error")
    ] * 38
    monitor = tools.budget_monitor
    assert {(o.outcome, o.reason) for o in monitor.observations.values()} == {
        ("refer", "tool_error")
    }
    status = monitor.status()["objectives"]["timeliness"]
    assert (status["eligible"], status["bad"], status["pending"]) == (38, 38, 0)
    assert status["signed_allowance"] == "112/3" and status["spent"]
    assert session.outcome == "ended" and session.grant.budget_spent
    ends = tuple(s for s in session.stops if s.outcome == "ended")
    assert len(ends) == 1 and ends[0].trigger == "budget_spent"
    assert len(session.store.rows("stop")) == 1
    assert Monitor.for_canary(session, selected).status() == monitor.status()
    assert session.stops == ends


@pytest.mark.parametrize("canary_budget_session", [True], indirect=True)
@pytest.mark.parametrize("paused", [False, True])
def test_selected_tools_expiry_demotes_without_an_explicit_reload(canary_budget_session, paused):
    from refund_agent.batch_stream.generate import snapshot_log
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    session, selected, record = canary_budget_session
    registry, bundle = admitted_registry(session, selected, record.start)
    if paused:
        StopRule(session).trigger("owner", at=record.start, owner="morgan")
    data, request = fixture(
        session.plan.canary_id, "tool-expiry", record.end - timedelta(seconds=1)
    )
    tools = build_tools(
        data, selected, "execute", event_log=snapshot_log(data), canary_session=session
    )
    assert registry.admitted(bundle.fingerprint, "canary")
    trace = run_request(selected, tools, request, at=record.end)
    assert trace.decision.reason == "canary_ended" and tools.payments == []
    assert session.outcome == "ended" and session.stops[-1].trigger == "time_box"
    assert not registry.admitted(bundle.fingerprint, "canary")
    demotions = tuple(r for r in registry.transitions() if r.kind == "demotion")
    assert len(demotions) == 1 and demotions[0].refs["stop"] == session.stops[-1].stop_id
    before = registry.transitions()
    Monitor.for_canary(session, selected)
    assert registry.transitions() == before


@pytest.mark.parametrize("canary_budget_session", [True], indirect=True)
def test_paused_host_must_close_due_count_windows(canary_budget_session):
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    session, selected, record = canary_budget_session
    registry, bundle = admitted_registry(session, selected, record.start)
    StopRule(session).trigger("owner", at=record.start, owner="morgan")
    monitor = Monitor.for_canary(session, selected)
    for i in range(17):
        monitor.observe(
            Observation(
                request_id=f"paused-window-{i}",
                received_at=record.start,
                decided_at=record.start + timedelta(minutes=10),
                outcome="refer",
                reason="tool_error" if i < 3 else "delivery_unknown",
                promise_hours=1,
            )
        )
    at = record.start + timedelta(hours=6)
    monitor.tick(at)
    assert monitor.alerts == []
    page, = monitor.close_windows(at)
    assert (page.rule, page.count, page.threshold) == ("timeliness", 3, 3)
    assert session.outcome == "paused" and registry.admitted(bundle.fingerprint, "canary")
    restarted = Monitor.for_canary(session, selected)
    assert restarted.close_windows(at) == () and restarted.alerts == [page]


@pytest.mark.parametrize(
    "mode,outcome,trigger",
    [
        ("executed", "paused", "owner"),
        ("shadow", "ended", "defect"),
    ],
)
def test_registered_breach_uses_real_stop_authority(canary_budget_session, mode, outcome, trigger):
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture

    session, selected, record = canary_budget_session
    data, request = fixture(session.plan.canary_id, "stale-breach", record.start)
    receipt = Agent(Tools(data)).run(request).receipt
    assert receipt is not None
    stale = receipt.model_copy(
        update={
            "evidence": receipt.evidence.model_copy(
                update={
                    "freshness": {
                        **receipt.evidence.freshness,
                        "shipment": 1860.0,
                    },
                }
            )
        }
    )
    monitor = Monitor.for_canary(session, selected)
    if mode == "executed":
        session.store.append("ledger", stale.idempotency_key, stale.model_dump(mode="json"))
        session.note_payment(stale)
    monitor.record_payment(PaymentRecord(stale, mode))
    assert session.outcome == outcome
    assert session.stops[-1].trigger == trigger
    assert "stale_payment" in session.stops[-1].detail


def test_shadow_only_correction_does_not_call_end_and_breach_is_separate():
    p = payment()
    at = p.receipt.request.at
    monitor = Monitor(signed_budget("modes", "fingerprint", at))
    ended, breaches = [], []
    monitor.register_budget_spent("canary.end", ended.append)
    monitor.register_breach(breaches.append)
    monitor.record_payment(p)
    change = correction(p, at + timedelta(days=2))
    monitor.record_correction(change)
    monitor.record_correction(change)
    assert monitor.status()["objectives"]["corrections"]["bad"] == 1
    assert not ended and not breaches


def test_correction_after_end_revises_original_payment_window_and_final_verdict(
    tmp_path,
):
    from pathlib import Path

    from refund_agent.observability_reliability import monitor as module
    from refund_agent.release import agent_evaluation_resources as resources

    p = payment()
    at = p.receipt.request.at
    record = signed_budget("late", "fingerprint", at)
    path = tmp_path / "observability-reliability-budget.jsonl"
    # Test-owned append store, authorized by the same bounded scope as the core.
    allowed = resources.ACCESS[(module.__name__, Path(module.__file__).resolve())][0]
    with resources.scope(allowed, {path.resolve(): "append"}):
        monitor = Monitor(record, path)
        monitor.record_payment(p)
        monitor.tick(record.end)
        assert monitor.status()["objectives"]["corrections"]["pending"] == 1
        monitor.record_correction(correction(p, record.end + timedelta(days=1)))
        monitor.tick(record.end + timedelta(days=2), outcomes_final=True)
        status = monitor.status()["objectives"]["corrections"]
        assert status["bad"] == 1 and status["eligible"] == 1
        assert status["realized_pass"] is False and not status["spent"]
        assert Monitor(record, monitor.path).status() == monitor.status()


def test_correction_exhaustion_invokes_registered_end_even_when_paused():
    payments = tuple(payment(i, "executed" if i % 2 else "shadow") for i in range(12))
    at = payments[0].receipt.request.at
    monitor = Monitor(signed_budget("corrections", "fingerprint", at))
    ended = []
    # The callback is independent of canary state; chapter 32 owns its transition.
    monitor.register_budget_spent("canary.end", ended.append)
    for p in payments:
        monitor.record_payment(p)
    for p in payments:
        monitor.record_correction(correction(p, at + timedelta(days=2)))
    assert len(ended) == 1 and ended[0].reason == "budget_spent"
    assert monitor.status()["objectives"]["corrections"]["bad"] == 12


def test_late_correction_revises_an_already_final_verdict():
    p = payment()
    record = signed_budget("revised-final", "fingerprint", p.receipt.at)
    monitor = Monitor(record)
    monitor.record_payment(p)
    monitor.tick(record.end, outcomes_final=True)
    assert monitor.status()["objectives"]["corrections"]["realized_pass"]
    monitor.record_correction(correction(p, record.end + timedelta(days=1)))
    status = monitor.status()["objectives"]["corrections"]
    assert status["bad"] == 1 and status["pending"] == 0
    assert status["realized_pass"] is False


def test_higher_tier_payment_is_outside_budget_but_still_checked_for_breach():
    from refund_agent import contracts as c

    p = payment()
    request = p.receipt.request.model_copy(update={"amount_cents": 60000})
    evidence = p.receipt.evidence.model_copy(
        update={
            "shipment_events": [
                c.ShipmentEvent(status="delivered", at=request.at - timedelta(hours=1)),
            ],
        }
    )
    p = PaymentRecord(
        p.receipt.model_copy(
            update={
                "request": request,
                "amount_cents": 60000,
                "evidence": evidence,
            }
        ),
        p.mode,
    )
    record = signed_budget("tier1-population", "fingerprint", p.receipt.at)
    monitor = Monitor(record)
    breaches = []
    monitor.register_breach(breaches.append)
    monitor.record_payment(p)
    monitor.record_correction(correction(p, p.receipt.at + timedelta(days=2)))
    monitor.tick(record.end, outcomes_final=True)
    status = monitor.status()["objectives"]["corrections"]
    assert status["eligible"] == 0 and status["bad"] == 0
    assert status["realized_pass"] is None
    assert len(breaches) == 1 and "forbidden_payment" in breaches[0]["rules"]


def test_missing_payment_evidence_is_refused_before_appending(tmp_path):
    from pathlib import Path

    from refund_agent.observability_reliability import monitor as module
    from refund_agent.release import agent_evaluation_resources as resources

    candidate()  # Establish the component's declared access before the test store.
    p = payment()
    path = tmp_path / "observability-reliability-budget.jsonl"
    record = signed_budget("evidence-required", "fingerprint", p.receipt.at)
    missing = p.receipt.model_copy(
        update={
            "receipt_id": "missing-evidence",
            "request": p.receipt.request.model_copy(update={"request_id": "missing"}),
            "evidence": None,
        }
    )
    allowed = resources.ACCESS[(module.__name__, Path(module.__file__).resolve())][0]
    with resources.scope(allowed, {path.resolve(): "append"}):
        monitor = Monitor(record, path)
        monitor.record_payment(p)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="recorded_payment_evidence_required"):
            monitor.record_payment(PaymentRecord(missing, p.mode))
        assert path.read_bytes() == before
        assert (p.mode, missing.receipt_id) not in monitor.payments


@pytest.mark.parametrize("hours,expected", [(40, "good"), (52, "bad")])
def test_held_claim_is_judged_at_terminal_answer(hours, expected):
    p = payment()
    at = p.receipt.request.at
    observed = Observation(
        request_id=p.receipt.request.request_id,
        received_at=at,
        decided_at=at + timedelta(hours=hours),
        outcome="approve",
        reason="eligible",
        promise_hours=48,
    )
    assert classify(observed, now=observed.decided_at).timeliness == expected


def test_receipt_evidence_checks_rule_breaches_without_confusing_corrections():
    p = payment()
    at = p.receipt.request.at
    observed = Observation(
        request_id=p.receipt.request.request_id,
        received_at=at,
        decided_at=p.receipt.at,
        outcome="approve",
        reason="eligible",
        promise_hours=1,
    )
    result = classify(
        observed,
        (p,),
        (correction(p, at + timedelta(days=2)),),
        now=at + timedelta(days=3),
    )
    assert result.corrections == "bad" and result.breaches == ()
    evidence = p.receipt.evidence.model_copy(
        update={"freshness": {**p.receipt.evidence.freshness, "shipment": 1860.0}}
    )
    stale = PaymentRecord(p.receipt.model_copy(update={"evidence": evidence}), p.mode)
    assert classify(observed, (stale,), now=at + timedelta(days=1)).breaches == ("stale_payment",)


def test_region_watermarks_use_chapter31_function_and_freshness_clock():
    from refund_agent.batch_stream.view import shipment_as_of, watermark

    data, log, request = opening()
    at = request.at
    regional = region_logs(log)
    for region in range(3):
        assert watermark(regional[region], at) == watermark(
            type(log)(e for e in log.entries if region_of(e.order_id) == region),
            at,
        )
    _, selected = candidate(fix=True)
    tools = build_tools(data, selected, "shadow", event_log=log)
    trace = run_request(selected, tools, request)
    assert (
        trace.steps[1].result.shipment.observed_at
        == shipment_as_of(log, request.order_reference, at).observed_at
    )
