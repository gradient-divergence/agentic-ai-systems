"""Chapter 34 registers only its own code through chapter 29's fixed hooks."""

from datetime import timedelta
from pathlib import Path

from refund_agent.batch_stream.tools import EventTimeTools
from refund_agent.batch_stream.view import (
    complete_through,
    first_complete,
    shipment_as_of,
)
from refund_agent.observability_reliability.budget import ErrorBudgetRecord
from refund_agent.observability_reliability.monitor import Monitor
from refund_agent.observability_reliability.silence import (
    RegionWatermarkTools,
    region_logs,
    region_of,
)
from refund_agent.observability_reliability.spans import (
    SpanStore,
    TracedModel,
    TracedTools,
)
from refund_agent.promotion.mlops_mle_component import REGISTRY_STORE
from refund_agent.release.extensions import Component, RuntimeStore, register
from refund_agent.runtimes.faults import FaultTools
from refund_agent.sources import ContractedTools

MODULE = "refund_agent.observability_reliability.component"
SPAN_STORE = RuntimeStore(
    "../../.tmp-sol/observability-reliability-spans.jsonl", "morgan", "append"
)
BUDGET_STORE = RuntimeStore(
    "../../.tmp-sol/observability-reliability-budget.jsonl", "morgan", "append"
)
BUDGET_DEFINITION = RuntimeStore(
    "../../.tmp-sol/observability-reliability-error-budget.json",
    "morgan",
    "append",
)


def selector(*, region_finality=True):
    if type(region_finality) is not bool:
        raise ValueError("regional finality must be boolean")
    return {
        "name": "observability-reliability",
        "module": MODULE,
        "settings": {"region_finality": region_finality},
    }


def compose(*, base, kwargs, candidate, settings, mode, context):
    if not issubclass(base, EventTimeTools) or not issubclass(base, ContractedTools):
        raise ValueError("observability requires selected contracts and event-time components")
    if set(settings) != {"region_finality"} or type(settings["region_finality"]) is not bool:
        raise ValueError("regional finality must be explicitly boolean")
    monitor = None
    session = context.get("canary_session") if mode != "shadow" else None
    if session is not None and session.plan.candidate_fingerprint == candidate.fingerprint:
        if session.store.latest("budget", session.plan.error_budget_ref) is None:
            record = context.get("observability_reliability_budget")
            if session.grant.lane != "rehearsal" or not isinstance(record, ErrorBudgetRecord):
                raise ValueError("exact budget admission required before execution")
            session.store.append(
                "budget",
                session.plan.error_budget_ref,
                record.canary_start_record(),
            )
        monitor = Monitor.for_canary(session, candidate)
    return type(
        "ObservabilityReliabilityTools",
        (TracedTools, RegionWatermarkTools, base),
        {},
    ), {
        **kwargs,
        "span_store": context.get("span_store", SpanStore()),
        "region_finality": settings["region_finality"],
        "traced_mode": mode,
        "budget_monitor": monitor,
    }


def before_run(*, tools, request, at, runtime_settings, candidate, settings):
    tools = tools.inner if isinstance(tools, FaultTools) else tools
    if tools.batch_stream_case == "observability-reliability-silence":
        observed = at or request.at
        tools.data = tools.data.model_copy(
            update={
                "shipments": [
                    shipment_as_of(tools.event_log, o.order_id, observed) for o in tools.data.orders
                ]
            }
        )
    logical_at = tools.run_time(request) if at is None else at
    roots = tuple(
        s for s in tools.span_store.trace(f"trace-{request.request_id}") if s.name == "run"
    )
    if roots and roots[-1].status == "held" and not tools._trace_links:
        progress = (
            region_logs(tools.event_log)[region_of(request.order_reference)]
            if tools.region_finality
            else tools.event_log
        )
        if complete_through(
            progress,
            logical_at,
            tools.margin,
            request.at + tools.allowed_lateness,
        ):
            event = max(
                (e for e in progress.entries if e.arrived_at <= logical_at),
                key=lambda e: e.at,
            )
            advance = tools.span_store.causal(
                request,
                logical_at,
                "watermark_advance",
                reads=(event.event_id,),
            )
            tools._trace_links = ((advance.span_id, "released_by"),)
    tools.begin_trace(request, logical_at, links=tools._trace_links)
    tools._trace_links = ()
    if tools.budget_monitor is not None:
        from refund_agent.observability_reliability.objectives import (
            Observation,
        )
        from refund_agent.rules import tier_for

        monitor = tools.budget_monitor
        monitor.tick(max(monitor.now, logical_at), outcomes_final=monitor.outcomes_final)
        if tier_for(request.amount_cents) == 1 and request.request_id not in monitor.observations:
            shipment = shipment_as_of(tools.event_log, request.order_reference, logical_at)
            held = request.reason == "not_delivered" and not any(
                e.status == "delivered" and e.at <= request.at for e in shipment.events
            )
            monitor.observe(
                Observation(
                    request_id=request.request_id,
                    received_at=request.at,
                    decided_at=None,
                    outcome=None,
                    reason="shipment_incomplete",
                    promise_hours=48 if held else 1,
                )
            )


def wrap_model(*, value, tools, **context):
    tools = tools.inner if isinstance(tools, FaultTools) else tools
    return TracedModel(value, tools)


def after_run(*, value, tools, request, at, runtime_settings, candidate, settings):
    tools = tools.inner if isinstance(tools, FaultTools) else tools
    # ContractedTools writes lineage at the run boundary before this hook.
    assert tools._trace_root is not None
    rows = tuple(
        r
        for r in tools.lineage
        if r.request.request_id == request.request_id
        and r.decision_at == tools._trace_root["start"]
    )
    for row in rows:
        with tools._operation("store:lineage") as span:
            from refund_agent.selection import fingerprint

            span["writes"] = (fingerprint(row.model_dump(mode="json")),)
            span["reads"] = tuple(i.fingerprint for i in row.inputs)
            span["sources"] = tuple(i.source for i in row.inputs)
    tools.finish_trace(value)
    if tools.budget_monitor is not None:
        from refund_agent.observability_reliability.objectives import (
            Observation,
        )

        monitor = tools.budget_monitor
        old = monitor.observations.get(request.request_id)
        if old is not None and (old.decided_at is None or old.reason == "shipment_incomplete"):
            shadow_payment = next(
                (
                    p
                    for p in tools.shadow_payments
                    if value.decision.reason == "canary_outside_cohort"
                    and p.request is not None
                    and p.request.request_id == request.request_id
                ),
                None,
            )
            decision = shadow_payment.decision if shadow_payment else value.decision
            monitor.observe(
                Observation(
                    request_id=request.request_id,
                    received_at=request.at,
                    decided_at=shadow_payment.at
                    if shadow_payment
                    else value.steps[-1].state.execution_at or request.at,
                    outcome=decision.outcome.value,
                    reason=decision.reason,
                    promise_hours=old.promise_hours,
                )
            )
        if tools.canary_session.outcome == "ended":
            tools.budget_monitor = Monitor.for_canary(tools.canary_session, candidate)
    return value


def start_checks(*, candidate, settings, plan, at, records):
    """A host must supply the actual session, never just a numeric projection."""
    import json

    try:
        session = records.get("canary_session")
        admission = records.get("error_budget")
        if session is None or session.plan != plan or not isinstance(admission, dict):
            raise ValueError("budget monitor requires the actual canary_session")
        record = ErrorBudgetRecord.model_validate_json(json.dumps(admission["definition"]))
        if admission != record.canary_start_record() or (
            record.canary_id != plan.canary_id
            or record.candidate_fingerprint != candidate.fingerprint
            or record.start != (session.grant.started_at if session.grant else at)
            or record.end != record.start + timedelta(days=plan.time_box_days)
        ):
            raise ValueError("exact budget admission plan/candidate/window differs")
        if session.grant is not None:
            monitor = Monitor.for_canary(session, candidate)
            monitor.tick(max(monitor.now, at), outcomes_final=monitor.outcomes_final)
            if monitor.status()["spent"] or session.outcome == "ended":
                raise ValueError("canary budget spent or authority ended")
        detail, passed = (
            "exact budget definition and persistent authority checked",
            True,
        )
    except (ValueError, KeyError, TypeError) as error:
        detail, passed = str(error), False
    return [
        {
            "name": "observability-reliability-budget",
            "passed": passed,
            "detail": detail,
        }
    ]


def runtime_paths():
    root = Path(__file__).resolve().parents[1]
    return root / SPAN_STORE.path, root / BUDGET_STORE.path


def constructed_case():
    from refund_agent.observability_reliability.fortnight import (
        _request_data,
        make_stream,
        source_snapshot,
    )

    stream = make_stream()
    request = next(r for r in stream.requests if r.request_id == stream.silence.request_id)
    data = source_snapshot(_request_data(stream, request), stream.log, request.at)
    return data, stream.log, request


def cases(*, settings):
    from refund_agent.release.suite import Case
    from refund_agent.runtimes.swap import SwapCase

    data, _, request = constructed_case()
    fixed = settings["region_finality"]
    return (
        Case(
            "observability-reliability-silence",
            "observability",
            "regional-silence",
            1,
            SwapCase(
                "observability-reliability-silence",
                "observability",
                data,
                request,
                0 if fixed else request.amount_cents,
            ),
            handler="observability-reliability-silence",
            expected={
                "paid_cents": 0 if fixed else request.amount_cents,
                "reason": "already_delivered" if fixed else "eligible",
                "outcome": "deny" if fixed else "approve",
            },
            must_pass=True,
        ),
    )


def fixture_context(*, value, data, mode, purpose, candidate, settings, **fixture):
    case = fixture.get("case")
    if (
        purpose == "loop-case"
        and case is not None
        and case.handler == "observability-reliability-silence"
    ):
        fixed, log, request = constructed_case()
        if case.fixture is None:
            raise ValueError("observability case needs its fixed fixture")
        if data != fixed or request != case.fixture.request:
            raise ValueError("observability fixture changed")
        return {**value, "event_log": log, "batch_stream_fixture": True}
    return value


def run_case(*, case, candidate, settings, execution, **context):
    if case.handler != "observability-reliability-silence":
        return None
    _, log, request = constructed_case()
    execution.run()
    progress = (
        region_logs(log)[region_of(request.order_reference)] if settings["region_finality"] else log
    )
    execution.run(at=first_complete(progress, request.at + timedelta(hours=27)))
    return True


register(
    Component(
        name="observability-reliability",
        version="observability-reliability-1",
        module=MODULE,
        modules=tuple(
            "refund_agent.observability_reliability." + name
            for name in (
                "budget",
                "reasons",
                "objectives",
                "monitor",
                "silence",
                "spans",
                "fortnight",
                "incident",
            )
        ),
        files=("release/owners.json",),
        order=34,
        stores=(SPAN_STORE, BUDGET_STORE, BUDGET_DEFINITION, REGISTRY_STORE),
        hooks={
            "compose": compose,
            "wrap_model": wrap_model,
            "start_checks": start_checks,
            "before_run": before_run,
            "after_run": after_run,
            "cases": cases,
            "fixture_context": fixture_context,
            "run_case": run_case,
            "run_loop_case": run_case,
        },
        slots={"34": "traces, error budgets and regional finality"},
    )
)
