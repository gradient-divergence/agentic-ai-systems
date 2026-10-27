"""Timed cases counted once, at their terminal decision, in chapter 29's suite."""

from datetime import timedelta

from refund_agent import contracts as c
from refund_agent.batch_stream.generate import (
    dataset,
    read_sources,
    scan,
    snapshot_log,
)
from refund_agent.batch_stream.view import first_complete
from refund_agent.release import extensions
from refund_agent.runtimes.faults import FaultTools


def extend(suite, **settings):
    from refund_agent.batch_stream.component import selector

    return extensions.extend_suite(suite, (selector(**settings),))


def cases(*, settings):
    from refund_agent.release.suite import Case
    from refund_agent.runtimes.swap import SwapCase

    result = []
    for name, reason, paid in (
        ("absence", "already_delivered", 0),
        ("return-window", "outside_return_window", 0),
        ("corrections", "eligible", 3499),
    ):
        data, _, request = _component_fixture(name)
        result.append(
            Case(
                f"batch-stream-{name}",
                "events",
                name,
                1,
                SwapCase(
                    f"batch-stream-{name}", "events", data, request, paid
                ),
                handler=f"batch-stream-{name}",
                expected={
                    "paid_cents": paid,
                    "reason": reason,
                    "outcome": "approve" if paid else "deny",
                },
                must_pass=True,
            )
        )
    return tuple(result)


def run_case(*, case, candidate, settings, execution):
    if not case.handler.startswith("batch-stream-"):
        return None
    execution.run()
    if case.handler != "batch-stream-return-window":
        _, log, request = _component_fixture(case.failure_class)
        execution.run(at=_final_at(log, request, settings))
    return True


def fixture_context(
    *, value, data, mode, purpose, candidate, settings, **fixture
):
    case = fixture.get("case")
    if purpose == "loop-case" and case.handler.startswith("batch-stream-"):
        fixed, log, request = _component_fixture(case.failure_class)
        if fixed != case.fixture.data or request != case.fixture.request:
            raise ValueError("batch_stream_fixture_changed")
    elif purpose == "loop-case":
        log = _loop_log(data, case.fixture.request)
    else:
        if purpose == "freshness" and case.handler.startswith("shipment-"):
            from refund_agent.runtimes.swap import _project

            # Isolate the source observation clock, without a background feed.
            data = _project(data, data.requests[0])
        log = snapshot_log(data)
    return {
        **value,
        "event_log": log,
        "batch_stream_fixture": purpose == "loop-case",
        "batch_stream_case": case.handler if case is not None else None,
    }


def changed_source(*, tools, case, candidate, settings):
    if case.handler == "shipment":
        tools.event_log = snapshot_log(tools.data)


def component_settings(candidate):
    return next(
        row["settings"]
        for row in candidate.tools_composition["components"]
        if row["name"] == "batch-stream"
    )


def _loop_log(data, request):
    log = snapshot_log(data)
    # Independent constructed scans supply progress, never delivery evidence.
    base_at = max(request.at, *(e.arrived_at for e in log.entries))
    for minute in range(5, 38 * 60 + 1, 5):
        made = base_at + timedelta(minutes=minute)
        log.append(
            scan(f"batch-stream-suite-feed-{minute}", "shipped", made, made)
        )
    return log


def _final_at(log, request, settings):
    return first_complete(
        log,
        request.at + timedelta(seconds=settings["allowed_seconds"]),
        timedelta(seconds=settings["margin_seconds"]),
    )


def run_loop_case(*, case, candidate, settings, mode, execution):
    if case.handler.startswith("batch-stream-"):
        return run_case(
            case=case,
            candidate=candidate,
            settings=settings,
            execution=execution,
        )
    if case.handler != "loop":
        return None
    execution.run()
    fixture = case.fixture
    if (
        fixture.request.reason == "not_delivered"
        and (
            case.expected["paid_cents"] > 0
            or case.expected.get("reason") == "approval_required"
            or case.expected.get("named_reason") == "approval_required"
        )
        and candidate.name != "deny-all"
    ):
        log = _loop_log(fixture.data, fixture.request)
        execution.run(at=_final_at(log, fixture.request, settings))
    if fixture.approval_role and execution.approve():
        execution.run()
    return True


def before_run(*, tools, request, at, runtime_settings, candidate, settings):
    mounted = tools.inner if isinstance(tools, FaultTools) else tools
    if mounted.batch_stream_case == "batch-stream-absence" and at is None:
        # Probe the core's tools without adding fixture calls to the agent transcript.
        state = c.State(
            request=request,
            trace_id=f"batch-stream-boundary-{request.request_id}",
            execution_at=request.at,
        )
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            mounted.call(c.ToolCall(name=name), state)
        mounted.call(
            c.ToolCall(
                name="refund.propose",
                arguments=c.ToolArguments(
                    amount_cents=request.amount_cents, reason=request.reason
                ),
            ),
            state,
        )
        result = mounted.call(
            c.ToolCall(
                name="refund.execute",
                arguments=c.ToolArguments(
                    amount_cents=request.amount_cents,
                    idempotency_key=f"batch-stream-boundary-{request.request_id}",
                ),
            ),
            state,
        )
        mounted.batch_stream_execution_boundaries += (result.reason,)
    if at is not None and mounted.batch_stream_fixture:
        # Only the core's explicitly timed fixtures simulate a live reread.
        mounted.data = read_sources(mounted.data, at)


def after_run(
    *, value, tools, request, at, runtime_settings, candidate, settings
):
    from refund_agent.batch_stream.replay import PaymentRecord, corrections

    mounted = tools.inner if isinstance(tools, FaultTools) else tools
    mounted.batch_stream_attempts += (value.decision.model_dump(mode="json"),)
    if mounted.batch_stream_case == "batch-stream-corrections":
        shadow = mounted.batch_stream_mode == "shadow"
        payments = mounted.shadow_payments if shadow else mounted.payments
        mode = "shadow" if shadow else "executed"
        before = mounted.ledger
        records = corrections(
            mounted.event_log, [PaymentRecord(p, mode) for p in payments]
        )
        if payments and (
            len(records) != 1
            or records[0].mode != mode
            or mounted.ledger != before
        ):
            raise ValueError("correction_preserves_payment_" + mode)
    return value


def _component_fixture(name):
    from refund_agent.batch_stream.generate import opening
    from refund_agent.batch_stream.log import EventLog

    data, log, request = (
        opening(delay=timedelta(hours=50))
        if name == "corrections"
        else opening()
    )
    if name == "return-window":
        made = request.at - timedelta(days=31, hours=1)
        event = scan(
            request.order_reference,
            "delivered",
            made,
            made + timedelta(days=2),
        )
        log = EventLog(
            [event]
            + [e for e in log.entries if e.order_id != request.order_reference]
        )
        request = request.model_copy(update={"reason": "damaged"})
        data = dataset((request,), log)
    return data, log, request
