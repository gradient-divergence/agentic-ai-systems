"""Chapter 31: absence is held until the carrier's finality rule is met."""

import json
from collections import Counter
from datetime import timedelta

import pytest

from demos.batch_stream_capture import key_run
from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.batch_stream.experiment import exercise_scans
from refund_agent.batch_stream.generate import (
    START,
    generate,
    opening,
    read_sources,
    scan,
    snapshot_log,
)
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.replay import (
    PaymentRecord,
    Views,
    apply,
    checkpoint,
    corrections,
    replay,
    restore,
)
from refund_agent.batch_stream.tools import EventTimeTools, LogShipmentTools
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    complete_through,
    first_complete,
    late_events,
    shipment_as_of,
    watermark,
)
from refund_agent.batch_stream.windows import day_counts
from refund_agent.evaluation import _positive_data


def test_settled_absence_denial_keeps_contracted_audit_evidence():
    from refund_agent.batch_stream.component import selector
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import _CaseRun, build_suite, run_case
    from refund_agent.sources.data_systems_release import SELECTOR

    suite = build_suite(trials=1, components=[SELECTOR, selector()])
    candidate = build_candidate(suite)
    case = next(c for c in suite.cases if c.case_id == "batch-stream-absence")
    result = run_case(case, candidate)
    assert not result["failed"]
    assert result["decision"]["outcome"] == "deny"
    assert result["decision"]["reason"] == "already_delivered"
    assert result["payments"] == 0
    execution = _CaseRun(case, candidate, "execute")
    execution.run(at=case.fixture.request.at + timedelta(hours=40))
    observed = execution._artifacts()
    row = observed.tools.lineage[-1]
    shipment = next(ref for ref in row.inputs if ref.record_kind == "shipments")
    recorded = observed.tools.record_store.get(shipment.fingerprint)
    assert "delivered" in [event["status"] for event in recorded["events"]]
    assert row.decision.reason == "already_delivered"


@pytest.mark.parametrize("name", ["absence", "return-window", "corrections"])
def test_component_handlers_schedule_core_owned_execution(name):
    from refund_agent.batch_stream import suite as component
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import (
        _CaseRun,
        build_suite,
        run_case,
    )

    suite = component.extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    case = next(c for c in suite.cases if c.case_id == f"batch-stream-{name}")
    assert case.fixture is not None and case.expected is not None
    execution = _CaseRun(case, candidate, "execute")
    assert (
        component.run_case(
            case=case,
            candidate=candidate,
            settings=component.component_settings(candidate),
            execution=execution.context,
        )
        is True
    )
    observed = execution._artifacts()
    assert execution.context.runs == (1 if name == "return-window" else 2)
    assert (
        observed.trace.decision.request_id == case.fixture.request.request_id
    )
    assert not run_case(case, candidate)["failed"]


def test_selected_component_uses_canonical_builder_and_dispatch():
    from refund_agent.batch_stream.suite import (
        component_settings,
        extend,
        run_loop_case,
    )
    from refund_agent.release.candidate import build_candidate, build_tools
    from refund_agent.release.suite import _CaseRun, build_suite, run_case

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    data, log, request = opening(False)
    tools = build_tools(data, candidate, "execute", event_log=log)
    assert isinstance(tools, EventTimeTools)
    case = next(c for c in suite.cases if c.case_id == "positive-100")
    result = run_case(case, candidate)
    assert not result["failed"]
    execution = _CaseRun(case, candidate, "execute")
    assert (
        run_loop_case(
            case=case,
            candidate=candidate,
            settings=component_settings(candidate),
            mode="execute",
            execution=execution.context,
        )
        is True
    )
    observed = execution._artifacts()
    assert observed.tool_calls
    assert (
        observed.tools.batch_stream_attempts[0]["reason"]
        == "shipment_incomplete"
    )
    assert result["decision"]["executed"]


@pytest.mark.parametrize(
    "case_id",
    [
        "listed-173",
        "listed-174",
        "listed-175",
        "listed-177",
        "listed-179",
        "public-r-7-0023",
    ],
)
def test_referral_cases_get_timed_reruns_through_core(case_id):
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import (
        _CaseRun,
        _run_loop_case,
        build_suite,
    )

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == case_id)
    assert case.tier in (2, 3) and case.expected["paid_cents"] == 0
    execution = _CaseRun(case, candidate, "execute")
    result = _run_loop_case(case, candidate, "execute", execution)
    observed = execution._artifacts()
    assert execution.context.runs == 2
    assert [
        attempt["reason"] for attempt in observed.tools.batch_stream_attempts
    ] == ["shipment_incomplete", "approval_required"]
    assert not result["failed"]
    assert result["decision"]["outcome"] == "refer"
    assert not result["decision"]["executed"]
    assert result["payments"] == 0 and not observed.tools.payments


@pytest.mark.parametrize("case_id", ["listed-176", "listed-191"])
def test_denial_cases_do_not_get_unneeded_timed_reruns(case_id):
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import (
        _CaseRun,
        _run_loop_case,
        build_suite,
    )

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == case_id)
    execution = _CaseRun(case, candidate, "execute")
    result = _run_loop_case(case, candidate, "execute", execution)
    assert execution.context.runs == 1
    assert not result["failed"] and result["payments"] == 0


def test_default_suite_matches_chapter29_stored_release():
    from pathlib import Path

    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    path = Path(__file__).resolve().parents[2] / (
        "demos/data/agent_evaluation_release.json"
    )
    records = json.loads(path.read_text())["records"]
    assert build_suite().fingerprint == records[0]["suite_fingerprint"]
    assert all(
        row["suite_fingerprint"] == records[0]["suite_fingerprint"]
        for row in records
    )
    for row in records:
        old = row["candidate"]
        current = build_candidate(
            build_suite(),
            old["name"],
            old["runtime"]["identity"],
            old["tools_composition"]["verifier"],
        )
        assert current.record() == old
        assert current.fingerprint == row["candidate_fingerprint"]


@pytest.mark.parametrize(
    "boundary", ["request", "watermark", "before-final", "final"]
)
def test_direct_execution_requires_final_evidence(boundary):
    data, log, request = opening(False)
    edges = (request.at + MARGIN, request.at + MARGIN + ALLOWED)
    log = EventLog(
        list(log.entries)
        + [
            scan(f"boundary-{i}", "shipped", at, at)
            for i, at in enumerate(edges)
        ]
    )
    final_at = first_complete(log, request.at + ALLOWED)
    at = {
        "request": request.at,
        "watermark": first_complete(log, request.at),
        "before-final": final_at - timedelta(microseconds=1),
        "final": final_at,
    }[boundary]
    tools = EventTimeTools(read_sources(data, at), event_log=log)
    state = c.State(
        request=request, trace_id="direct-payment", execution_at=at
    )
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        assert tools.call(c.ToolCall(name=name), state).ok
    proposed = tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(
                amount_cents=request.amount_cents, reason=request.reason
            ),
        ),
        state,
    )
    assert proposed.ok and proposed.record_id in tools.proposals
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(
                amount_cents=request.amount_cents,
                idempotency_key="direct-payment",
            ),
        ),
        state,
    )
    if boundary == "final":
        assert result.ok and state.receipt is not None
        assert len(tools.payments) == 1
        assert tools.payments[0].amount_cents == request.amount_cents
    else:
        assert not result.ok and result.reason == "shipment_incomplete"
        assert state.receipt is None
        assert not tools.ledger and not tools.payments


def test_direct_execution_at_watermark_only_boundary():
    data, log, request = opening(False)
    edge = request.at + MARGIN
    log = EventLog(
        list(log.entries) + [scan("boundary", "shipped", edge, edge)]
    )
    at = first_complete(log, request.at)
    assert watermark(log, at) == request.at
    tools = EventTimeTools(
        read_sources(data, at), event_log=log, allowed_lateness=timedelta(0)
    )
    state = c.State(
        request=request, trace_id="watermark-only", execution_at=at
    )
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        assert tools.call(c.ToolCall(name=name), state).ok
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(
                amount_cents=request.amount_cents, reason=request.reason
            ),
        ),
        state,
    ).ok
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(
                amount_cents=request.amount_cents, idempotency_key="watermark"
            ),
        ),
        state,
    )
    assert result.ok and len(tools.payments) == 1
    assert state.receipt is not None


def test_must_pass_execution_boundaries():
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import _CaseRun, _run_loop_case, build_suite

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    case = next(c for c in suite.cases if c.handler == "batch-stream-absence")
    assert case.must_pass
    execution = _CaseRun(case, candidate, "execute")
    result = _run_loop_case(case, candidate, "execute", execution)
    assert result["failed"] == []
    tools = execution._artifacts().tools
    assert tools.batch_stream_execution_boundaries == ("shipment_incomplete",)
    assert not tools.payments and not tools.ledger
    assert result["decision"]["reason"] == "already_delivered"


def test_selected_suite_has_its_own_recorded_fingerprint():
    from pathlib import Path

    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import (
        build_candidate,
    )
    from refund_agent.release.suite import build_suite

    path = (
        Path(__file__).resolve().parents[2]
        / "demos/data/batch-stream-records.json"
    )
    record = json.loads(path.read_text())["candidate"]["release"]
    suite = extend(build_suite())
    candidate = build_candidate(suite)
    assert suite.fingerprint != build_suite().fingerprint
    assert suite.fingerprint == candidate.suite["fingerprint"]
    assert record["suite_fingerprint"] == suite.fingerprint
    assert record["candidate_fingerprint"] == candidate.fingerprint
    assert (
        record["manifest"]["components"]
        == candidate.tools_composition["components"]
    )


@pytest.mark.parametrize("changed", ["component", "core"])
def test_one_byte_change_changes_identity_and_refuses_shadow(
    monkeypatch, tmp_path, changed
):
    from dataclasses import replace
    from pathlib import Path

    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import PACKAGE, build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign
    from refund_agent.release.suite import build_suite
    from refund_agent.shadow.__main__ import start

    ordinary = build_candidate(build_suite(trials=1))
    full = extend(build_suite(trials=1))
    suite = replace(
        full,
        cases=tuple(
            c for c in full.cases if c.part in ("public", "positive", "events")
        ),
        generated=(),
    )
    candidate = build_candidate(suite)
    journal = Journal(tmp_path / "journal.jsonl")
    passed = evaluate_release(candidate, suite, journal)
    assert passed["decision"] == "pass"
    signed = sign(passed, journal, "morgan", "2026-10-12T12:01:00+00:00")
    path = tmp_path / "release.json"
    path.write_text(json.dumps(signed))
    changed_path = PACKAGE / (
        "batch_stream/tools.py" if changed == "component" else "tools.py"
    )
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: read_bytes(p) + (b"\n" if p == changed_path else b""),
    )
    rebuilt = build_candidate(suite)
    assert rebuilt.fingerprint != candidate.fingerprint
    assert suite.fingerprint != passed["suite_fingerprint"]
    if changed == "component":
        assert (
            build_candidate(build_suite(trials=1)).record()
            == ordinary.record()
        )
    with pytest.raises(ValueError, match="fingerprint"):
        start(
            path, suite=suite, journal=journal, output=tmp_path / "shadow.json"
        )
    assert not (tmp_path / "shadow.json").exists()


def test_unselected_batch_stream_registration_changes_no_identity(monkeypatch):
    from refund_agent.batch_stream.component import MODULE
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    plugin = extensions._REGISTRY["batch-stream"]
    monkeypatch.delitem(extensions._REGISTRY, "batch-stream")
    before = build_candidate(build_suite(trials=1))
    extensions.register(plugin)
    after = build_candidate(build_suite(trials=1))
    assert plugin.module == MODULE
    assert after.record() == before.record()
    assert after.fingerprint == before.fingerprint


def test_selected_release_keeps_journal_blocks_and_signoff(tmp_path):
    from dataclasses import replace

    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign
    from refund_agent.release.suite import build_suite
    from refund_agent.shadow.__main__ import validate_release

    full = extend(build_suite(trials=1))
    suite = replace(
        full,
        cases=tuple(
            c for c in full.cases if c.part in ("public", "positive", "events")
        ),
        generated=(),
    )
    journal = Journal(tmp_path / "journal.jsonl")
    passing = build_candidate(suite)
    record = evaluate_release(passing, suite, journal)
    assert record["decision"] == "pass"
    signed = sign(record, journal, "morgan", "2026-10-12T12:01:00+00:00")
    validate_release(signed, passing, suite, journal)
    denied = build_candidate(suite, name="deny-all")
    failed = evaluate_release(denied, suite, journal)
    assert failed["decision"] == "fail"
    assert "positive-100" in failed["failed_cases"]
    with pytest.raises(ValueError, match="only a journaled passing"):
        sign(failed, journal, "morgan", "2026-10-12T12:02:00+00:00")
    again = evaluate_release(denied, suite, journal)
    assert again["status"] == "unchanged_candidate_previously_failed"
    assert again["cases_run"] == 0 and again["decision"] == "fail"
    assert len(journal.read()) == 4


def test_positive_absence_waits_before_payment():
    data = _positive_data()
    request = next(r for r in data.requests if r.request_id == "positive-100")
    tools = EventTimeTools(data, event_log=snapshot_log(data))
    trace = Agent(tools).run(request)
    assert trace.decision.reason == "shipment_incomplete"
    assert not tools.payments


def test_view_cutoff_deduplication_and_arrival_order():
    data, log, request = opening()
    at = request.at
    shipped = next(
        e for e in log.entries if e.order_id == request.order_reference
    )
    copied = shipped.model_copy(
        update={"arrived_at": shipped.arrived_at + timedelta(minutes=5)}
    )
    reordered = EventLog(list(reversed(log.entries)) + [copied])
    before = shipment_as_of(log, request.order_reference, at)
    assert (
        before.events
        == shipment_as_of(reordered, request.order_reference, at).events
    )
    assert [e.status for e in before.events] == ["shipped"]
    after = shipment_as_of(
        log, request.order_reference, START + timedelta(hours=21)
    )
    assert [e.status for e in after.events] == ["shipped", "delivered"]
    assert before.observed_at == START + timedelta(hours=18, minutes=3)
    assert data.shipments[0].events == before.events


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "arrived"},
        {"arrived_at": START - timedelta(seconds=1)},
        {"at": START.replace(tzinfo=None)},
        {"arrived_at": START.replace(tzinfo=None)},
        {"event_id": " "},
    ],
)
def test_invalid_scan_is_refused(updates):
    row = dict(
        event_id="s1",
        order_id="o1",
        status="delivered",
        at=START,
        arrived_at=START,
    )
    with pytest.raises(ValueError):
        ScanEvent.model_validate({**row, **updates})


def test_conflicting_identity_and_out_of_order_append_are_refused():
    event = scan("o1", "shipped", START, START)
    log = EventLog([event])
    with pytest.raises(ValueError, match="conflict"):
        log.append(event.model_copy(update={"status": "delivered"}))
    later = event.model_copy(update={"arrived_at": START + timedelta(hours=2)})
    log.append(later)
    with pytest.raises(ValueError, match="arrival_order"):
        log.append(event)


def test_watermark_stalls_and_lateness_has_strict_boundary():
    log = EventLog([scan("feed", "shipped", START + MARGIN, START + MARGIN)])
    assert watermark(log, START) is None
    assert not complete_through(log, START, MARGIN, START)
    assert watermark(log, START + MARGIN) == START
    assert watermark(log, START + timedelta(days=9)) == START
    log.append(scan("equal", "delivered", START, START + MARGIN))
    log.append(
        scan("late", "delivered", START - timedelta(seconds=1), START + MARGIN)
    )
    assert [e.order_id for e in late_events(log)] == ["late"]
    with pytest.raises(ValueError, match="not_final"):
        first_complete(log, START + timedelta(seconds=1))


def test_absence_waits_for_watermark():
    for delivered, expected in (
        (True, "already_delivered"),
        (False, "eligible"),
    ):
        data, log, request = opening(delivered)
        tools = EventTimeTools(data, event_log=log)
        assert (
            Agent(tools).run(request).decision.reason == "shipment_incomplete"
        )
        final_at = first_complete(log, request.at + ALLOWED)
        tools.data = read_sources(data, final_at)
        trace = Agent(tools).run(request, at=final_at)
        assert trace.decision.reason == expected
        assert sum(p.amount_cents for p in tools.payments) == (
            0 if delivered else 3499
        )
        fast = LogShipmentTools(data, event_log=log)
        assert Agent(fast).run(request).decision.executed


def test_watermark_only_payment_is_corrected_default_denies():
    data, original, request = opening()
    delivery = scan(
        request.order_reference,
        "delivered",
        request.at - timedelta(hours=2),
        request.at + timedelta(hours=28),
    )
    log = EventLog(
        [delivery]
        + [
            e
            for e in original.entries
            if not (
                e.order_id == request.order_reference
                and e.status == "delivered"
            )
        ]
    )
    for allowed, expected in (
        (timedelta(0), "eligible"),
        (ALLOWED, "already_delivered"),
    ):
        tools = EventTimeTools(data, event_log=log, allowed_lateness=allowed)
        assert (
            Agent(tools).run(request).decision.reason == "shipment_incomplete"
        )
        ready = first_complete(log, request.at + allowed)
        tools.data = read_sources(data, ready)
        assert Agent(tools).run(request, at=ready).decision.reason == expected
        records = corrections(
            log, [PaymentRecord(p, "executed") for p in tools.payments]
        )
        assert len(records) == int(allowed == timedelta(0))


@pytest.mark.parametrize("reason", ["damaged", "changed_mind"])
def test_return_window_uses_scan_time_without_absence_wait(reason):
    from refund_agent.batch_stream.generate import dataset

    data, original, request = opening()
    made = request.at - timedelta(days=31, hours=1)
    delivery = scan(
        request.order_reference, "delivered", made, made + timedelta(days=2)
    )
    log = EventLog(
        [delivery]
        + [
            e
            for e in original.entries
            if e.order_id != request.order_reference
        ]
    )
    request = request.model_copy(update={"reason": reason})
    data = dataset((request,), log)
    results = [
        Agent(cls(data, event_log=log)).run(request).decision
        for cls in (LogShipmentTools, EventTimeTools)
    ]
    assert results[0] == results[1]
    assert (
        results[0].reason == "outside_return_window"
        and not results[0].executed
    )


def test_windows_panes_cutoff_and_full_day_identity():
    log = EventLog(exercise_scans())
    expected = (
        (24, 7, 6, None, "provisional"),
        (26, 8, 7, None, "on_time"),
        (29, 9, 8, None, "revised"),
        (32, 9, 8, 8, "final"),
        (33, 10, 9, 8, "final"),
    )
    for hour, fast, distinct, final, pane in expected:
        row = day_counts(
            log,
            START,
            START + timedelta(hours=hour),
            timedelta(hours=6),
            watermark_at=lambda t: t - timedelta(hours=2),
        )
        assert (row["fast"], row["distinct"], row["n_final"], row["pane"]) == (
            fast,
            distinct,
            final,
            pane,
        )
        assert row["fast"] == row["n_day"] + row["copies"] - row["misses"]
        assert row["arrival_day"] == 7
    assert row["corrections"] == ["s9"]
    assert key_run()["late_ids"] == ["s4", "s6", "s8", "s9"]


def test_replay_equals_live_views_without_tool_calls_and_keeps_ledger(
    monkeypatch,
):
    data, log, request = opening()
    tools = LogShipmentTools(data, event_log=log)
    Agent(tools).run(request)
    before = json.dumps(
        {k: v.model_dump(mode="json") for k, v in tools.ledger.items()},
        sort_keys=True,
    )

    def no_dispatch(*args):
        raise AssertionError("replay must never dispatch tools")

    monkeypatch.setattr(LogShipmentTools, "_dispatch", no_dispatch)
    live = Views()
    for offset, event in log.arrived_by(request.at + timedelta(days=2)):
        live = apply(live, offset, event)
    rebuilt = replay(log, through=request.at + timedelta(days=2))
    assert (
        rebuilt == live == replay(log, through=request.at + timedelta(days=2))
    )
    original = replay(log, through=request.at)
    assert [
        e.status
        for e in original.shipment(request.order_reference, request.at).events
    ] == ["shipped"]
    assert (
        restore(
            checkpoint(original), log, through=request.at + timedelta(days=2)
        )
        == rebuilt
    )
    assert before == json.dumps(
        {k: v.model_dump(mode="json") for k, v in tools.ledger.items()},
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="prefix"):
        restore(checkpoint(original), EventLog(), through=request.at)


def test_corrections_deduplicate_both_modes_without_ledger_mutation():
    from refund_agent.shadow.tools import ShadowTools

    data, log, request = opening()
    executed = LogShipmentTools(data, event_log=log)
    shadow = type("BatchStreamShadow", (ShadowTools, LogShipmentTools), {})(
        data, event_log=log
    )
    Agent(executed).run(request)
    Agent(shadow).run(request)
    before = executed.ledger
    payments = [PaymentRecord(p, "executed") for p in executed.payments] + [
        PaymentRecord(p, "shadow") for p in shadow.shadow_payments
    ]
    assert not corrections(log, payments, through=request.at)
    rows = corrections(log, payments + payments)
    assert [r.mode for r in rows] == ["executed", "shadow"]
    assert all(r.payment_id == executed.payments[0].receipt_id for r in rows)
    assert executed.ledger == before and not shadow.payments
    data, log, request = opening(False)
    honest = LogShipmentTools(data, event_log=log)
    Agent(honest).run(request)
    assert not corrections(
        log, [PaymentRecord(p, "executed") for p in honest.payments]
    )


def test_generator_reproducibility_counts_and_delay_bound():
    one, two = generate(), generate()
    assert one.fingerprint == two.fingerprint
    assert len(one.requests) == 700 and len(one.honest) == 140
    assert set(Counter(r.at.date() for r in one.requests).values()) == {50}
    ids = {}
    for event in one.log.entries:
        ids.setdefault(event.event_id, event)
    assert set(Counter(e.at.date() for e in ids.values()).values()) == {400}
    assert (
        max((e.arrived_at - e.at).total_seconds() for e in ids.values())
        <= 36 * 3600
    )
    for hours in (24, 33, 60, 70):
        row = day_counts(one.log, START, START + timedelta(hours=hours))
        assert row["fast"] == row["n_day"] + row["copies"] - row["misses"]


def test_candidate_and_timed_positive_cases_preserve_must_pay_total():
    from refund_agent.batch_stream.suite import (
        component_settings,
        extend,
        run_loop_case,
    )
    from refund_agent.release.candidate import build_candidate, build_tools
    from refund_agent.release.suite import (
        _CaseRun,
        build_suite,
        run_case,
        selected,
    )

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    assert (
        candidate.tools_composition["components"][0]["version"]
        == "batch-stream-1"
    )
    assert (
        build_candidate(
            extend(build_suite(trials=1), margin=timedelta(hours=8))
        ).fingerprint
        != candidate.fingerprint
    )
    assert (
        build_candidate(
            extend(build_suite(trials=1), allowed_lateness=timedelta(0))
        ).fingerprint
        != candidate.fingerprint
    )
    assert not isinstance(
        build_tools(
            _positive_data(), build_candidate(build_suite(trials=1)), "execute"
        ),
        EventTimeTools,
    )
    cases = [case for case in suite.cases if case.part == "positive"]
    results = {case.case_id: run_case(case, candidate) for case in cases}
    assert not any(row["failed"] for row in results.values()), results
    assert (
        sum(
            row["decision"]["amount_cents"]
            for row in results.values()
            if row["decision"]["executed"]
        )
        == 62598
    )
    timed = results["positive-100"]
    positive = next(case for case in cases if case.case_id == "positive-100")
    execution = _CaseRun(positive, candidate, "execute")
    assert (
        run_loop_case(
            case=positive,
            candidate=candidate,
            settings=component_settings(candidate),
            mode="execute",
            execution=execution.context,
        )
        is True
    )
    observed = execution._artifacts()
    assert (
        observed.tools.batch_stream_attempts[0]["reason"]
        == "shipment_incomplete"
    )
    assert observed.tools.batch_stream_attempts[-1] == timed["decision"]
    assert len(observed.tools.payments) == 1
    slots = [case for case in suite.cases if case.part == "events"]
    assert len(slots) == 3 and all(
        case.must_pass and selected(case, candidate) for case in slots
    )
    assert all(not run_case(case, candidate)["failed"] for case in slots)
    deny = build_candidate(suite, name="deny-all")
    assert all(run_case(case, deny)["failed"] for case in cases)


def test_component_requires_its_suite_log_and_unwaivable_cases():
    from dataclasses import replace

    from refund_agent.batch_stream.component import selector
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import (
        build_candidate,
        build_tools,
        run,
    )
    from refund_agent.release.suite import build_suite

    with pytest.raises(ValueError, match="components differ"):
        build_candidate(build_suite(trials=1), components=(selector(),))
    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    with pytest.raises(ValueError, match="event_log_required"):
        build_tools(_positive_data(), candidate, "execute")
    with pytest.raises(ValueError, match="must-pass case missing or changed"):
        suite.without("batch-stream-absence")
    changed = tuple(
        replace(c, must_pass=False)
        if c.handler == "batch-stream-absence"
        else c
        for c in suite.cases
    )
    with pytest.raises(ValueError, match="must-pass case missing or changed"):
        replace(suite, cases=changed)
    data, log, request = opening()
    observed = run(candidate, data, "own", event_log=log)
    assert observed.traces[0].decision.reason == "shipment_incomplete"
    assert not observed.tools.payments


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
@pytest.mark.parametrize("verifier", ["off", "independent"])
def test_composition_across_runtimes_modes_and_verifier(
    runtime, mode, verifier
):
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import (
        build_candidate,
        build_tools,
        run_request,
    )
    from refund_agent.release.suite import build_suite

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite, runtime=runtime, verifier=verifier)
    data, log, request = opening(False)
    tools = build_tools(data, candidate, mode, event_log=log)
    assert (
        run_request(candidate, tools, request).decision.reason
        == "shipment_incomplete"
    )
    at = first_complete(log, request.at + ALLOWED)
    tools.data = read_sources(data, at)
    terminal = run_request(candidate, tools, request, at=at)
    assert terminal.decision.reason == "eligible"
    recorded = tools.payments if mode == "execute" else tools.shadow_payments
    assert len(recorded) == 1 and recorded[0].amount_cents == 3499
    if mode == "shadow":
        assert not tools.payments
    if verifier == "independent":
        assert tools.verification_calls == 1


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
@pytest.mark.parametrize("reason", ["damaged", "not_delivered", "unknown"])
def test_selected_component_routes_every_tier3_decision_to_named_approver(
    runtime, mode, reason
):
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import (
        build_candidate,
        build_tools,
        run_request,
    )
    from refund_agent.release.suite import approve, build_suite
    from refund_agent.runtimes.swap import _project

    data = _positive_data()
    request = data.requests[0].model_copy(
        update={"amount_cents": 65000, "reason": reason}
    )
    data = _project(data, request)
    candidate = build_candidate(extend(build_suite(trials=1)), runtime=runtime)
    tools = build_tools(data, candidate, mode, event_log=snapshot_log(data))
    trace = run_request(candidate, tools, request)
    reviews = tools.shadow_reviews if mode == "shadow" else tools.reviews
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.tier == c.Tier.NAMED_APPROVER
    assert not trace.decision.executed and not tools.payments
    assert len(reviews) == 1
    review = next(iter(reviews.values()))
    assert review.required_role == "named_approver"
    assert review.proposal.evidence.policy.named_approver == "morgan"
    assert not approve(tools, request, "reviewer")
    again = run_request(candidate, tools, request)
    assert not again.decision.executed and not tools.payments


def test_removing_finality_guard_fails_behavior_tests(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from xml.etree import ElementTree

    from refund_agent.batch_stream import tools

    source = Path(tools.__file__).read_text()
    guard = (
        '        if state.request.reason == "not_delivered" and not self._final(state):\n'
        '            return "shipment_incomplete"\n'
    )
    assert source.count(guard) == 1
    mutant = tmp_path / "tools.py"
    mutant.write_text(source.replace(guard, ""))
    script = (
        "import sys, pytest; "
        "import refund_agent.release.suite; "
        "import refund_agent.batch_stream.tools as module; "
        "from pathlib import Path; "
        "exec(compile(Path(sys.argv[1]).read_text(), module.__file__, 'exec'), module.__dict__); "
        "raise SystemExit(pytest.main(sys.argv[2:]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(mutant),
            str(Path(__file__).resolve()),
            "-k",
            "direct_execution_requires_final_evidence or must_pass_execution_boundaries",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(tmp_path / "mutant-results.xml"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = ElementTree.parse(tmp_path / "mutant-results.xml")
    cases = {case.attrib["name"]: case for case in report.iter("testcase")}
    failures = {
        name: case.find("failure")
        for name, case in cases.items()
        if case.find("failure") is not None
    }
    assert not list(report.iter("error")), result.stdout + result.stderr
    assert set(failures) == {
        "test_direct_execution_requires_final_evidence[request]",
        "test_direct_execution_requires_final_evidence[watermark]",
        "test_direct_execution_requires_final_evidence[before-final]",
        "test_must_pass_execution_boundaries",
    }, result.stdout + result.stderr
    for name, failure in failures.items():
        assert failure.attrib["message"].startswith("AssertionError:")
        assert (
            "ledger_amount_and_single_receipt"
            if name == "test_must_pass_execution_boundaries"
            else "ok=True"
        ) in failure.attrib["message"]
    assert cases["test_direct_execution_requires_final_evidence[final]"].find("failure") is None
