"""Chapter 34 mathematics, deadline spend, correction modes and restart gates."""

from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb

import pytest

from refund_agent.observability_reliability.budget import (
    alert_threshold,
    binom_tail,
    burn_rate,
    every_event_false_page,
    min_count,
    remaining_lifetime,
    signed_budget,
    within_budget,
)
from refund_agent.observability_reliability.monitor import Monitor
from refund_agent.observability_reliability.objectives import (
    Observation,
    nearest_rank,
    share_within,
)
from refund_agent.observability_reliability.reasons import reason_class
from refund_agent.observability_reliability.silence import (
    SilenceDetector,
    threshold,
)

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def observation(i, *, bad=False, answered=True, promise=1):
    return Observation(
        request_id=f"observability-reliability-test-{i}",
        received_at=START,
        decided_at=START + timedelta(minutes=10) if answered else None,
        outcome="refer" if answered else None,
        reason="policy_not_in_bundle"
        if bad
        else "delivery_unknown"
        if answered
        else "shipment_incomplete",
        promise_hours=promise,
    )


def test_empty_measurement_does_not_report_compliance():
    with pytest.raises(ValueError, match="empty"):
        nearest_rank((), 0.99)
    with pytest.raises(ValueError, match="empty"):
        share_within((), 30)


def test_latency_and_referral_cost_capture_values():
    durations = (2,) * 99 + (2100,)
    assert sum(durations) / len(durations) == 22.98
    assert nearest_rank(durations, 0.99) == 2
    assert share_within(durations, 30) == 0.99
    assert nearest_rank((2,) * 98 + (2100,) * 2, 0.99) == 2100
    assert nearest_rank((320, 280, 360, 410, 295, 330, 310, 500), 0.95) == 500


@pytest.mark.parametrize(
    "n,p",
    [
        (0, Fraction(0)),
        (5, Fraction(1)),
        (17, Fraction(1, 50)),
        (40, Fraction(1, 100)),
    ],
)
def test_exact_tails_against_independent_binomial_sum(n, p):
    for k in range(n + 2):
        expected = sum(
            Fraction(comb(n, j)) * p**j * (1 - p) ** (n - j)
            for j in range(k, n + 1)
        )
        assert binom_tail(k, n, p) == expected


def test_closed_thresholds_and_every_event_inflation():
    assert min_count(17, 0.02, 0.01) == 3
    assert min_count(40, 0.01, 0.01) == 3
    assert min_count(40, 0.02, 0.01) == 4
    assert float(binom_tail(3, 40, 0.01)) == pytest.approx(0.0074973634)
    # Independent enumeration: no bad first event, at most one in the first
    # fifteen, and at most two in forty. There are 650 admissible bad pairs.
    p, q = Fraction(1, 100), Fraction(99, 100)
    expected = 1 - (
        q**40 + 39 * p * q**39 + (comb(25, 2) + 14 * 25) * p**2 * q**38
    )
    assert every_event_false_page(40, 0.01, 0.01) == expected
    assert every_event_false_page(40, 0.01, 0.01) > 3 * binom_tail(3, 40, 0.01)


def test_burn_forecast_lifetime_and_actual_traffic():
    assert burn_rate(1, 4, 0.02) == Fraction(25, 2)
    assert remaining_lifetime(
        Fraction(112, 3), Fraction(25, 9), 0.25
    ) == Fraction(1344, 25)
    assert remaining_lifetime(
        Fraction(112, 3), Fraction(50, 9), 0.25
    ) == Fraction(672, 25)
    assert [
        alert_threshold(share, 672, hours)
        for share, hours in ((0.02, 1), (0.05, 6), (0.1, 72))
    ] == [Fraction(336, 25), Fraction(28, 5), Fraction(14, 15)]
    assert within_budget(56, 0.56, 0.005) == Fraction(199, 200) ** 56


@pytest.mark.parametrize(
    "n,allowance,p",
    [
        (56, Fraction(14, 25), Fraction(1, 200)),
        (56, Fraction(14, 25), Fraction(1, 50)),
        (1120, Fraction(56, 5), Fraction(1, 200)),
        (1120, Fraction(56, 5), Fraction(1, 50)),
        (1120, Fraction(56, 5), Fraction(1, 100)),
        (1867, Fraction(112, 3), Fraction(1, 50)),
    ],
)
def test_forecast_headroom_matches_independent_closed_form(n, allowance, p):
    # Sum lower-tail integer masses with one common denominator, independently
    # of budget.py's adjacent-mass upper-tail recurrence.
    a, denominator = p.numerator, p.denominator
    allowed = allowance.numerator // allowance.denominator
    numerator = sum(
        comb(n, j) * a**j * (denominator - a) ** (n - j)
        for j in range(allowed + 1)
    )
    assert within_budget(n, allowance, p) == Fraction(
        numerator, denominator**n
    )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_durations_and_probabilities(value):
    with pytest.raises(ValueError):
        nearest_rank((2, value), 0.95)
    with pytest.raises(ValueError):
        binom_tail(1, 3, value)


def test_signature_and_threshold_basis():
    record = signed_budget(
        "observability-reliability-test", "fingerprint", START
    )
    with pytest.raises(ValueError, match="decision-owner"):
        signed_budget("test", "fingerprint", START, owner="absent")
    with pytest.raises(ValueError, match="basis"):
        record.objectives[0].model_validate(
            {**record.objectives[0].model_dump(), "basis": ""}
        )


def test_signed_definition_is_immutable_and_reloadable(tmp_path):
    from refund_agent.observability_reliability.budget import ErrorBudgetRecord

    record = signed_budget("definition", "fingerprint", START)
    path = tmp_path / "observability-reliability-error-budget.json"
    record.write(path)
    original = path.read_bytes()
    assert ErrorBudgetRecord.model_validate_json(original) == record
    record.write(path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="cannot be replaced"):
        signed_budget("definition", "different", START).write(path)
    assert path.read_bytes() == original


def test_closed_count_page_latches_across_restart(tmp_path):
    record = signed_budget("page-restart", "fingerprint", START)
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(record, path)
    for i in range(17):
        monitor.observe(observation(i, bad=i < 3))
    closed_at = START + timedelta(hours=6)
    assert len(monitor.close_windows(closed_at)) == 1
    restarted = Monitor(record, path)
    assert restarted.alerts == monitor.alerts
    assert restarted.close_windows(closed_at + timedelta(seconds=1)) == ()
    assert len(restarted.alerts) == 1


def test_closed_window_keeps_held_deadlines_across_reload(tmp_path):
    record = signed_budget("held-window", "fingerprint", START)
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(record, path)
    for i in range(17):
        monitor.observe(observation(i, answered=False, promise=48))
    assert monitor.close_windows(START + timedelta(hours=6)) == ()
    monitor = Monitor(record, path)
    monitor.tick(START + timedelta(hours=54))
    page = next((p for p in monitor.alerts if p.rule == "timeliness"), None)
    assert page is not None, "closed boundary discarded 17 pending deadlines"
    assert (page.count, page.threshold) == (17, 3)
    assert page.window_start == START
    assert page.window_end == START + timedelta(hours=6)
    restarted = Monitor(record, path)
    assert restarted.close_windows(START + timedelta(hours=60)) == ()
    assert len([p for p in restarted.alerts if p.rule == "timeliness"]) == 1


def test_unclassified_observation_is_refused_before_appending(tmp_path):
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(
        signed_budget("unknown-reason", "fingerprint", START), path
    )
    monitor.observe(observation(0))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unclassified"):
        monitor.observe(
            observation(1).model_copy(update={"reason": "invented"})
        )
    assert path.read_bytes() == before
    assert "observability-reliability-test-1" not in monitor.observations


def test_clock_finality_requires_boolean_before_appending(tmp_path):
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(signed_budget("typed-clock", "fingerprint", START), path)
    monitor.observe(observation(0))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="boolean"):
        monitor.tick(START + timedelta(hours=2), outcomes_final="false")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "n,bad,pages", [(17, 2, 0), (17, 3, 1), (40, 3, 0), (40, 4, 1)]
)
def test_pages_only_once_at_close_using_actual_eligible_count(n, bad, pages):
    monitor = Monitor(signed_budget("count", "fingerprint", START))
    for i in range(n):
        monitor.observe(observation(i, bad=i < bad))
    monitor.close_windows(START + timedelta(hours=6, microseconds=-1))
    assert not monitor.alerts
    monitor.close_windows(START + timedelta(hours=6))
    assert len(monitor.alerts) == pages
    monitor.close_windows(START + timedelta(hours=6, minutes=1))
    assert len(monitor.alerts) == pages


def test_unanswered_deadline_consumes_budget_and_restart_callbacks(tmp_path):
    record = signed_budget("restart", "fingerprint", START)
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(record, path)
    for i in range(38):
        monitor.observe(observation(i, answered=False))
    ended = []
    monitor.register_budget_spent(
        "canary.end", lambda event: ended.append(event.reason)
    )
    monitor.tick(START + timedelta(hours=1))
    assert not ended
    monitor.tick(START + timedelta(hours=1, microseconds=1))
    assert ended == ["budget_spent"]
    monitor.tick(START + timedelta(hours=2))
    assert ended == ["budget_spent"]
    restarted = Monitor(record, path)
    stage6 = []
    restarted.register_budget_spent(
        "canary.end", lambda event: stage6.append(event.reason)
    )
    restarted.register_budget_spent(
        "stage6.reversal", lambda event: stage6.append(event.objectives)
    )
    assert stage6 == ["budget_spent", ("timeliness",)]
    assert restarted.status() == monitor.status()
    with pytest.raises(ValueError, match="window differs"):
        Monitor(signed_budget("other", "fingerprint", START), path)


def test_realized_ratio_can_miss_without_spending_signed_allowance():
    record = signed_budget("low-traffic", "fingerprint", START)
    monitor = Monitor(record)
    for i in range(10):
        monitor.observe(observation(i, bad=i == 0))
    monitor.tick(record.end, outcomes_final=True)
    status = monitor.status()["objectives"]["timeliness"]
    assert (
        status["bad"] == 1
        and not status["spent"]
        and status["realized_pass"] is False
    )


def test_exhaustion_cannot_clear_when_late_records_repair_counts(tmp_path):
    record = signed_budget("irreversible-spend", "fingerprint", START)
    path = tmp_path / "observability-reliability-budget.jsonl"
    monitor = Monitor(record, path)
    for i in range(38):
        monitor.observe(observation(i, answered=False))
    monitor.tick(START + timedelta(hours=2))
    assert monitor.status()["spent"]
    for i in range(38):
        monitor.observe(observation(i))
    assert monitor.status()["objectives"]["timeliness"]["bad"] == 0
    assert Monitor(record, path).status()["spent"]


def test_reason_classes_are_closed():
    assert reason_class("delivery_unknown") == "policy"
    assert reason_class("policy_not_in_bundle") == "system"
    assert reason_class("shipment_incomplete") == "hold"
    assert reason_class("shipment.status:stale_shipment") == "system"
    with pytest.raises(ValueError, match="unclassified"):
        reason_class("invented")
    with pytest.raises(ValueError, match="changes"):
        reason_class("delivery_unknown", extra={"delivery_unknown": "system"})


def test_silence_boundary_latches_and_rate_arithmetic():
    detector = SilenceDetector(400 / 72, 2)
    detector.arrival(START)
    assert detector.check(START + timedelta(hours=2, microseconds=-1)) is None
    page = detector.check(START + timedelta(hours=2))
    assert page.window_end == START + timedelta(hours=2)
    assert detector.check(START + timedelta(hours=3)) is None
    detector.arrival(START + timedelta(hours=4))
    assert detector.check(START + timedelta(hours=6)) is not None
    assert threshold(400 / 72, 672, 0.1) == pytest.approx(1.895, abs=0.0005)


def test_d6_pairs_use_mature_labels_and_exclude_unpaid_person_denials():
    from refund_agent.observability_reliability.objectives import (
        DenialPair,
        paired_denials,
    )

    mature = START + timedelta(days=30)
    rows = (
        DenialPair(
            request_id="paid",
            agent_denied=True,
            baseline_denied=False,
            person_paid_at=START,
            found_false=False,
            label_mature_at=mature,
        ),
        DenialPair(
            request_id="unpaid",
            agent_denied=True,
            baseline_denied=True,
            person_paid_at=None,
            found_false=False,
            label_mature_at=mature,
        ),
    )
    assert paired_denials(rows, now=START)["pending"] == 2
    result = paired_denials(rows, now=mature)
    assert (
        result["agent"] == 1
        and result["baseline"] == 0
        and result["pass"] is False
    )
    assert result["mature"] == 2 and result["pending"] == 0
    with pytest.raises(ValueError, match="distinct"):
        paired_denials(rows + rows, now=mature)


def test_monitor_ingests_written_span_and_checks_its_record():
    from refund_agent.observability_reliability.spans import Span

    obs = observation(1, bad=True)
    span = Span(
        trace_id="trace-" + obs.request_id,
        run_id="run-1",
        attempt=1,
        span_id="observability-reliability-read-span",
        parent_id=None,
        name="run",
        start=obs.decided_at,
        end=obs.decided_at,
        status="ok",
        reason=obs.reason,
        requester_id="account",
    )
    monitor = Monitor(signed_budget("span-input", "fingerprint", START))
    with pytest.raises(ValueError, match="requires"):
        monitor.ingest_span(span)
    monitor.ingest_span(span, observation=obs)
    assert monitor.status()["objectives"]["timeliness"]["bad"] == 1


def test_reload_rejects_backwards_clock_and_concurrent_writer(tmp_path):
    import json

    record = signed_budget("log-proof", "fingerprint", START)
    path = tmp_path / "observability-reliability-budget.jsonl"
    first = Monitor(record, path)
    second = Monitor(record, path)
    first.observe(observation(1))
    with pytest.raises(ValueError, match="reload"):
        second.observe(observation(2))
    first.tick(START + timedelta(hours=2))
    forged = {
        "entry": first._rows + 1,
        "binding": record.fingerprint,
        "kind": "tick",
        "data": {"at": START.isoformat(), "outcomes_final": False},
    }
    with path.open("a") as stream:
        stream.write(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="backwards"):
        Monitor(record, path)


def test_literal_reason_emissions_have_a_class():
    import ast
    from pathlib import Path

    from refund_agent.observability_reliability.reasons import REASON_CLASS

    root = Path(__file__).resolve().parents[1] / "refund_agent"
    codes = set()
    for path in (
        root / "rules.py",
        root / "tools.py",
        root / "agent/__init__.py",
        root / "runtimes/adapter.py",
        root / "canary/cloud_native_deployment_binding.py",
    ):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            values = [kw.value for kw in node.keywords if kw.arg == "reason"]
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "decision"
                and len(node.args) > 2
            ):
                values.append(node.args[2])
            for value in values:
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value
                ):
                    codes.add(value.value)
    assert codes <= REASON_CLASS.keys()
    assert reason_class("model_error:assess:RuntimeError") == "system"
    assert reason_class("tool_error:refund.execute:OSError") == "system"


@pytest.mark.parametrize(
    "code",
    [
        "canary_shadow",
        "canary_running",
        "canary_paused",
        "canary_ended",
        "canary_rollback",
        "canary_amount",
        "canary_outside_cohort",
        "canary_daily_cap",
    ],
)
def test_canary_state_and_capacity_refusals_are_system_caused(code):
    assert reason_class(code) == "system"


def test_tracing_preserves_unknown_tool_refusal():
    from refund_agent import contracts as c
    from refund_agent.batch_stream.generate import opening
    from refund_agent.observability_reliability.spans import (
        SpanStore,
        TracedTools,
    )
    from refund_agent.tools import Tools

    data, _, request = opening()
    call = c.ToolCall(name="unavailable.operation")
    state = c.State(request=request, trace_id="trace-" + request.request_id)
    expected = Tools(data).call(call, state.model_copy(deep=True))
    traced = TracedTools(data, span_store=SpanStore(), traced_mode="execute")
    traced.begin_trace(request)
    actual = traced.call(call, state.model_copy(deep=True))
    assert actual == expected
    assert any(s.reason == "unknown_tool" for s in traced.span_store.entries)


def test_region_finality_advances_when_the_same_log_is_appended():
    from refund_agent import contracts as c
    from refund_agent.batch_stream.generate import opening, scan
    from refund_agent.batch_stream.log import EventLog
    from refund_agent.batch_stream.tools import EventTimeTools
    from refund_agent.observability_reliability.silence import (
        RegionWatermarkTools,
    )

    data, full, request = opening(delivered=False)
    log = EventLog(e for e in full.entries if e.arrived_at <= request.at)
    cls = type("RegionalTestTools", (RegionWatermarkTools, EventTimeTools), {})
    tools = cls(data, event_log=log, region_finality=True)
    state = c.State(
        request=request,
        trace_id="trace-" + request.request_id,
        execution_at=request.at,
    )
    assert tools._region_final(state) is False
    at = request.at + timedelta(hours=36)
    log.append(scan(request.order_reference, "shipped", at, at, n=1))
    state.execution_at = at
    assert tools._region_final(state) is True


def test_run_span_uses_the_cores_logical_clock(monkeypatch):
    from refund_agent.batch_stream.generate import opening
    from refund_agent.batch_stream.tools import EventTimeTools
    from refund_agent.observability_reliability.component import before_run
    from refund_agent.observability_reliability.spans import (
        SpanStore,
        TracedTools,
    )

    data, log, request = opening()
    cls = type("TimedTraceTools", (TracedTools, EventTimeTools), {})
    tools = cls(
        data, event_log=log, span_store=SpanStore(), traced_mode="execute"
    )
    logical_at = request.at + timedelta(hours=1)
    monkeypatch.setattr(tools, "run_time", lambda request: logical_at)
    before_run(
        tools=tools,
        request=request,
        at=None,
        runtime_settings=None,
        candidate=None,
        settings={},
    )
    assert tools._trace_root["start"] == logical_at
