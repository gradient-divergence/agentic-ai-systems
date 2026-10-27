"""Arrival-order holds and exact model arithmetic, independent of release gates."""

from datetime import datetime, timedelta, timezone
from fractions import Fraction

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

# isort: split
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.canary.cloud_native_deployment_component import SELECTOR
from refund_agent.canary.cloud_native_deployment_exposure import (
    clean_hours,
    exposure,
    further_requests,
    loss_cents,
    zero_failure_bound,
)
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
from refund_agent.canary.cloud_native_deployment_rehearsal import fixture, run_rehearsal
from refund_agent.canary.cloud_native_deployment_window import (
    live_event_log,
    next_run_at,
    run_window,
    sign_window,
)
from refund_agent.data.generate import Dataset
from refund_agent.data.live import LiveStream


def test_generic_rehearsal_has_no_chapter32_plant_and_cannot_admit_a_start():
    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    cid = "cloud-native-deployment-external-test"
    pieces = [fixture(cid, f"external-{i}", start + timedelta(days=i))[0] for i in range(3)]
    data = Dataset(orders=sum((p.orders for p in pieces), []), requests=sum((p.requests for p in pieces), []),
                   shipments=sum((p.shipments for p in pieces), []), policies=pieces[0].policies,
                   retrievals=sum((p.retrievals for p in pieces), []), correct=[])
    stream = LiveStream(data, {"seed": 7, "start": start.isoformat(), "days": 3, "per_day": 1})
    suite = build_suite(components=[event_selector(), SELECTOR])
    candidate = build_candidate(suite)
    plan = sign_plan(CanaryPlan(canary_id=cid, candidate_fingerprint=candidate.fingerprint,
                               suite_fingerprint=suite.fingerprint, rollback_target_fingerprint=candidate.fingerprint,
                               requirement_ref="rehearsal-only", rehearsal_ref=cid), "morgan", start)
    result = run_rehearsal(plan, stream, 3, [], "4h", candidate, candidate=candidate)
    assert result.triggering_payment_cents == 0 and result.exposure_stop_at is None
    assert result.corrections == 0 and result.final_outcome == "running"
    assert result.verdict == "observed" and not result.restart_fresh_process


def test_held_clock_advances_after_global_finality_and_refuses_an_ended_log():
    from refund_agent.batch_stream.view import ALLOWED, first_complete

    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    data, request = fixture("cloud-native-deployment-clock-test", "held-clock", start)
    stream = LiveStream(data, {"seed": 7, "start": start.isoformat(), "days": 1, "per_day": 1})
    log = live_event_log(stream)
    final = first_complete(log, request.at + ALLOWED)
    assert next_run_at(log, request.at, final) > final
    with pytest.raises(ValueError, match="ends before"):
        next_run_at(log, request.at, max(e.arrived_at for e in log.entries))


def test_first_arrivals_keep_held_case_and_complete_out_of_order():
    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    pieces = [fixture("cloud-native-deployment-window-test", tag, start + timedelta(hours=i))[0]
              for i, tag in enumerate(("held-first", "complete-second", "replacement-forbidden"))]
    first = pieces[0]
    request = first.requests[0].model_copy(update={"reason": "not_delivered"})
    first = first.model_copy(update={
        "requests": [request],
        "shipments": [first.shipments[0].model_copy(update={
            "events": [e for e in first.shipments[0].events if e.status == "shipped"],
        })],
    })
    pieces[0] = first
    data = Dataset(orders=sum((p.orders for p in pieces), []), requests=sum((p.requests for p in pieces), []),
                   shipments=sum((p.shipments for p in pieces), []), policies=first.policies,
                   retrievals=sum((p.retrievals for p in pieces), []), correct=[])
    stream = LiveStream(data, {"seed": 7, "start": start.isoformat(), "days": 3, "per_day": 1})
    suite = build_suite(components=[event_selector(), SELECTOR])
    result = run_window(build_candidate(suite), start, 2, stream=stream)
    assert result.request_ids == (pieces[0].requests[0].request_id, pieces[1].requests[0].request_id)
    assert result.completion_order == result.request_ids[::-1]
    assert result.comparisons[0]["agent"] == "approve"
    assert result.comparisons[0]["baseline"] == "approve"
    assert datetime.fromisoformat(result.comparisons[0]["terminal_at"]) >= start + timedelta(hours=36)
    assert result.generated_label_shortcut and result.signed_by is None
    with pytest.raises(ValueError, match="passing"):
        sign_window(result, "morgan", result.completed_at)


@pytest.mark.parametrize("seconds,outcome", [(0, "approve"), (1, "deny")])
def test_terminal_comparators_use_execution_clock_at_return_boundary(seconds, outcome):
    from refund_agent.batch_stream.generate import read_sources
    from refund_agent.canary import cloud_native_deployment_window as window

    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    data, request = fixture("cloud-native-deployment-boundary", "return-boundary", start)
    at = start + timedelta(hours=1, seconds=seconds)
    delivered = start + timedelta(hours=1) - timedelta(days=45)
    shipment = data.shipments[0].model_copy(update={
        "observed_at": at - timedelta(seconds=120),
        "events": [data.shipments[0].events[-1].model_copy(update={"at": delivered})],
    })
    data = read_sources(data, at).model_copy(update={"shipments": [shipment]})
    person, base = window.terminal_comparators(data, request, at)
    assert person.outcome.value == base.outcome.value == outcome
    assert request.at == start
    if seconds:
        assert person.reason == base.reason == "outside_return_window"


@pytest.mark.parametrize("hours,further,loss,outward", [
    (4, Fraction(5, 9), Fraction(69986, 9), 7777),
    (24, Fraction(10, 3), Fraction(64987, 3), 21663),
    (36, Fraction(5), Fraction(29994), 29994),
])
def test_exact_exposure_includes_trigger_and_bounds_round_outward(hours, further, loss, outward):
    from math import ceil

    share, rate = Fraction(1, 20), Fraction(25, 9)
    assert further_requests(share, rate, Fraction(hours)) == further
    assert loss_cents(share, rate, Fraction(hours)) == loss
    assert ceil(loss) == outward


def test_clean_time_projection_and_detection_closed_forms():
    row = exposure(Fraction(36))
    assert row["clean"] == {"1/20": Fraction(10944, 5), "1/10": Fraction(5562, 5), "1/4": Fraction(23328, 50)}
    assert row["projection"] == Fraction(265, 3)
    assert row["quarter_daily_loss"] == Fraction(264947, 3)
    assert row["fault_seen"][88] == 1 - Fraction(999, 1000) ** 88
    assert row["fault_seen"][299] == 1 - Fraction(999, 1000) ** 299
    assert clean_hours(Fraction(1, 20), Fraction(5), 299, Fraction(36)) == 1232


def test_zero_failure_bound_matches_chapter_three_function():
    from demos.data_science_strategy_demo import zero_failure_bound as chapter_three

    for n in (1, 88, 299, 10000):
        assert zero_failure_bound(n) == pytest.approx(chapter_three(n), abs=1e-14)
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            zero_failure_bound(bad)
