# ruff: noqa: E402 -- initialize the fixed core before component imports.
"""Incident reconstruction uses reviewed records, not imagined journal events."""

from datetime import timedelta

import pytest

from refund_agent.release.candidate import (
    build_candidate,
    build_tools,
    run_request,
)
from refund_agent.release.journal import Journal
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite())

# isort: split
from refund_agent.agent import Agent
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.batch_stream.generate import opening, scan
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.replay import CorrectionRecord, PaymentRecord
from refund_agent.canary.cloud_native_deployment_stop import StopRecord
from refund_agent.observability_reliability.component import selector
from refund_agent.observability_reliability.incident import reconstruct
from refund_agent.sources.data_systems_release import SELECTOR
from refund_agent.tools import Tools


def records():
    suite = build_suite(
        trials=1,
        components=[
            SELECTOR,
            event_selector(),
            selector(region_finality=False),
        ],
    )
    candidate = build_candidate(suite)
    data, log, request = opening(delivered=False)
    tools = build_tools(data, candidate, "shadow", event_log=log)
    run_request(candidate, tools, request)
    row = tools.lineage[0]
    outside = row.model_copy(
        update={
            "request": request.model_copy(update={"request_id": "outside"}),
            "trace_id": "trace-outside",
            "decision_at": request.at - timedelta(hours=4),
        }
    )
    log = EventLog(
        (
            scan(
                request.order_reference,
                "shipped",
                request.at - timedelta(hours=3),
                request.at - timedelta(hours=3),
            ),
            scan(
                request.order_reference,
                "shipped",
                request.at,
                request.at + timedelta(hours=3),
                n=1,
            ),
            scan(
                request.order_reference,
                "delivered",
                request.at + timedelta(hours=1),
                request.at + timedelta(hours=3),
                n=2,
            ),
        )
    )
    return {
        "spans": (),
        "audit": (),
        "lineage": (row, outside),
        "journal": None,
        "event_log": log,
        "corrections": (),
        "stops": None,
        "payments": (),
        "snapshots": (tools.record_store.export(),),
        "window_start": request.at - timedelta(days=1),
        "window_end": request.at + timedelta(days=1),
        "budget_status": {},
    }


def test_reused_record_does_not_pull_decisions_outside_the_incident():
    inputs = records()
    incident = reconstruct(**inputs)
    assert incident.affected_decisions == (inputs["lineage"][0].request.request_id,)


def test_cohort_stop_joins_real_receipt_evidence_and_exposure():
    inputs = records()
    data, _, request = opening(delivered=False)
    trace = Agent(Tools(data)).run(request)
    payment = trace.receipt
    assert payment is not None and payment.evidence is not None
    at = request.at + timedelta(hours=3)
    change = CorrectionRecord(
        payment.receipt_id,
        "executed",
        request.request_id,
        payment.order_id,
        "observability-reliability-correcting-scan",
        request.at - timedelta(hours=1),
        at,
        "shipped",
        "delivered",
    )
    stop = StopRecord(
        stop_id="observability-reliability-stop",
        canary_id="incident",
        at=at,
        trigger="correction",
        payment_id=payment.receipt_id,
        payments_after_trigger=0,
        cents_after_trigger=0,
        outcome="paused",
    )
    inputs.update(
        stops=(stop,),
        cohort_payments=(PaymentRecord(payment, "executed"),),
        cohort_corrections=(change,),
    )
    incident = reconstruct(**inputs)
    assert incident.cash_loss_cents == payment.amount_cents
    assert incident.cohort_exposure == 1
    joined = next(r for r in incident.timeline if r["name"] == "canary_stop:correction")
    assert joined["trace_id"] == payment.evidence.trace_id
    assert sum(r["scans"] for r in incident.arrivals_by_region_hour) == 3
    inputs["stops"] = (stop.model_copy(update={"payments_after_trigger": 1}),)
    with pytest.raises(ValueError, match="exposure differs"):
        reconstruct(**inputs)


def test_actual_release_journal_is_read_without_inventing_deployments(
    tmp_path,
):
    inputs = records()
    at = inputs["lineage"][0].decision_at.isoformat()
    journal = Journal(tmp_path / "observability-reliability-journal.jsonl")
    journal.append(
        {
            "version": "refund-release-record-2",
            "at": at,
            "status": "evaluated",
            "decision": "pass",
            "record_id": "evaluated-record",
            "candidate_fingerprint": "candidate",
            "software_fingerprint": "candidate",
        }
    )
    journal.append(
        {
            "event": "sign",
            "record_id": "evaluated-record",
            "candidate_fingerprint": "candidate",
            "software_fingerprint": "candidate",
            "signature": {
                "owner": "morgan",
                "at": at,
                "kind": "recorded sign-off; not cryptographic",
            },
        }
    )
    inputs["journal"] = journal.read()
    incident = reconstruct(**inputs)
    assert incident.releases is None
    assert [r["event"] for r in incident.release_history] == [
        "evaluation",
        "sign",
    ]
    assert all(r["candidate_fingerprint"] == "candidate" for r in incident.release_history)


def test_incident_reads_signed_deployments_from_the_bundle_registry(tmp_path):
    from refund_agent.promotion.mlops_mle_bundle import Component, VersionBundle
    from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH
    from refund_agent.promotion.mlops_mle_record import PromotionRecord, sign
    from refund_agent.promotion.mlops_mle_registry import Registry

    inputs = records()
    at = inputs["lineage"][0].decision_at
    bundle = VersionBundle(
        **{
            name: Component(name=name, items={"versions": ["v1"]} if name == "policy" else {})
            for name in ("model", "prompt", "tools", "policy", "index")
        },
        refs={},
    )
    registry = Registry(REGISTRY_PATH, str(tmp_path))
    promotion = PromotionRecord(
        candidate_fingerprint=bundle.fingerprint,
        incumbent_fingerprint=None,
        from_stage="offline",
        to_stage="shadow",
        changes=(),
        requirements={},
        status="promoted",
        unmet=(),
        rollback_target=None,
    )
    inputs["lineage"] = tuple(
        row.model_copy(update={"bundle": bundle.fingerprint}) for row in inputs["lineage"]
    )
    from pathlib import Path

    import refund_agent.promotion.mlops_mle_registry as module
    from refund_agent.release import agent_evaluation_resources as resources

    allowed = resources.ACCESS[(module.__name__, Path(module.__file__).resolve())][0]
    with resources.scope(allowed, {REGISTRY_PATH: "append"}):
        registry.append(bundle)
        registry.append(sign(promotion, "morgan", at))
        registry.append(promotion.model_copy(update={"status": "held", "unmet": ("release",)}))
        incident = reconstruct(**inputs, registry=registry)
    assert incident.bundles == (bundle.fingerprint,)
    assert incident.unstamped_lineage == 0 and incident.releases == 1
    assert [row["kind"] for row in incident.deployment_history] == ["promotion"]
    assert "bundle stamps until chapter 33 integration" not in incident.unknowns


def test_continuous_watermark_lag_includes_prearrival_and_end_limits():
    from refund_agent.observability_reliability.fortnight import START
    from refund_agent.observability_reliability.incident import watermark_lag_range

    log = EventLog(
        scan("lag", "shipped", START + timedelta(hours=i), START + timedelta(hours=i), n=i)
        for i in range(3)
    )
    assert watermark_lag_range(log, START, START + timedelta(hours=3)) == (9, 10)
    assert watermark_lag_range(log, START, START + timedelta(hours=2, minutes=1)) == (9, 10)


@pytest.mark.parametrize(
    "missing,expected",
    [
        ({0: {3, 4, 5}, 2: {8, 9, 10}}, [(0, 4), (2, 9)]),
        ({0: {2, 3, 6, 7}}, [(0, 3), (0, 7)]),
        ({}, []),
    ],
)
def test_fortnight_replays_clean_regions_multiple_silences_and_recovery(missing, expected):
    from dataclasses import replace

    from refund_agent.observability_reliability.fortnight import (
        START,
        Silence,
        make_stream,
        run_shadow_fortnight,
    )
    from refund_agent.observability_reliability.silence import region_of

    orders = {}
    for i in range(100):
        order = f"observability-reliability-detector-{i}"
        orders.setdefault(region_of(order), order)
    log = EventLog(
        scan(
            orders[region],
            "shipped",
            START + timedelta(hours=hour),
            START + timedelta(hours=hour),
            n=hour,
        )
        for hour in range(337)
        for region in range(3)
        if hour not in missing.get(region, set())
    )
    # Deliberately describes a clean region. Detection must use only the log.
    decoy = Silence(
        1,
        START + timedelta(hours=1),
        START + timedelta(hours=2),
        "unused-request",
        "unused-event",
    )
    stream = replace(make_stream(), requests=(), log=log, silence=decoy)
    suite = build_suite(
        trials=1,
        components=[
            SELECTOR,
            event_selector(),
            selector(region_finality=False),
        ],
    )
    run = run_shadow_fortnight(build_candidate(suite), stream, fix=False)
    pages = run.get("pages")
    assert pages is not None, "fortnight did not replay all regional detectors"
    observed = [
        (
            r["region"],
            int((r["alert"].window_end - START).total_seconds() / 3600),
        )
        for r in pages
    ]
    assert observed == expected
