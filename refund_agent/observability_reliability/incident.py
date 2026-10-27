"""Chapter 34 reconstructs from records; the planted fault is never an input."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import groupby
from math import ceil

from refund_agent import contracts as c
from refund_agent.batch_stream.view import MARGIN, late_events, watermark
from refund_agent.canary.cloud_native_deployment_stop import StopRecord
from refund_agent.observability_reliability.objectives import payment_evidence
from refund_agent.observability_reliability.silence import region_of
from refund_agent.sources.data_systems_lineage import (
    RecordStore,
    decisions_using,
)


class IncidentRecord(c.Record):
    region: int
    last_arrival: datetime
    batch_at: datetime
    batch_scans: int
    observed_gap_hours: float
    corrections: int
    corrected_modes: dict[str, int]
    affected_decisions: tuple[str, ...]
    held_after_batch: int
    policy_referrals: int
    cash_loss_cents: int
    shadow_exposure_cents: int
    illustrative_review_cost_cents: int
    cohort_exposure: int | None
    cohort_exposure_cents: int | None
    cohort_corrections: int
    cohort_stops: tuple[dict, ...]
    arrivals_by_region_hour: tuple[dict, ...]
    bundles: tuple[str, ...]
    unstamped_lineage: int
    releases: int | None
    release_history: tuple[dict, ...]
    deployment_history: tuple[dict, ...]
    watermark_lag_hours: tuple[float, float]
    late_share: float
    shipment_ages_seconds: tuple[float, ...]
    timeline: tuple[dict, ...]
    budget_status: dict
    unknowns: tuple[str, ...]
    actions: tuple[dict, ...]


def reconstruct(
    spans,
    audit,
    lineage,
    journal,
    event_log,
    corrections,
    stops,
    *,
    payments,
    snapshots,
    window_start,
    window_end,
    budget_status,
    cohort_payments=(),
    cohort_corrections=(),
    cohort_spans=(),
    registry=None,
) -> IncidentRecord:
    # These are real joins, not a list of unverified identifiers. Audit content
    # remains in its store; the reconstruction carries only its identifier.
    audit_by_id = {a.audit_id: a for a in audit}
    for span in spans:
        if span.name == "store:audits":
            for key in span.writes:
                if key not in audit_by_id or audit_by_id[key].trace_id != span.trace_id:
                    raise ValueError("span audit join is missing or mismatched")
    unique = {e.event_id: e for e in event_log.entries}
    arrivals = defaultdict(list)
    for event in unique.values():
        if window_start <= event.arrived_at < window_end:
            arrivals[region_of(event.order_id), event.arrived_at].append(event)
    if not arrivals:
        raise ValueError("no incident event records")
    (region, batch_at), batch = max(arrivals.items(), key=lambda item: len(item[1]))
    before = [at for (r, at) in arrivals if r == region and at < batch_at]
    if not before:
        raise ValueError("no arrival before the batch")
    last = max(before)
    store = RecordStore()
    for snapshot in snapshots:
        for record in snapshot.values():
            store.put(record)
    for row in lineage:
        store.append(row)
    referenced = {
        ref.fingerprint
        for row in lineage
        if last <= row.decision_at < batch_at and region_of(row.request.order_reference) == region
        for ref in row.inputs
        if ref.source == "shipment" and ref.used
    }
    affected = tuple(
        sorted(
            {
                row.request.request_id
                for key in referenced
                for row in decisions_using(key, store)
                if last <= row.decision_at < batch_at
                and region_of(row.request.order_reference) == region
            }
        )
    )
    corrected = {(r.mode, r.payment_id) for r in corrections}
    wrong = tuple(p for p in payments if (p.mode, p.receipt.receipt_id) in corrected)
    cash = sum(p.receipt.amount_cents for p in wrong if p.mode == "executed")
    shadow = sum(p.receipt.amount_cents for p in wrong if p.mode == "shadow")
    cohort = {p.receipt.receipt_id: p.receipt for p in cohort_payments}
    if any(p.mode != "executed" for p in cohort_payments):
        raise ValueError("cohort exposure requires executed receipts")
    for change in cohort_corrections:
        payment = cohort.get(change.payment_id)
        if (
            change.mode != "executed"
            or payment is None
            or payment.request.request_id != change.request_id
            or payment.order_id != change.order_id
        ):
            raise ValueError("cohort correction does not join its receipt")
    cohort_bad = {r.payment_id for r in cohort_corrections}
    cash += sum(cohort[key].amount_cents for key in cohort_bad)
    stop_records = tuple(c._validated(StopRecord, row) for row in (() if stops is None else stops))
    for stop in stop_records:
        if stop.payment_id is not None and stop.payment_id not in cohort:
            raise ValueError("canary stop does not join its executed receipt")
        trigger = cohort.get(stop.payment_id)
        after = tuple(
            p for p in cohort.values() if trigger is not None and trigger.at < p.at <= stop.at
        )
        if stop.payments_after_trigger != len(after) or (
            stop.cents_after_trigger != sum(p.amount_cents for p in after)
        ):
            raise ValueError("canary stop exposure differs from its receipts")
    paused = tuple(s for s in stop_records if s.trigger == "correction")
    exposed = ()
    if paused:
        first = min(cohort[s.payment_id].at for s in paused)
        ended = min(s.at for s in paused)
        exposed = tuple(p for p in cohort.values() if first <= p.at <= ended)
    hourly = Counter(
        (
            region_of(e.order_id),
            e.arrived_at.replace(
                minute=0,
                second=0,
                microsecond=0,
            ),
        )
        for e in unique.values()
        if window_start <= e.arrived_at < window_end
    )
    first_hour = window_start.replace(minute=0, second=0, microsecond=0)
    arrivals_by_hour = tuple(
        {"hour": at.isoformat(), "region": r, "scans": hourly[r, at]}
        for hour in range(ceil((window_end - first_hour).total_seconds() / 3600))
        for r in range(3)
        if (at := first_hour + timedelta(hours=hour)) < window_end
    )
    policy_referrals = {
        row.request.request_id
        for row in lineage
        if row.request.request_id in affected and row.decision.reason == "delivery_unknown"
    }
    held_after = {
        row.request.request_id
        for row in lineage
        if row.request.request_id in affected
        and row.decision_at >= batch_at
        and row.decision.reason != "shipment_incomplete"
    }
    lag = watermark_lag_range(event_log, window_start, window_end)
    eligible = tuple(e for e in unique.values() if window_start <= e.at < window_end)
    late = {e.event_id for e in late_events(event_log)}
    timeline_rows = [
        {
            "at": s.at.isoformat(),
            "name": "canary_stop:" + s.trigger,
            "span_id": s.stop_id,
            "trace_id": payment_evidence(cohort[s.payment_id]).trace_id if s.payment_id else "",
            "run_id": "",
            "links": (),
            "audit_ids": (),
            "bundle": next(
                (
                    span.bundle
                    for span in cohort_spans
                    if s.payment_id is not None
                    and span.trace_id == payment_evidence(cohort[s.payment_id]).trace_id
                ),
                None,
            ),
        }
        for s in stop_records
    ]
    timeline = tuple(
        sorted(
            timeline_rows
            + [
                {
                    "at": s.start.isoformat(),
                    "name": s.name,
                    "span_id": s.span_id,
                    "trace_id": s.trace_id,
                    "run_id": s.run_id,
                    "links": s.links,
                    "bundle": s.bundle,
                    "audit_ids": tuple(
                        a.audit_id
                        for a in audit
                        if a.trace_id == s.trace_id and s.start <= a.at <= s.end
                    ),
                }
                for s in spans
                if s.name in ("run", "watermark_advance", "correction")
            ],
            key=lambda row: (row["at"], row["span_id"]),
        )
    )
    deployments = registry_history(registry, window_start, window_end)
    return IncidentRecord(
        region=region,
        last_arrival=last,
        batch_at=batch_at,
        batch_scans=len(batch),
        observed_gap_hours=(batch_at - last).total_seconds() / 3600,
        corrections=len(corrected),
        corrected_modes=dict(Counter(r.mode for r in corrections)),
        affected_decisions=affected,
        held_after_batch=len(held_after),
        policy_referrals=len(policy_referrals),
        cash_loss_cents=cash,
        shadow_exposure_cents=shadow,
        illustrative_review_cost_cents=400 * len(policy_referrals),
        cohort_exposure=len(exposed) if stops is not None else None,
        cohort_exposure_cents=sum(p.amount_cents for p in exposed) if stops is not None else None,
        cohort_corrections=len(cohort_bad),
        cohort_stops=tuple(s.model_dump(mode="json") for s in stop_records),
        arrivals_by_region_hour=arrivals_by_hour,
        bundles=tuple(
            sorted(
                {r.bundle for r in lineage if r.bundle is not None}
                | {s.bundle for s in (*spans, *cohort_spans) if s.bundle is not None}
            )
        ),
        unstamped_lineage=sum(r.bundle is None for r in lineage),
        releases=sum(r["kind"] != "demotion" for r in deployments)
        if registry is not None
        else None,
        release_history=journal_history(journal, window_start, window_end),
        deployment_history=deployments,
        watermark_lag_hours=lag,
        late_share=sum(e.event_id in late for e in eligible) / len(eligible) if eligible else 0,
        shipment_ages_seconds=tuple(
            sorted({payment_evidence(p.receipt).freshness["shipment"] for p in wrong})
        ),
        timeline=timeline,
        budget_status=budget_status,
        unknowns=(
            "batch completeness",
            "cause of regional silence",
            *(("canary stop/executed exposure unavailable",) if stops is None else ()),
            *(("unstamped lineage records",) if any(r.bundle is None for r in lineage) else ()),
            *(("deployment history: registry not supplied",) if registry is None else ()),
        ),
        actions=(
            {
                "owner": "release_owner",
                "action": "regional watermark and silence regression",
            },
            {
                "owner": "the carrier integration team",
                "action": "supply independently reconciled regional counts",
            },
        ),
    )


def watermark_lag_range(log, start, end):
    """Continuous limits: lag rises between arrivals and can fall at an arrival."""
    progress = watermark(log, start)
    values = []
    if progress is not None:
        values.append((start - progress).total_seconds() / 3600)
    arrivals = (e for e in log.entries if start < e.arrived_at < end)
    for at, events in groupby(arrivals, key=lambda e: e.arrived_at):
        if progress is not None:
            values.append((at - progress).total_seconds() / 3600)
        latest = max(e.at for e in events) - MARGIN
        progress = max(progress, latest) if progress is not None else latest
        values.append((at - progress).total_seconds() / 3600)
    if progress is None:
        raise ValueError("watermark unobserved throughout incident window")
    values.append((end - progress).total_seconds() / 3600)
    return min(values), max(values)


def registry_history(registry, start, end):
    """Signed promotions/rollbacks and ended-stop demotions, in this namespace."""
    if registry is None:
        return ()
    return tuple(
        row.record()
        for row in registry.transitions()
        if row.status == "promoted"
        and not row.unmet
        and (row.signed_by or row.kind == "demotion")
        and row.signed_at is not None
        and start <= row.signed_at < end
    )


def journal_history(journal, start, end):
    """Read the two actual public record types returned by Journal.read()."""
    rows = []
    for record in () if journal is None else journal:
        if record.get("version") == "refund-release-record-2":
            at = c._utc_time(datetime.fromisoformat(record["at"]))
            details = {
                "event": "evaluation",
                "decision": record["decision"],
                "status": record["status"],
            }
        elif record.get("event") == "sign":
            at = c._utc_time(datetime.fromisoformat(record["signature"]["at"]))
            details = {"event": "sign", "owner": record["signature"]["owner"]}
        else:
            raise ValueError("unknown release journal record type")
        if start <= at < end:
            rows.append(
                {
                    "at": at.isoformat(),
                    "record_id": record["record_id"],
                    "candidate_fingerprint": record["candidate_fingerprint"],
                    "software_fingerprint": record["software_fingerprint"],
                    **details,
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["at"], row["event"])))
