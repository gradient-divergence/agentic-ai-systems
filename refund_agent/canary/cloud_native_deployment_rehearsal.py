"""Constructed rollback and crash fixtures, isolated from canary admission.

The planted payment hook deliberately bypasses the gate. Crash fixtures enter
the core commit after their gate approved, to interrupt an admitted in-flight
call. Neither hook is used by the operational binding or by start.
"""

import heapq
import json
from collections import Counter
from datetime import datetime, timedelta
from fractions import Fraction
from os import getpid
from sys import executable
from typing import cast
from uuid import uuid4

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.batch_stream.generate import read_sources, snapshot_log
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.replay import PaymentRecord, corrections
from refund_agent.batch_stream.view import shipment_as_of
from refund_agent.canary.cloud_native_deployment_desk import desk
from refund_agent.canary.cloud_native_deployment_exposure import further_requests
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, in_cohort
from refund_agent.canary.cloud_native_deployment_restart import reconcile, recorded_sources
from refund_agent.canary.cloud_native_deployment_stop import (
    CanarySession,
    Grant,
    SettledEvidence,
    StopRule,
)
from refund_agent.canary.cloud_native_deployment_store import (
    AppendStore,
    ReadContracts,
)
from refund_agent.canary.cloud_native_deployment_window import live_event_log, next_run_at
from refund_agent.data.generate import Dataset, _policies
from refund_agent.release.agent_evaluation_resources import rehearsal_restart
from refund_agent.release.candidate import Candidate, build_candidate, build_tools, run_request
from refund_agent.release.extensions import resource_guard
from refund_agent.rules import freshness_problem, gate
from refund_agent.runtimes.swap import _project


class RehearsalRecord(c.Record):
    rehearsal_id: str
    candidate_fingerprint: str
    rollback_target_fingerprint: str
    cadence: str
    first_payment_cents: int | None
    triggering_payment_cents: int
    payments_after_trigger: int
    cents_after_trigger: int
    expected_further: str
    exposure_stop_trigger: str | None
    exposure_stop_at: datetime | None
    detected_at: datetime | None
    paid_twice: int
    unreadable_records: int
    checks: dict[str, bool]
    components: dict[str, bool]
    policy_version: str
    verdict: str
    restart_fresh_process: bool
    corrections: int
    recheck_overdue_pauses: int
    final_outcome: str
    blockers: tuple[str, ...] = ()


def fixture(canary_id, tag, at, *, forbidden=False, amount=4999):
    rid = f"cloud-native-deployment-{tag}"
    i = 0
    while not in_cohort(canary_id, f"{rid}-person-{i}"):
        i += 1
    request = c.Request(request_id=rid, requester_id=f"{rid}-person-{i}",
                        order_reference=rid + "-order", at=at, amount_cents=amount,
                        reason="not_delivered" if forbidden else "damaged")
    order = c.Order(order_id=request.order_reference, reference=request.order_reference,
                    purchased_at=at - timedelta(days=5), observed_at=at - timedelta(seconds=60),
                    items=[c.Item(name="physical goods", quantity=1, unit_cents=100000)],
                    paid_cents=100000, payment_id=rid + "-original-payment")
    shipment = c.Shipment(order_id=order.order_id, observed_at=at - timedelta(seconds=120),
                          events=[c.ShipmentEvent(at=at - timedelta(days=2), status="shipped"),
                                  c.ShipmentEvent(at=at - timedelta(hours=1), status="delivered")])
    policy = _policies()[-1]
    data = Dataset(orders=[order], requests=[request], shipments=[shipment], policies=[policy],
                   retrievals=[c.PolicySnapshot(request_id=rid, version=policy.version,
                                                observed_at=at - timedelta(hours=1))], correct=[])
    return data, request


def paid_twice(raw_receipts):
    counts = Counter(row["request"]["request_id"] for row in raw_receipts if row.get("request"))
    return sum(count > 1 for count in counts.values())


def rehearsal_session(plan, store, start):
    """An explicit simulated lane. It never produces a CanaryStart."""
    session = CanarySession(plan, store)
    if session.grant is not None:
        raise ValueError("rehearsal namespace already used")
    session.save(Grant(canary_id=plan.canary_id, plan_fingerprint=plan.fingerprint,
                       window_fingerprint="rehearsal; no admission", started_at=start,
                       expires_at=start + timedelta(days=plan.time_box_days),
                       outcome="running", lane="rehearsal"))
    return session


def prepare(tools, request, at):
    state = c.State(request=request, trace_id=f"trace-{request.request_id}", execution_at=at)
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        tools._dispatch(c.ToolCall(name=name), state)
    state.decision = tools.assess(state)
    tools._dispatch(c.ToolCall(name="refund.propose", arguments=c.ToolArguments(
        amount_cents=request.amount_cents, reason=request.reason)), state)
    return state


def _raw_receipts(store):
    latest = {}
    for row in store.rows("ledger"):
        latest[row["key"]] = row["value"]
    return tuple(v for v in latest.values() if v is not None and v.get("request"))


class CrashStore(AppendStore):
    def __init__(self, path, namespace, point):
        super().__init__(path, namespace)
        self.point = point

    def append(self, kind, key, value):
        if (self.point == "intent" and kind == "ledger" or
                self.point == "receipt" and kind == "audits" and value["event"] == "confirmation"):
            self.point = None
            raise OSError("constructed crash boundary")
        return super().append(kind, key, value)


def _interrupt(candidate, plan, store, log, at, contracts):
    fixtures = []
    for point in ("intent", "receipt"):
        data, request = fixture(plan.canary_id, "interrupted-" + point, at)
        crash = CrashStore(store.path, store.namespace, point)
        session = CanarySession(plan, crash)
        tools = build_tools(data, candidate, "execute", event_log=snapshot_log(data),
                            canary_session=session, canary_contracts=contracts)
        state = prepare(tools, request, at)
        proposal = tools.proposals[state.proposal_id]
        if gate(data, state, request.amount_cents, proposal, None, 0) is not None:
            raise ValueError("in-flight fixture was not gate-admitted")
        try:
            tools._commit(state, f"execute-{request.request_id}", request.amount_cents, None)
        except (OSError, RuntimeError):
            pass  # Deliberate process-crash fixture; fresh process alone reconciles.
        fixtures.append((data, request))
    return fixtures


def restart_payload(payload):
    from refund_agent.canary.cloud_native_deployment_component import restore_selection

    restore_selection(payload["target"])
    target = Candidate(**payload["target"])
    return restart_target(target, payload)


def public_regression(candidate, suite):
    from refund_agent.release.suite import run_case

    decisions = []
    for case in suite.cases:
        if case.part != "public":
            continue
        try:
            decisions.append(run_case(case, candidate)["decision"])
        except ValueError as error:
            if "canary_session_required" not in str(error):
                raise
            return decisions, (case.case_id + ": authenticated core fixture authority is absent",)
    return decisions, ()


def recheck_evidence(payment, store, contracts):
    """Re-evaluate the gate on the evidence the paying run actually read."""
    if payment.request is None or payment.evidence is None:
        raise ValueError("recheck requires recorded payment evidence")
    state = c.State(request=payment.request, **payment.evidence.model_dump(), execution_at=payment.at)
    proposals = store.records("proposals", contracts)
    proposal = next((p for p in proposals.values() if p.request == payment.request), None)
    problem = gate(recorded_sources(payment, payment.at), state, payment.amount_cents, proposal, None, 0)
    return SettledEvidence(payment_id=payment.receipt_id, final=True, allowed_on_read=problem is None,
                           stale_data=freshness_problem(payment.evidence.freshness) is not None)


@resource_guard(0)
def restart_target(candidate, payload):
    from refund_agent.canary.cloud_native_deployment_component import read_contracts, runtime_paths
    from refund_agent.release.suite import build_suite

    target = candidate
    plan = CanaryPlan.model_validate_json(json.dumps(payload["plan"]))
    path = runtime_paths(target)["store"]
    if payload.get("store_path", str(path)) != str(path):
        raise ValueError("restart store differs from the selected owner declaration")
    store = AppendStore(path, payload["namespace"])
    contracts = read_contracts(target)
    raw_before = _raw_receipts(store)
    try:
        data = Dataset.model_validate_json(json.dumps(payload["fixtures"][0]))
        log = EventLog(ScanEvent.model_validate_json(json.dumps(raw)) for raw in payload["event_log"])
        # A closed rollback target refuses unfamiliar fields before any execution.
        tools = build_tools(data, target, "execute", event_log=log,
                            canary_session=CanarySession(plan, store))
        repaired = reconcile(tools)
        second = reconcile(tools)
        receipts_kept = raw_before == _raw_receipts(store)
        closed = len(repaired["unpaid"]) == 1
        confirmed = len(repaired["confirmed"]) == 1
        idempotent = not second["confirmed"] and not second["unpaid"]
        to_person = True
        for raw in payload["fixtures"]:
            data = Dataset.model_validate_json(json.dumps(raw))
            request = data.requests[0]
            data = data.model_copy(update={"shipments": [shipment_as_of(log, request.order_reference, request.at)]})
            stopped = build_tools(data, target, "execute", event_log=log,
                                  canary_session=CanarySession(plan, store))
            before = len(stopped.payments)
            already_paid = stopped.payment_for(request.request_id) is not None
            trace = run_request(target, stopped, request)
            if trace.decision.executed and len(stopped.payments) != before:
                raise ValueError("rollback reopened cohort authority")
            if not already_paid:
                to_person = to_person and trace.decision.outcome == c.Outcome.REFER and (
                    request.request_id in stopped.person_requests
                )
            person, _ = desk(data, store, [request], event_log=log, candidate=target, contracts=contracts)
            desk(data, store, [request], event_log=log, candidate=target, contracts=contracts)
            if request.request_id in {r.request_id for r in repaired["unpaid"]} and person.payment_for(request.request_id) is None:
                raise ValueError("desk did not reconcile the in-flight request")
        suite = build_suite(components=target.tools_composition["components"])
        current = build_candidate(suite, target.name, target.runtime["identity"],
                                  target.tools_composition["verifier"])
        loaded = tools._release_candidate.record()
        components = {key: loaded[key] == current.record()[key] == target.record()[key]
                      for key in ("agent", "model", "runtime", "adapter", "tools_composition", "offer_policy")}
        public, blockers = public_regression(target, suite)
        checks = {
            "next_cohort_to_person": to_person, "receipts_kept": receipts_kept,
            "intents_closed_unpaid": closed, "receipts_confirmed_at_restart": confirmed,
            "reconcile_idempotent": idempotent, "readable_records": True,
            "post_restart": not blockers and not payload["public_blockers"]
            and len(public) == 40 and public == payload["public"],
        }
        unreadable = 0
    except ValidationError as error:
        queued = []
        for raw in payload["fixtures"]:
            rid = raw["requests"][0]["request_id"]
            store.append("reconciliation", rid, {"request_id": rid, "reason": "unreadable_records", "for_person": True})
            queued.append(store.latest("reconciliation", rid)["for_person"])
        checks = {"next_cohort_to_person": bool(queued) and all(queued), "receipts_kept": raw_before == _raw_receipts(store),
                  "intents_closed_unpaid": False, "receipts_confirmed_at_restart": False,
                  "reconcile_idempotent": False, "readable_records": False, "post_restart": False}
        unreadable = 1
        return {"checks": checks, "components": {}, "blockers": [], "process_id": getpid(), "unreadable_records": unreadable,
                "paid_twice": paid_twice(_raw_receipts(store)), "detail": str(error)}
    return {"checks": checks, "components": components, "blockers": list(blockers), "process_id": getpid(), "unreadable_records": unreadable,
            "paid_twice": paid_twice(_raw_receipts(store))}


@resource_guard(6)
def run_rehearsal(plan, stream, days, faults, cadence, rollback_target, *, candidate,
                  event_log=None, context=None):
    from refund_agent.canary.cloud_native_deployment_component import (
        STORE_PATH,
        read_contracts,
        runtime_paths,
    )
    from refund_agent.release.suite import build_suite

    if cadence not in ("4h", "daily") or days < 3 or not stream.intact():
        raise ValueError("rehearsal requires intact inputs, at least three days and a known cadence")
    planted_scenario = list(faults) == ["forbidden_payment"]
    if any(isinstance(fault, str) and fault != "forbidden_payment" for fault in faults) or (
        faults and not planted_scenario and event_log is None
    ):
        raise ValueError("external incident records require their transformed event_log")
    plan = c._validated(CanaryPlan, plan)
    path = runtime_paths(candidate)["store"]
    if path != runtime_paths(rollback_target)["store"]:
        raise ValueError("rollback target must declare the same canary store")
    store = AppendStore(path, "cloud-native-deployment-rehearsal-" + uuid4().hex)
    start = c._utc_time(datetime.fromisoformat(stream.manifest["start"]))
    session = rehearsal_session(plan, store, start)
    log = event_log if event_log is not None else live_event_log(stream)
    contracts = {"read_contracts": read_contracts(candidate)}
    tasks: list[tuple[datetime, int, int, str]] = [(r.at, 3, i, "request") for i, r in enumerate(stream.data.requests)
             if start <= r.at < start + timedelta(days=days) and r.amount_cents < 5000
             and in_cohort(plan.canary_id, r.requester_id)]
    hours = 4 if cadence == "4h" else 24
    for hour in range(hours, days * 24 + hours, hours):
        tasks.append((start + timedelta(hours=hour), 0, hour, "review"))
    planted_at = start + timedelta(days=2, seconds=1)
    if planted_scenario:
        tasks.append((planted_at, 1, 0, "plant"))
    heapq.heapify(tasks)
    first, planted, exposure_stop, detected, correction_count = None, None, None, None, 0
    while tasks:
        at, _, index, kind = heapq.heappop(tasks)
        receipts = store.records("ledger", contracts.get("read_contracts", ReadContracts())).values()
        receipts = tuple(p for p in receipts if p.request)
        rule = StopRule(session, receipts)
        if kind == "review":
            for payment in receipts:
                if payment.receipt_id in session.grant.payment_ids:
                    rule.rechecked(payment.receipt_id, at=at, owner="morgan")
            if planted is not None:
                if session.outcome == "running":
                    rule.trigger("recheck", at=at, payment_id=planted.receipt_id)
                detected = at
                rule.end("defect", at=at, detail="constructed forbidden payment")
            elif session.outcome == "paused" and all(s.trigger == "recheck_overdue" for s in session.pending_stops):
                pending_ids = {s.payment_id for s in session.pending_stops}
                settled = tuple(recheck_evidence(p, store, contracts["read_contracts"])
                                for p in receipts if p.receipt_id in pending_ids)
                rule.resume("morgan", settled, at=at)
        else:
            session.tick(at, receipts)
        if planted is not None and exposure_stop is None:
            exposure_stop = next((s for s in session.stops if s.at >= planted_at and s.outcome == "paused"), None)
        if session.outcome == "ended":
            break
        if kind == "plant":
            data, request = fixture(plan.canary_id, "planted-forbidden", at, forbidden=True)
            tools = build_tools(data, candidate, "execute", event_log=snapshot_log(data), canary_session=session,
                                canary_contracts=contracts, **(context or {}).get("tools_context", {}))
            state = prepare(tools, request, at)
            tools._commit(state, f"execute-{request.request_id}", request.amount_cents, None)
            planted = tools.payment_for(request.request_id)
            deadline = cast(tuple[datetime, int, int, str], (at + timedelta(hours=4), 2, 0, "deadline"))
            heapq.heappush(tasks, deadline)
        elif kind in ("request", "rerun"):
            request = stream.data.requests[index]
            data = _project(stream.data, request).model_copy(update={"correct": []})
            if kind == "rerun":
                data = read_sources(data, at)
            data = data.model_copy(update={
                "shipments": [shipment_as_of(log, request.order_reference, at)],
            })
            tools = build_tools(data, candidate, "execute", event_log=log, canary_session=session,
                                canary_contracts=contracts, **(context or {}).get("tools_context", {}))
            previous = set(p.receipt_id for p in tools.payments)
            trace = run_request(candidate, tools, request, at=at)
            if trace.decision.reason == "shipment_incomplete":
                rerun = cast(tuple[datetime, int, int, str], (next_run_at(log, request.at, at), 3, index, "rerun"))
                heapq.heappush(tasks, rerun)
            for payment in tools.payments:
                if payment.receipt_id not in previous:
                    if first is None:
                        first = payment.amount_cents
                    due = cast(tuple[datetime, int, int, str], (payment.at + timedelta(hours=4), 2, index, "deadline"))
                    heapq.heappush(tasks, due)
                    for correction in corrections(log, [PaymentRecord(payment, "executed")]):
                        arrives = cast(tuple[datetime, int, int, str], (correction.arrived_at, 1, index, "correction"))
                        heapq.heappush(tasks, arrives)
        elif kind == "correction":
            for correction in corrections(log, [PaymentRecord(p, "executed") for p in receipts], through=at):
                if rule.correction(correction, at=at):
                    correction_count += 1
    if not planted_scenario:
        # Chapter 34 owns the incident and its validation; this replay records
        # actual effects without claiming an unperformed rollback acceptance.
        stop = next((s for s in session.stops if s.trigger == "correction"), None)
        trigger = next((p for p in receipts if stop and p.receipt_id == stop.payment_id), None)
        actual = [p for p in receipts if trigger and stop and trigger.at < p.at <= stop.at]
        checks = {"stream_unchanged": stream.intact(), "no_request_paid_twice": paid_twice(_raw_receipts(store)) == 0,
                  "original_expiry": session.grant.expires_at == start + timedelta(days=plan.time_box_days)}
        row = RehearsalRecord(
            rehearsal_id=plan.rehearsal_ref, candidate_fingerprint=candidate.fingerprint,
            rollback_target_fingerprint=rollback_target.fingerprint, cadence=cadence,
            first_payment_cents=first, triggering_payment_cents=trigger.amount_cents if trigger else 0,
            payments_after_trigger=len(actual), cents_after_trigger=sum(p.amount_cents for p in actual),
            expected_further=str(further_requests(Fraction(1, 20), Fraction(2000, 720),
                Fraction(str((stop.at - trigger.at).total_seconds())) / 3600)) if stop and trigger else "0",
            exposure_stop_trigger=stop.trigger if stop else None, exposure_stop_at=stop.at if stop else None,
            detected_at=stop.at if stop else None, paid_twice=paid_twice(_raw_receipts(store)),
            unreadable_records=0, checks=checks, components={},
            policy_version="in-force " + stream.data.policies[0].version,
            verdict="observed", restart_fresh_process=False, corrections=correction_count,
            recheck_overdue_pauses=sum(s.trigger == "recheck_overdue" and s.outcome == "paused" for s in session.stops),
            final_outcome=session.outcome,
        )
        store.append("rehearsal", row.rehearsal_id, row.model_dump(mode="json"))
        return row
    if planted is None or exposure_stop is None or detected is None:
        raise ValueError("rehearsal never reached the planted payment and its stop")
    actual = [p for p in receipts if planted.at < p.at <= exposure_stop.at]
    fixtures = _interrupt(candidate, plan, store, log, exposure_stop.at - timedelta(microseconds=1), contracts)
    worker_log = EventLog(log.entries + tuple(e for data, _ in fixtures for e in snapshot_log(data).entries))
    remainder = []
    for request in stream.data.requests:
        if request.at.date() != exposure_stop.at.date() or request.at <= exposure_stop.at or (
            request.amount_cents >= 5000 or not in_cohort(plan.canary_id, request.requester_id)
        ):
            continue
        data = _project(stream.data, request).model_copy(update={"correct": [], "shipments": [
            shipment_as_of(worker_log, request.order_reference, request.at),
        ]})
        remainder.append((data, request))
    suite = build_suite(components=rollback_target.tools_composition["components"])
    public, public_blockers = public_regression(rollback_target, suite)
    payload = {"target": rollback_target.record(), "plan": plan.model_dump(mode="json"),
               "namespace": store.namespace, "public_blockers": public_blockers,
               "fixtures": [data.model_dump(mode="json") for data, _ in fixtures + remainder], "public": public,
               "event_log": [event.model_dump(mode="json") for event in worker_log.entries]}
    store_args = []
    if path != STORE_PATH:
        payload["store_path"] = str(path)
        store_args = ["--store", str(path)]
    # The declared append store is the transport; scoped components cannot open
    # arbitrary inherited pipe descriptors. The worker prints and pays nothing.
    store.append("restart-payload", "rollback", payload)
    process = rehearsal_restart()([executable, "-m", "refund_agent.canary", "restart-store",
                                   "--namespace", store.namespace, *store_args])
    if process.returncode:
        raise ValueError("fresh-process restart failed")
    restarted = store.latest("restart-result", "rollback")
    if restarted is None:
        raise ValueError("fresh-process restart produced no result")
    components = restarted["components"]
    components["verifier_changed"] = candidate.tools_composition["verifier"] != rollback_target.tools_composition["verifier"]
    # The planted-field case intentionally rolls back to the original verifier.
    components_ok = len(components) == 7 and all(v for k, v in components.items() if k != "verifier_changed")
    checks = {**restarted["checks"], "components_restored": components_ok,
              "no_request_paid_twice": restarted["paid_twice"] == 0}
    record = RehearsalRecord(
        rehearsal_id=plan.rehearsal_ref, candidate_fingerprint=candidate.fingerprint,
        rollback_target_fingerprint=rollback_target.fingerprint, cadence=cadence,
        first_payment_cents=first, triggering_payment_cents=planted.amount_cents,
        payments_after_trigger=len(actual), cents_after_trigger=sum(p.amount_cents for p in actual),
        expected_further=str(further_requests(Fraction(1, 20), Fraction(2000, 720), Fraction(4))),
        exposure_stop_trigger=exposure_stop.trigger, exposure_stop_at=exposure_stop.at,
        detected_at=detected, paid_twice=restarted["paid_twice"],
        unreadable_records=restarted["unreadable_records"], checks=checks, components=components,
        policy_version="not in the candidate record; in-force policy remains " + stream.data.policies[0].version,
        verdict="pass" if all(checks.values()) else "fail", restart_fresh_process=restarted["process_id"] != getpid(),
        corrections=correction_count,
        recheck_overdue_pauses=sum(s.trigger == "recheck_overdue" and s.outcome == "paused" for s in session.stops),
        final_outcome=session.outcome,
        blockers=tuple(dict.fromkeys((*public_blockers, *restarted["blockers"]))),
    )
    store.append("rehearsal", record.rehearsal_id, record.model_dump(mode="json"))
    return record
