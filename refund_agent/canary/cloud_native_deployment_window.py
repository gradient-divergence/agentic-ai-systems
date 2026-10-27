"""Freeze arrivals first, await every terminal decision, then sign D6 once."""

import heapq
from datetime import datetime, timedelta
from typing import cast

from pydantic import model_validator

from refund_agent import contracts as c
from refund_agent.baseline import baseline
from refund_agent.batch_stream.generate import generate as generate_carrier
from refund_agent.batch_stream.generate import read_sources, scan
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.view import ALLOWED, MARGIN, first_complete, shipment_as_of
from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.data.generate import _label, number
from refund_agent.data.live import generate_live
from refund_agent.release.candidate import build_tools, run_request
from refund_agent.runtimes.swap import _project
from refund_agent.selection import fingerprint


def measures(rows, at):
    # Candidate construction initializes the core's offer adapters first.
    from refund_agent.shadow.compare import shadow_to_canary

    return shadow_to_canary(rows, at=at)


def terminal_comparators(data, request, at):
    # Comparators read refreshed evidence at execution, retaining the arrival elsewhere.
    executing = request.model_copy(update={"at": at})
    person = _label(executing, data.orders[0], data.shipments[0], data.policy_at(at))
    return person, baseline(data, [executing])[0]


def live_event_log(stream) -> EventLog:
    """Constructed join: chapter 29's event times, chapter 31's generated delays.

    A snapshot's observation clock is not a scan-arrival clock. Arrival patterns
    (including copies) are drawn from the public generated carrier stream,
    independently of outcomes; no snapshot timestamp is repaired or relabeled.
    """
    patterns = {}
    for event in generate_carrier().log.entries:
        patterns.setdefault(event.event_id, []).append(event.arrived_at - event.at)
    patterns = tuple(patterns.values())
    events = []
    for shipment in stream.data.shipments:
        for n, event in enumerate(shipment.events):
            pattern = patterns[number(stream.manifest["seed"], len(events), "cloud-native-deployment-arrival", len(patterns))]
            events.extend(scan(shipment.order_id, event.status, event.at, event.at + delay, n)
                          for delay in pattern)
    log = EventLog(events)
    start = min(r.at for r in stream.data.requests) - timedelta(minutes=5)
    end = max(r.at for r in stream.data.requests) + ALLOWED + MARGIN + timedelta(hours=2)
    ticks = []
    at = start
    while at <= end:
        ticks.append(scan(f"cloud-native-deployment-feed-{len(ticks)}", "shipped", at, at))
        at += timedelta(minutes=5)
    return EventLog(log.entries + tuple(ticks))


def next_run_at(log, requested_at, at):
    """Advance a held run on the known arrival clock; never retry at its own time."""
    future = [event.arrived_at for event in log.entries if event.arrived_at > at]
    if not future:
        raise ValueError("event log ends before the held request can complete")
    return max(first_complete(log, requested_at + ALLOWED, MARGIN), min(future))


class ShadowWindowResult(c.Record):
    window_id: str
    candidate_fingerprint: str
    bundle_fingerprint: str | None = None
    start: datetime
    completed_at: datetime
    labels_mature_at: datetime
    generated_label_shortcut: bool
    required_requests: int
    request_ids: tuple[str, ...]
    completion_order: tuple[str, ...]
    comparisons: tuple[dict, ...]
    forbidden_payments: int
    referrals: int
    baseline_referrals: int
    wrongful_denials: int
    baseline_wrongful_denials: int
    measure_verdicts: dict[str, bool]
    measures_pass: bool
    signed_by: str | None = None
    signed_at: datetime | None = None

    @model_validator(mode="after")
    def counts_match(self):
        ids = tuple(row["request_id"] for row in self.comparisons)
        if ids != self.request_ids or len(set(ids)) != len(ids):
            raise ValueError("window must retain each frozen request exactly once")
        if set(self.completion_order) != set(ids) or len(self.completion_order) != len(ids):
            raise ValueError("window requires every terminal decision")
        if len(ids) != self.required_requests:
            raise ValueError("window incomplete; held requests cannot be replaced")
        times = [datetime.fromisoformat(r["terminal_at"]) for r in self.comparisons]
        if self.completed_at != max(times) or min(times) < self.start:
            raise ValueError("window terminal clocks differ")
        paid = [datetime.fromisoformat(r["payment_at"]) for r in self.comparisons if r["person_paid"]]
        maturity = (max(paid) if paid else self.completed_at) + timedelta(days=30)
        if self.labels_mature_at != maturity:
            raise ValueError("labels mature 30 days after the last person payment")
        result = measures(list(self.comparisons), self.labels_mature_at.isoformat())
        keys = ("forbidden_payments", "referrals", "baseline_referrals",
                "wrongful_denials", "baseline_wrongful_denials")
        values = self.model_dump()
        if any(values[k] != result[k] for k in keys):
            raise ValueError("window measures differ from its comparisons")
        verdicts = {
            "forbidden_payments": self.forbidden_payments == 0,
            "referral_share": self.referrals <= self.baseline_referrals,
            "wrongful_denials": self.wrongful_denials <= self.baseline_wrongful_denials,
        }
        passes = result["window_complete"] and result["measures_pass"] and self.required_requests == 299
        if self.measure_verdicts != verdicts or self.measures_pass != passes:
            raise ValueError("window verdicts differ from their measures")
        if (self.signed_by is None) != (self.signed_at is None):
            raise ValueError("window signature requires owner and time")
        if self.signed_by and (self.signed_at is None or
            not passes or not has_role(self.signed_by, "decision_owner")
            or self.signed_at < self.completed_at
            or not self.generated_label_shortcut and self.signed_at < maturity
        ):
            raise ValueError("window signature lacks passing, mature decision-owner evidence")
        return self

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def run_window(candidate, start: datetime, n=299, *, stream=None, event_log=None,
               bundle_fingerprint=None) -> ShadowWindowResult:
    start = c._utc_time(start)
    stream = stream if stream is not None else generate_live(7, start, 14, 120)
    if not stream.intact() or type(n) is not int or n < 1:
        raise ValueError("intact generated stream and positive window required")
    arrivals = sorted(enumerate(stream.data.requests), key=lambda row: (row[1].at, row[0]))
    eligible = [(i, r) for i, r in arrivals if r.at >= start and r.amount_cents < 5000]
    frozen = eligible[:n]
    if len(frozen) != n or len({r.request_id for _, r in frozen}) != n:
        raise ValueError("stream does not contain the next distinct eligible requests")
    log = event_log if event_log is not None else live_event_log(stream)
    tasks: list[tuple[datetime, int, bool]] = [(r.at, j, False) for j, (_, r) in enumerate(frozen)]
    heapq.heapify(tasks)
    mounted, terminal, completions = {}, {}, []
    while tasks:
        at, j, rerun = heapq.heappop(tasks)
        index, request = frozen[j]
        data = _project(stream.data, request).model_copy(update={"correct": []})
        if rerun:
            data = read_sources(data, at)
        data = data.model_copy(update={
            "shipments": [shipment_as_of(log, request.order_reference, at)],
        })
        if rerun:
            tools = mounted[j]
            tools.data = data
        else:
            tools = mounted[j] = build_tools(data, candidate, "shadow", event_log=log)
        trace = run_request(candidate, tools, request, at=at)
        if trace.decision.reason == "shipment_incomplete":
            final = next_run_at(log, request.at, at)
            pending = cast(tuple[datetime, int, bool], (final, j, True))
            heapq.heappush(tasks, pending)
            continue
        observed = data.model_copy(update={
            "shipments": [shipment_as_of(log, request.order_reference, at)],
        })
        # The generated person reads the same terminal evidence, blind to the agent.
        person, base = terminal_comparators(observed, request, at)
        paid = person.outcome == c.Outcome.APPROVE
        payment_at = at if paid else None
        would_pay = bool(tools.shadow_payments)
        terminal[j] = {
            "request_id": request.request_id, "request_at": request.at.isoformat(),
            "terminal_at": at.isoformat(), "tier": 1,
            "agent": trace.decision.outcome.value, "agent_reason": trace.decision.reason,
            "baseline": base.outcome.value, "person": person.outcome.value,
            "person_saw_shadow": False, "person_paid": paid,
            "payment_at": payment_at.isoformat() if payment_at is not None else None,
            "label_final_at": (at + timedelta(days=30)).isoformat() if paid else None,
            "found_false_30d": int(number(stream.manifest["seed"], index, "live-paid-outcome", 50) == 0)
            if paid else None,
            "would_pay": would_pay,
            "forbidden_payment": would_pay and person.outcome != c.Outcome.APPROVE,
        }
        completions.append(request.request_id)
    rows = tuple(terminal[j] for j in range(n))
    completed = max(datetime.fromisoformat(r["terminal_at"]) for r in rows)
    paid_at = [datetime.fromisoformat(r["payment_at"]) for r in rows if r["person_paid"]]
    mature = (max(paid_at) if paid_at else completed) + timedelta(days=30)
    counts = measures(list(rows), mature.isoformat())
    verdicts = {
        "forbidden_payments": counts["forbidden_payments"] == 0,
        "referral_share": counts["referrals"] <= counts["baseline_referrals"],
        "wrongful_denials": counts["wrongful_denials"] <= counts["baseline_wrongful_denials"],
    }
    return ShadowWindowResult(
        window_id="cloud-native-deployment-window-" + fingerprint([candidate.fingerprint, start.isoformat(), rows])[:16],
        candidate_fingerprint=candidate.fingerprint, bundle_fingerprint=bundle_fingerprint,
        start=start, completed_at=completed, labels_mature_at=mature,
        generated_label_shortcut=True, required_requests=n,
        request_ids=tuple(r.request_id for _, r in frozen), completion_order=tuple(completions),
        comparisons=rows, measure_verdicts=verdicts,
        measures_pass=counts["measures_pass"],
        **{k: counts[k] for k in ("forbidden_payments", "referrals", "baseline_referrals",
                                   "wrongful_denials", "baseline_wrongful_denials")},
    )


def sign_window(result: ShadowWindowResult, owner: str, at: datetime) -> ShadowWindowResult:
    return c._validated(ShadowWindowResult, result.model_copy(update={
        "signed_by": owner, "signed_at": at,
    }))
