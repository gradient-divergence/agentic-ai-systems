"""Append-only budget accounting, bounded windows and restart authority.

The log binds every observation to the signed plan/window. Spend is rebuilt on
load. Registering a callback on an already-spent log invokes it immediately.
Boundaries close admission, not outcomes. Late outcomes can page each window once.
"""

import fcntl
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.batch_stream.replay import CorrectionRecord, PaymentRecord
from refund_agent.observability_reliability.budget import (
    ErrorBudgetRecord,
    binom_tail,
    burn_rate,
    fraction,
    min_count,
)
from refund_agent.observability_reliability.objectives import (
    DenialPair,
    Observation,
    classify,
    paired_denials,
    payment_request,
)
from refund_agent.observability_reliability.reasons import reason_class
from refund_agent.rules import tier_for


class AlertRecord(c.Record):
    rule: str
    window_start: datetime
    window_end: datetime
    count: int
    threshold: int
    owner_roles: tuple[str, ...]
    kind: str = "page"


class BudgetSpent(c.Record):
    canary_id: str
    candidate_fingerprint: str
    budget_fingerprint: str
    at: datetime
    objectives: tuple[str, ...]
    reason: str = "budget_spent"


class Monitor:
    def __init__(
        self,
        record: ErrorBudgetRecord,
        path: Path | None = None,
        *,
        store=None,
    ):
        if path is not None and store is not None:
            raise ValueError("select one budget append store")
        self.record = c._validated(ErrorBudgetRecord, record)
        self.binding = record.fingerprint
        self.path, self.store = path, store
        self._host_binding = None
        self._spent: BudgetSpent | None = None
        self.observations: dict[str, Observation] = {}
        self.denial_pairs: dict[str, DenialPair] = {}
        self.payments: dict[tuple[str, str], PaymentRecord] = {}
        self.corrections: dict[tuple[str, str], CorrectionRecord] = {}
        self.alerts: list[AlertRecord] = []
        self._closed: set[tuple[str, int]] = set()
        self._paged: set[tuple[str, int]] = set()
        self._callbacks: dict[str, Callable[[BudgetSpent], None]] = {}
        self._delivered: set[str] = set()
        self._breach_callbacks: list[Callable[[dict], None]] = []
        self._breaches_delivered: set[tuple[str, str]] = set()
        self._payment_breaches: dict[tuple[str, str], tuple[str, ...]] = {}
        self._rows = 0
        self._bytes = 0
        self.now = record.start
        self.outcomes_final = False
        rows = (
            [r["value"] for r in store.rows("observability-reliability-budget")]
            if store is not None
            else [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path is not None and path.exists()
            else []
        )
        for row in rows:
            if row["entry"] != self._rows + 1 or row["binding"] != self.binding:
                raise ValueError("budget log sequence or signed plan/window differs")
            self._rows += 1
            self._apply(row["kind"], row["data"])
        if path is not None and path.exists():
            self._bytes = path.stat().st_size

    @classmethod
    def for_canary(cls, session, candidate):
        """Bind persisted combined spend and callbacks before payment or resume."""
        from refund_agent.canary.cloud_native_deployment_component import (
            read_contracts,
        )
        from refund_agent.canary.cloud_native_deployment_stop import StopRule
        from refund_agent.selection import fingerprint

        plan, grant = session.plan, session.grant
        admission = session.store.latest("budget", plan.error_budget_ref)
        if admission is None or "definition" not in admission:
            raise ValueError("exact budget admission required before execution")
        record = ErrorBudgetRecord.model_validate_json(json.dumps(admission["definition"]))
        if admission != record.canary_start_record():
            raise ValueError("budget admission projection or definition differs")
        if grant is None or (
            record.canary_id != plan.canary_id
            or record.candidate_fingerprint != plan.candidate_fingerprint
            or candidate.fingerprint != plan.candidate_fingerprint
            or plan.suite_fingerprint != candidate.suite["fingerprint"]
            or record.start != grant.started_at
            or record.end != grant.expires_at
            or plan.signed_at is None
            or plan.signed_at > record.start
            or grant.lane == "canary"
            and grant.budget_fingerprint != fingerprint(admission)
        ):
            raise ValueError("budget admission plan, candidate or window differs")
        session.verify_window_authority()
        monitor = cls(record, store=session.store)
        binding = {
            "plan_fingerprint": plan.fingerprint,
            "admission_fingerprint": fingerprint(admission),
            "window_fingerprint": grant.window_fingerprint,
            "error_budget_ref": plan.error_budget_ref,
            "namespace": session.store.namespace,
            "store": str(session.store.path),
        }
        if monitor._host_binding is None:
            monitor._append("canary_binding", binding)
        elif monitor._host_binding != binding:
            raise ValueError("budget host admission or window binding differs")

        def receipts():
            return tuple(session.store.records("ledger", read_contracts(candidate)).values())

        def demote(stop):
            if stop.outcome != "ended" or not any(
                row["name"] == "mlops-mle.bundle"
                for row in candidate.tools_composition.get("components", ())
            ):
                return
            from refund_agent.promotion.mlops_mle_bundle import from_candidate
            from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH
            from refund_agent.promotion.mlops_mle_registry import Registry

            bundle = from_candidate(candidate)
            registry = Registry(REGISTRY_PATH, session.store.namespace)
            # An unadmitted rehearsal exercises stops but has no release to demote.
            if grant.lane == "rehearsal" and not any(
                r.to_stage == "canary"
                and r.signed_by
                and r.candidate_fingerprint == bundle.fingerprint
                and r.refs.get("canary_id") == plan.canary_id
                for r in registry.transitions()
            ):
                return
            registry.demote(stop, bundle, canary_id=plan.canary_id)

        for stop in session.stops:
            demote(stop)

        def end(event):
            if (
                event.canary_id,
                event.candidate_fingerprint,
                event.budget_fingerprint,
            ) != (
                plan.canary_id,
                candidate.fingerprint,
                record.fingerprint,
            ):
                raise ValueError("budget exhaustion authority differs")
            stop = StopRule(session, receipts()).end("budget_spent", at=event.at)
            demote(stop)
            if not session.grant.budget_spent:
                session.save(session.grant.model_copy(update={"budget_spent": True}))

        def breach(event):
            if session.outcome == "ended":
                return
            rule = StopRule(session, receipts())
            detail = event["standing_instruction"] + ": " + ",".join(event["rules"])
            if event["mode"] == "executed":
                if event["payment_id"] not in session.grant.payment_ids:
                    raise ValueError("breach does not name a cohort receipt")
                stop = rule.trigger(
                    "owner",
                    at=monitor.now,
                    owner=record.signed_by,
                    payment_id=event["payment_id"],
                    detail=detail,
                )
            else:
                stop = rule.end("defect", at=monitor.now, detail=detail)
            demote(stop)

        monitor.register_budget_spent("canary.end", end)
        monitor.register_breach(breach)
        for receipt in sorted(receipts(), key=lambda p: (p.at, p.receipt_id)):
            if receipt.request is not None:
                monitor.tick(
                    max(monitor.now, receipt.at),
                    outcomes_final=monitor.outcomes_final,
                )
                monitor.record_payment(PaymentRecord(receipt, "executed"))
        return monitor

    def _append(self, kind: str, data: dict) -> None:
        row = {
            "entry": self._rows + 1,
            "binding": self.binding,
            "kind": kind,
            "data": data,
        }
        if self.store is not None:
            with self.store.transaction():
                if len(self.store.rows("observability-reliability-budget")) != self._rows:
                    raise ValueError("budget log changed; reload before appending")
                self.store.append("observability-reliability-budget", str(row["entry"]), row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                if self.path.stat().st_size != self._bytes:
                    raise ValueError("budget log changed; reload before appending")
                text = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                stream.write(text)
                stream.flush()
                with self.path.open(encoding="utf-8") as reader:
                    reader.seek(self._bytes)
                    if reader.read() != text:
                        raise ValueError("budget append failed readback")
                self._bytes = self.path.stat().st_size
        self._rows += 1
        self._apply(kind, data)

    def _apply(self, kind: str, data: dict) -> None:
        if kind == "canary_binding":
            if self._host_binding is not None:
                raise ValueError("budget host binding cannot be replaced")
            self._host_binding = data
        elif kind == "spent":
            event = BudgetSpent.model_validate_json(json.dumps(data))
            if self._spent is not None or (
                event.canary_id != self.record.canary_id
                or event.candidate_fingerprint != self.record.candidate_fingerprint
                or event.budget_fingerprint != self.binding
            ):
                raise ValueError("irreversible budget exhaustion binding differs")
            self._spent = event
        elif kind == "observation":
            row = Observation.model_validate_json(json.dumps(data))
            if row.reason:
                reason_class(row.reason)
            old = self.observations.get(row.request_id)
            if not self.record.start <= row.received_at < self.record.end:
                raise ValueError("log request outside signed window")
            if (
                old
                and old != row
                and (
                    old.received_at != row.received_at
                    or old.promise_hours != row.promise_hours
                    or old.tier != row.tier
                    or old.decided_at is not None
                    and old.reason != "shipment_incomplete"
                )
            ):
                raise ValueError("log changes a terminal observation")
            self.observations[row.request_id] = row
        elif kind == "denial_pair":
            row = DenialPair.model_validate_json(json.dumps(data))
            self._check_denial_pair(row)
            if row.request_id in self.denial_pairs:
                raise ValueError("D6 pair needs one observed request")
            self.denial_pairs[row.request_id] = row
        elif kind == "payment":
            row = PaymentRecord(
                c.Receipt.model_validate_json(json.dumps(data["receipt"])),
                data["mode"],
            )
            if not self.record.start <= row.receipt.at < self.record.end:
                raise ValueError("log payment outside signed window")
            if (row.mode, row.receipt.receipt_id) in self.payments:
                raise ValueError("duplicate payment log entry")
            self.payments[row.mode, row.receipt.receipt_id] = row
            from refund_agent.observability_reliability.objectives import (
                receipt_breaches,
            )

            self._payment_breaches[row.mode, row.receipt.receipt_id] = receipt_breaches(
                row, tuple(self.payments.values())
            )
        elif kind == "correction":
            data = dict(data)
            data["at"], data["arrived_at"] = (
                datetime.fromisoformat(data["at"]),
                datetime.fromisoformat(data["arrived_at"]),
            )
            row = CorrectionRecord(**data)
            payment = self.payments.get((row.mode, row.payment_id))
            if (
                payment is None
                or row.request_id != payment_request(payment.receipt).request_id
                or row.order_id != payment.receipt.order_id
            ):
                raise ValueError("log correction does not match payment")
            if row.arrived_at < payment.receipt.at or row.at > payment_request(payment.receipt).at:
                raise ValueError("log correction chronology differs")
            if (row.mode, row.payment_id) in self.corrections:
                raise ValueError("duplicate correction log entry")
            self.corrections[row.mode, row.payment_id] = row
        elif kind == "tick":
            if type(data["outcomes_final"]) is not bool:
                raise ValueError("outcomes_final requires a boolean")
            at = c._utc_time(datetime.fromisoformat(data["at"]))
            if at < self.now or self.outcomes_final and not data["outcomes_final"]:
                raise ValueError("log clock or finality moved backwards")
            self.now = at
            self.outcomes_final = data["outcomes_final"]
        elif kind == "closed":
            key = data["objective"], data["index"]
            self._closed.add(key)
            if data["alert"] is not None:
                self._paged.add(key)
                self.alerts.append(AlertRecord.model_validate_json(json.dumps(data["alert"])))
        else:
            raise ValueError("unknown budget log entry")

    def observe(self, observation: Observation) -> None:
        observation = c._validated(Observation, observation)
        if observation.reason:
            reason_class(observation.reason)
        old = self.observations.get(observation.request_id)
        if old == observation:
            return
        if old and (
            old.received_at != observation.received_at
            or old.promise_hours != observation.promise_hours
            or old.tier != observation.tier
            or old.decided_at is not None
            and old.reason != "shipment_incomplete"
        ):
            raise ValueError("request observation changed after its terminal answer")
        if not self.record.start <= observation.received_at < self.record.end:
            raise ValueError("request outside signed budget window")
        self._append("observation", observation.model_dump(mode="json"))
        self._notify()

    def record_payment(self, payment: PaymentRecord) -> None:
        key = payment.mode, payment.receipt.receipt_id
        old = self.payments.get(key)
        if old:
            if old != payment:
                raise ValueError("payment identity conflict")
            return
        if not self.record.start <= payment.receipt.at < self.record.end:
            raise ValueError("payment outside signed budget window")
        request_id = payment_request(payment.receipt).request_id
        if any(
            p.mode == payment.mode and payment_request(p.receipt).request_id == request_id
            for p in self.payments.values()
        ):
            raise ValueError("more than one payment per request and mode")
        self._append(
            "payment",
            {
                "mode": payment.mode,
                "receipt": payment.receipt.model_dump(mode="json"),
            },
        )
        self._notify()

    def record_denial_pair(self, pair: DenialPair) -> None:
        pair = c._validated(DenialPair, pair)
        self._check_denial_pair(pair)
        old = self.denial_pairs.get(pair.request_id)
        if old == pair:
            return
        if old is not None:
            raise ValueError("D6 pair identity conflict")
        self._append("denial_pair", pair.model_dump(mode="json"))

    def _check_denial_pair(self, pair):
        observation = self.observations.get(pair.request_id)
        if (
            observation is None
            or observation.tier != 1
            or observation.decided_at is None
            or observation.reason == "shipment_incomplete"
            or pair.agent_denied != (observation.outcome == "deny")
            or pair.person_paid_at is not None
            and pair.person_paid_at < observation.received_at
        ):
            raise ValueError("D6 pair differs from its observed terminal tier 1 request")

    def denial_report(self) -> dict:
        population = {o.request_id for o in self.observations.values() if o.tier == 1}
        missing = len(population - self.denial_pairs.keys())
        report = paired_denials(tuple(self.denial_pairs.values()), now=self.now)
        available = bool(population) and not missing and not report["pending"]
        return {
            **report,
            "available": available,
            "missing_pairs": missing,
            "pass": report["pass"] if available else None,
            "reason": None
            if available
            else (
                "paired baseline decisions and mature person-payment labels unavailable"
                if not self.denial_pairs
                else "paired population incomplete"
                if missing
                else "person-payment labels have not matured"
            ),
        }

    def record_correction(self, correction: CorrectionRecord) -> None:
        key = correction.mode, correction.payment_id
        if key not in self.payments:
            raise ValueError("correction needs its recorded payment and mode")
        old = self.corrections.get(key)
        if old:
            if old != correction:
                raise ValueError("correction identity conflict")
            return
        payment = self.payments[key].receipt
        if (
            correction.request_id != payment_request(payment).request_id
            or correction.order_id != payment.order_id
        ):
            raise ValueError("correction does not match payment")
        if correction.arrived_at < payment.at or correction.at > payment_request(payment).at:
            raise ValueError("correction chronology differs")
        data = {
            "payment_id": correction.payment_id,
            "mode": correction.mode,
            "request_id": correction.request_id,
            "order_id": correction.order_id,
            "event_id": correction.event_id,
            "at": correction.at.isoformat(),
            "arrived_at": correction.arrived_at.isoformat(),
            "status_at_payment": correction.status_at_payment,
            "status_as_settled": correction.status_as_settled,
        }
        self._append("correction", data)
        self.tick(
            max(self.now, correction.arrived_at),
            outcomes_final=self.outcomes_final,
        )

    def register_budget_spent(self, name: str, callback: Callable[[BudgetSpent], None]) -> None:
        if not name or name in self._callbacks or not callable(callback):
            raise ValueError("budget callback needs a unique name and callable")
        self._callbacks[name] = callback
        self._notify()

    def register_breach(self, callback: Callable[[dict], None]) -> None:
        if not callable(callback):
            raise ValueError("breach callback required")
        self._breach_callbacks.append(callback)
        self._notify()

    def tick(self, at: datetime, *, outcomes_final=False) -> None:
        if type(outcomes_final) is not bool:
            raise ValueError("outcomes_final requires a boolean")
        at = c._utc_time(at)
        if at < self.now or self.outcomes_final and not outcomes_final:
            raise ValueError("budget clock or finality moved backwards")
        if at != self.now or outcomes_final != self.outcomes_final:
            self._append(
                "tick",
                {"at": at.isoformat(), "outcomes_final": outcomes_final},
            )
        # Deadline passes count without requiring another span or answer.
        self._notify()

    def _counts(self, name: str, start: datetime, end: datetime) -> tuple[int, int, int]:
        if name == "corrections":
            payments = tuple(
                p
                for p in self.payments.values()
                if start <= p.receipt.at < end
                and p.receipt.at <= self.now
                and tier_for(payment_request(p.receipt).amount_cents) == c.Tier.AGENT
            )
            bad = sum((p.mode, p.receipt.receipt_id) in self.corrections for p in payments)
            return (
                len(payments),
                bad,
                0 if self.outcomes_final else len(payments) - bad,
            )
        rows = tuple(
            o
            for o in self.observations.values()
            if o.tier == 1 and start <= o.received_at < end and o.received_at <= self.now
        )
        states = tuple(classify(o, now=self.now).timeliness for o in rows)
        return len(rows), states.count("bad"), states.count("pending")

    def status(self) -> dict:
        objectives = {}
        for objective in self.record.objectives:
            n, bad, pending = self._counts(objective.name, self.record.start, self.record.end)
            rho = 1 - fraction(objective.target)
            allowance = fraction(objective.allowance)
            windows = {}
            for hours in (6, 72):
                recent_n, recent_bad, _ = self._counts(
                    objective.name,
                    max(self.record.start, self.now - timedelta(hours=hours)),
                    self.now,
                )
                windows[str(hours)] = (
                    str(burn_rate(recent_bad, recent_n, rho)) if recent_n else None
                )
            objectives[objective.name] = {
                "eligible": n,
                "bad": bad,
                "pending": pending,
                "signed_allowance": str(allowance),
                "remaining": str(max(0, allowance - bad)),
                "spent": bad > allowance
                or self._spent is not None
                and objective.name in self._spent.objectives,
                "bad_share": str(fraction(bad) / n) if n else None,
                "realized_pass": (
                    bad <= rho * n if n and not pending and self.now >= self.record.end else None
                ),
                "recent_burn": windows,
            }
        return {
            "canary_id": self.record.canary_id,
            "candidate_fingerprint": self.record.candidate_fingerprint,
            "budget_fingerprint": self.binding,
            "authority": {
                key: self._host_binding[key]
                for key in (
                    "plan_fingerprint",
                    "admission_fingerprint",
                    "window_fingerprint",
                    "error_budget_ref",
                )
            }
            if self._host_binding is not None
            else None,
            "window_start": self.record.start.isoformat(),
            "window_end": self.record.end.isoformat(),
            "as_of": self.now.isoformat(),
            "objectives": objectives,
            "d6": self.denial_report(),
            "spent": any(o["spent"] for o in objectives.values()),
        }

    def _notify(self) -> None:
        self._evaluate_windows(self.now, closed_only=True)
        spent = tuple(name for name, row in self.status()["objectives"].items() if row["spent"])
        if spent:
            if self._spent is None:
                self._append(
                    "spent",
                    BudgetSpent(
                        canary_id=self.record.canary_id,
                        candidate_fingerprint=self.record.candidate_fingerprint,
                        budget_fingerprint=self.binding,
                        at=self.now,
                        objectives=spent,
                    ).model_dump(mode="json"),
                )
            event = self._spent
            for name, callback in self._callbacks.items():
                if name not in self._delivered:
                    callback(event)
                    self._delivered.add(name)
        # A breach is never suppressed, in either mode. The consumer owns end/stop.
        for key, payment in self.payments.items():
            breaches = self._payment_breaches[key]
            if breaches and key not in self._breaches_delivered and self._breach_callbacks:
                for callback in self._breach_callbacks:
                    callback(
                        {
                            "mode": payment.mode,
                            "payment_id": key[1],
                            "rules": breaches,
                            "standing_instruction": self.record.breach_instruction,
                        }
                    )
                self._breaches_delivered.add(key)

    def ingest_span(self, span, *, observation=None, payment=None, correction=None):
        """Read a written span plus its referenced record, never infer contents.

        Hosts register a listener with TracedTools.on_span and supply the record
        held in their store. Deadline polling still uses tick without any span.
        """
        from refund_agent.observability_reliability.spans import Span

        span = c._validated(Span, span)
        if span.end < span.start:
            raise ValueError("span ends before start")
        if span.name == "run":
            if observation is None or span.trace_id != "trace-" + observation.request_id:
                raise ValueError("run span requires its request observation")
            if observation.reason != span.reason:
                raise ValueError("observation differs from recorded run reason")
            self.observe(observation)
        elif span.name == "store:ledger":
            if payment is None or payment.receipt.receipt_id not in span.writes:
                raise ValueError("payment span requires its recorded payment")
            self.record_payment(payment)
        elif span.name == "correction":
            if correction is None or correction.payment_id not in span.reads:
                raise ValueError("correction span requires its correction record")
            self.record_correction(correction)
        self.tick(max(self.now, span.start), outcomes_final=self.outcomes_final)

    def close_windows(self, at: datetime) -> tuple[AlertRecord, ...]:
        before = len(self.alerts)
        self.tick(at, outcomes_final=self.outcomes_final)
        self._evaluate_windows(at)
        return tuple(self.alerts[before:])

    def _evaluate_windows(self, at: datetime, *, closed_only=False) -> None:
        for objective in self.record.objectives:
            window = timedelta(hours=objective.alert_window_hours)
            total = int((min(at, self.record.end) - self.record.start) // window)
            for index in range(total):
                key = objective.name, index
                if key in self._paged or closed_only and key not in self._closed:
                    continue
                start, end = (
                    self.record.start + index * window,
                    self.record.start + (index + 1) * window,
                )
                n, bad, pending = self._counts(objective.name, start, end)
                rho = 1 - fraction(objective.target)
                threshold = min_count(n, rho, objective.tolerance)
                alert = (
                    AlertRecord(
                        rule=objective.name,
                        window_start=start,
                        window_end=end,
                        count=bad,
                        threshold=threshold,
                        owner_roles=(self.record.owner_role,),
                    )
                    if bad >= threshold
                    else None
                )
                if key in self._closed and alert is None:
                    continue
                self._append(
                    "closed",
                    {
                        "objective": objective.name,
                        "index": index,
                        "eligible": n,
                        "pending": pending,
                        "false_page_probability": str(binom_tail(threshold, n, rho)),
                        "alert": alert.model_dump(mode="json") if alert else None,
                    },
                )
        # The slow 10%-of-allowance rule is a ticket, on disjoint 3-day blocks.
        window = timedelta(hours=72)
        total = int((min(at, self.record.end) - self.record.start) // window)
        objective = next(o for o in self.record.objectives if o.name == "timeliness")
        for index in range(total):
            key = "slow_ticket", index
            if key in self._paged or closed_only and key not in self._closed:
                continue
            start, end = (
                self.record.start + index * window,
                self.record.start + (index + 1) * window,
            )
            _, bad, _ = self._counts("timeliness", start, end)
            threshold = -(
                -fraction(objective.allowance).numerator
                // (10 * fraction(objective.allowance).denominator)
            )
            alert = (
                AlertRecord(
                    rule="slow_ticket",
                    window_start=start,
                    window_end=end,
                    count=bad,
                    threshold=threshold,
                    owner_roles=(self.record.owner_role,),
                    kind="ticket",
                )
                if bad >= threshold
                else None
            )
            if key in self._closed and alert is None:
                continue
            self._append(
                "closed",
                {
                    "objective": "slow_ticket",
                    "index": index,
                    "alert": alert.model_dump(mode="json") if alert else None,
                },
            )


def resume_canary(session, candidate, owner, settled, *, at):
    """Reload and deliver exhaustion before chapter 32 evaluates a resume."""
    from refund_agent.canary.cloud_native_deployment_component import (
        read_contracts,
    )
    from refund_agent.canary.cloud_native_deployment_stop import StopRule

    monitor = Monitor.for_canary(session, candidate)
    monitor.tick(max(monitor.now, at), outcomes_final=monitor.outcomes_final)
    receipts = tuple(session.store.records("ledger", read_contracts(candidate)).values())
    try:
        return StopRule(session, receipts).resume(owner, settled, at=at)
    finally:
        if session.outcome == "ended":
            Monitor.for_canary(session, candidate)
