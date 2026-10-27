"""Persisted authority; one irreversible end transition, and resumable pauses."""

from datetime import datetime, timedelta
from typing import Literal

from refund_agent import contracts as c
from refund_agent.batch_stream.replay import CorrectionRecord
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, has_role
from refund_agent.canary.cloud_native_deployment_store import INTEGRITY_ERRORS, AppendStore
from refund_agent.selection import fingerprint


class Grant(c.Record):
    canary_id: str
    plan_fingerprint: str
    window_fingerprint: str
    started_at: datetime
    expires_at: datetime
    outcome: Literal["running", "paused", "ended"]
    lane: Literal["canary", "rehearsal"]
    budget_fingerprint: str | None = None
    budget_spent: bool = False
    payment_ids: tuple[str, ...] = ()
    rechecks: dict[str, datetime] = {}
    residuals: tuple[str, ...] = ()
    stop_id: str | None = None


class StopRecord(c.Record):
    stop_id: str
    canary_id: str
    at: datetime
    trigger: str
    payment_id: str | None = None
    payments_after_trigger: int
    cents_after_trigger: int
    outcome: Literal["paused", "resumed", "ended"]
    detail: str = ""


class SettledEvidence(c.Record):
    payment_id: str | None = None
    final: bool
    allowed_on_read: bool
    stale_data: bool = False
    late_scan: bool = False


class CanarySession:
    def __init__(self, plan: CanaryPlan, store: AppendStore):
        self.plan, self.store = c._validated(CanaryPlan, plan), store
        self._untrusted_stop: StopRecord | None = None
        self._stopping_untrusted = False

    @property
    def grant(self) -> Grant | None:
        rows = [r["value"] for r in self.store.rows("authority") if r["key"] == self.plan.canary_id]
        if not rows:
            return None
        previous = None
        for raw in rows:
            result = Grant.model_validate_json(encode_json(raw))
            if previous is not None and (
                result.started_at != previous.started_at or result.expires_at != previous.expires_at
                or result.window_fingerprint != previous.window_fingerprint
                or result.budget_fingerprint != previous.budget_fingerprint
                or result.lane != previous.lane
                or previous.outcome == "ended" and result.outcome != "ended"
                or previous.budget_spent and not result.budget_spent
                or not set(previous.payment_ids) <= set(result.payment_ids)
            ):
                raise ValueError("persisted grant cannot renew or clear spent authority")
            previous = result
        if result.plan_fingerprint != self.plan.fingerprint or result.canary_id != self.plan.canary_id:
            raise ValueError("persisted plan authority differs")
        if result.expires_at != result.started_at + timedelta(days=self.plan.time_box_days):
            raise ValueError("persisted expiry differs from original plan")
        return result

    def verify_window_authority(self):
        from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult

        grant = self.grant
        if grant is None or grant.lane == "rehearsal":
            return
        raw = self.store.latest("window", self.plan.requirement_ref)
        if raw is None:
            raise ValueError("persisted window authority is absent")
        try:
            window = ShadowWindowResult.model_validate_json(encode_json(raw))
        except ValueError as error:
            raise ValueError("persisted window authority is invalid") from error
        if window.fingerprint != grant.window_fingerprint or window.window_id != self.plan.requirement_ref or (
            not window.measures_pass or not window.signed_by
            or window.candidate_fingerprint != self.plan.candidate_fingerprint
            or window.signed_at is None or window.signed_at > grant.started_at
        ):
            raise ValueError("persisted window authority differs from the original grant")
        budget = self.store.latest("budget", self.plan.error_budget_ref)
        if budget is None or fingerprint(budget) != grant.budget_fingerprint:
            raise ValueError("persisted error budget authority differs")

    @property
    def outcome(self):
        if self._untrusted_stop is not None or self._stopping_untrusted:
            return "paused"
        grant = self.grant
        return grant.outcome if grant else "shadow"

    @property
    def stops(self):
        if self._untrusted_stop is not None:
            return (self._untrusted_stop,)
        return tuple(
            StopRecord.model_validate_json(encode_json(r["value"]))
            for r in self.store.rows("stop")
            if r["value"]["canary_id"] == self.plan.canary_id
        )

    def untrusted_record(self, at):
        """Close payment authority even when the file cannot accept its stop.

        Never append to a broken chain. Keep the stop in this session until a
        trusted store can persist it; a fresh process also refuses that file.
        """
        if self._stopping_untrusted:
            return
        self._stopping_untrusted = True
        self._untrusted_stop = None
        try:
            grant = self.grant
            if grant is not None and grant.outcome != "ended":
                StopRule(self).trigger("untrusted_record", at=at)
        except INTEGRITY_ERRORS:
            self._untrusted_stop = StopRecord(
                stop_id="cloud-native-deployment-untrusted-record",
                canary_id=self.plan.canary_id, at=at, trigger="untrusted_record",
                payments_after_trigger=0, cents_after_trigger=0, outcome="paused",
                detail="store cannot persist stop; payment authority closed",
            )
        finally:
            self._stopping_untrusted = False

    @property
    def pending_stops(self):
        pending = []
        for row in self.stops:
            if row.outcome == "paused":
                pending.append(row)
            else:
                pending.clear()
        return tuple(pending)

    def save(self, grant: Grant):
        previous = self.grant
        if previous is not None and previous.outcome == "ended" and grant.outcome != "ended":
            raise ValueError("spent grant cannot restart")
        self.store.append("authority", self.plan.canary_id, grant.model_dump(mode="json"))

    def note_payment(self, receipt: c.Receipt):
        with self.store.transaction():
            grant = self.grant
            if grant is not None and receipt.receipt_id not in grant.payment_ids:
                self.save(grant.model_copy(update={
                    "payment_ids": grant.payment_ids + (receipt.receipt_id,),
                }))

    def tick(self, at: datetime, receipts):
        at = c._utc_time(at)
        if self._untrusted_stop is not None:
            self.untrusted_record(self._untrusted_stop.at)
            if self._untrusted_stop is not None:
                return
        grant = self.grant
        if grant is None or grant.outcome == "ended":
            return
        rule = StopRule(self, receipts)
        if at >= grant.expires_at:
            rule.end("time_box", at=at)
        elif grant.budget_spent:
            rule.end("budget_spent", at=at)
        elif grant.outcome == "running":
            overdue = sorted(
                (p for p in receipts if p.receipt_id in grant.payment_ids
                 and p.receipt_id not in grant.rechecks
                 and at >= p.at + timedelta(hours=self.plan.recheck_hours)),
                key=lambda p: (p.at, p.receipt_id),
            )
            if overdue:
                rule.trigger("recheck_overdue", at=at, payment_id=overdue[0].receipt_id)


def encode_json(value):
    # Static JSON conversion; no dynamic module loader is used.
    import json

    return json.dumps(value)


class StopRule:
    def __init__(self, session: CanarySession, receipts=()):
        self.session = session
        self.receipts = tuple(c._validated(c.Receipt, p) for p in receipts)

    def _record(self, trigger, at, payment_id, outcome, detail=""):
        trigger_payment = next((p for p in self.receipts if p.receipt_id == payment_id), None)
        grant = self.session.grant
        if grant is None:
            raise ValueError("stop record requires a started grant")
        after = [
            p for p in self.receipts
            if trigger_payment and p.receipt_id in grant.payment_ids
            and trigger_payment.at < p.at <= at and p.receipt_id != payment_id
        ]
        row = StopRecord(
            stop_id=f"cloud-native-deployment-stop-{len(self.session.stops) + 1}",
            canary_id=self.session.plan.canary_id, at=at, trigger=trigger,
            payment_id=payment_id, payments_after_trigger=len(after),
            cents_after_trigger=sum(p.amount_cents for p in after), outcome=outcome, detail=detail,
        )
        self.session.store.append("stop", row.stop_id, row.model_dump(mode="json"))
        return row

    def trigger(self, trigger, *, at: datetime, payment_id=None, owner=None, detail=""):
        if trigger not in self.session.plan.stop_triggers:
            raise ValueError("unknown canary pause trigger")
        if trigger == "owner" and not has_role(owner, "release_owner"):
            raise ValueError("owner stop requires a listed release owner")
        with self.session.store.transaction():
            grant = self.session.grant
            if grant is None or grant.outcome == "ended":
                raise ValueError("canary has no remaining authority")
            if trigger in ("correction", "recheck", "recheck_overdue") and (
                payment_id not in grant.payment_ids
                or not any(p.receipt_id == payment_id for p in self.receipts)
            ):
                raise ValueError("pause must name an existing cohort payment")
            duplicate = next((s for s in self.session.pending_stops
                              if (s.trigger, s.payment_id, s.detail) == (trigger, payment_id, detail)), None)
            if duplicate is not None:
                return duplicate
            row = self._record(trigger, c._utc_time(at), payment_id, "paused", detail)
            self.session.save(grant.model_copy(update={"outcome": "paused", "stop_id": row.stop_id}))
            return row

    def end(self, trigger, *, at: datetime, detail=""):
        if trigger not in ("time_box", "budget_spent", "defect"):
            raise ValueError("unknown canary end trigger")
        with self.session.store.transaction():
            grant = self.session.grant
            if grant is None:
                raise ValueError("canary has not started")
            if grant.outcome == "ended":
                return self.session.stops[-1]
            payment_id = self.session.stops[-1].payment_id if grant.stop_id else None
            row = self._record(trigger, c._utc_time(at), payment_id, "ended", detail)
            self.session.save(grant.model_copy(update={
                "outcome": "ended", "stop_id": row.stop_id,
                "budget_spent": grant.budget_spent or trigger == "budget_spent",
            }))
            return row

    def rechecked(self, payment_id: str, *, at: datetime, owner: str):
        if not has_role(owner, "release_owner"):
            raise ValueError("recheck requires a listed person")
        with self.session.store.transaction():
            grant = self.session.grant
            payment = next((p for p in self.receipts if p.receipt_id == payment_id), None)
            if grant is None or payment_id not in grant.payment_ids or payment is None or at < payment.at:
                raise ValueError("recheck does not name an existing cohort payment")
            self.session.save(grant.model_copy(update={"rechecks": {**grant.rechecks, payment_id: at}}))

    def correction(self, record: CorrectionRecord, *, at=None):
        grant = self.session.grant
        if record.mode != "executed" or grant is None or record.payment_id not in grant.payment_ids:
            return None
        receipt = next((p for p in self.receipts if p.receipt_id == record.payment_id), None)
        if receipt is None or receipt.request.request_id != record.request_id or receipt.order_id != record.order_id:
            raise ValueError("correction does not match its executed receipt")
        with self.session.store.transaction():
            key = fingerprint([record.payment_id, record.event_id, record.arrived_at.isoformat()])
            if self.session.store.latest("correction", key) is not None:
                return None
            row = self.trigger("correction", at=at or record.arrived_at,
                               payment_id=record.payment_id, detail=record.event_id)
            self.session.store.append("correction", key, row.model_dump(mode="json"))
            return row

    def resume(self, release_owner: str, settled, *, at: datetime,
               budget_spent=False):
        supplied = settled if isinstance(settled, (tuple, list)) else (settled,)
        supplied = tuple(c._validated(SettledEvidence, row) for row in supplied)
        at = c._utc_time(at)
        if not has_role(release_owner, "release_owner"):
            raise ValueError("resume requires release_owner")
        with self.session.store.transaction():
            grant = self.session.grant
            if grant is None or grant.outcome != "paused":
                raise ValueError("only a paused, unspent grant may resume")
            self.session.verify_window_authority()
            if at >= grant.expires_at:
                self.end("time_box", at=at)
                raise ValueError("original grant expired")
            if budget_spent or grant.budget_spent:
                self.end("budget_spent", at=at)
                raise ValueError("canary budget spent")
            previous = self.session.stops[-1]
            pending = self.session.pending_stops
            required = {row.payment_id for row in pending}
            by_payment = {row.payment_id: row for row in supplied}
            if (set(by_payment) != required or len(by_payment) != len(supplied)
                or not all(row.final for row in supplied)
                or any(at < row.at for row in pending)):
                raise ValueError("settled evidence for every pending stop required")
            if any(not row.allowed_on_read or row.stale_data for row in supplied):
                self.end("defect", at=at)
                raise ValueError("defect permanently ended canary")
            if any(p.receipt_id in grant.payment_ids and p.receipt_id not in grant.rechecks
                   and at >= p.at + timedelta(hours=self.session.plan.recheck_hours)
                   for p in self.receipts):
                raise ValueError("every overdue recheck must be completed")
            residuals = grant.residuals
            for settled in supplied:
                corrected = any(row.trigger == "correction" and row.payment_id == settled.payment_id
                                for row in pending)
                if settled.payment_id is not None and (settled.late_scan or corrected) and settled.payment_id not in residuals:
                    residuals += (settled.payment_id,)
                    self.session.store.append("reconciliation", settled.payment_id, {
                        "payment_id": settled.payment_id, "reason": "late_scan",
                        "count_against_error_budget": True,
                    })
            row = self._record(previous.trigger, at, previous.payment_id, "resumed")
            self.session.save(grant.model_copy(update={
                "outcome": "running", "residuals": residuals, "stop_id": row.stop_id,
            }))
            return row
