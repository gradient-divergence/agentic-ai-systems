"""Offline tools guard inputs, model replies, handouts and repeated/concurrent calls.

Stores may raise, drop writes/deletions, or retain/return altered records. Keyed
and scanned readback verifies every mutation; conflicts block named requests for
human reconciliation. Private access, function replacement and a process crash
between payment and confirmation are outside stage 1. Deployment needs a durable
ledger, provider idempotency and reconciliation with the audit log.
"""

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from hashlib import sha256
from threading import RLock
from typing import TypeVar, cast

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.contracts import _detach, _validated
from refund_agent.data.generate import Dataset
from refund_agent.rules import (
    approval_problem,
    assess,
    decision,
    execution_time,
    gate,
    proposal_problem,
    refresh_summaries,
    sources,
    tier_for,
)

TOOL_NAMES = (*c.PHASE_TO_TOOL.values(), "audit.append")
_Record = TypeVar("_Record", bound=c.Record)


class _Absent(RuntimeError):
    """A mutation was read back as absent, so it cannot be used."""


def _request_ids(value: object) -> set[str]:
    ids, seen, pending = set(), set(), [value]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, c.Record):
            rid = current.__dict__.get("request_id")
            pending.extend(current.__dict__.values())
        elif isinstance(current, dict):
            rid = dict.get(current, "request_id")
            pending.extend(dict.values(current))
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        else:
            continue
        if isinstance(rid, str):
            ids.add(str.__str__(rid))
    return ids


class PaymentError(RuntimeError):
    def __init__(self, receipt: c.Receipt | None, cause: BaseException) -> None:
        self.receipt, self.confirmed = _detach(receipt), receipt is not None
        self._owner: object = None
        self.trace: c.Trace | None = None
        message = "payment_outcome_unknown"
        if receipt:
            message = f"payment_recorded payment={receipt.receipt_id} cents={receipt.amount_cents}"
        super().__init__(f"{message} failure={type(cause).__name__}")


class AuditWriteError(RuntimeError):
    """The audit append was not acknowledged; receipt and trace permit reconciliation."""

    def __init__(self, record: c.AuditRecord, receipt: c.Receipt | None = None) -> None:
        self._record, self.receipt = _detach(record), _detach(receipt)
        self.expected, self.observed = _detach(record), None
        self._owner: object = None
        self.trace: c.Trace | None = None
        message = f"audit_{record.event}_failed trace={record.trace_id}"
        if receipt:
            message += f" payment={receipt.receipt_id}"
        super().__init__(message)


class ReconciliationError(PaymentError):
    """Storage is contradictory or unreadable; expected and observed are evidence."""

    def __init__(
        self, store: str, expected: object, observed: object, receipt: c.Receipt | None
    ) -> None:
        super().__init__(receipt, RuntimeError("reconciliation_required"))
        rid = receipt.request.request_id if receipt and receipt.request else None
        self._facts = _detach((store, expected, observed, rid))
        self.store, self.expected, self.observed = store, _detach(expected), _detach(observed)
        self.ledger_expected: dict[str, c.Receipt] = {}
        self.ledger_observed: dict[str, object] = {}
        payment = expected.receipt_id if isinstance(expected, c.Receipt) else None
        self.args = (f"reconciliation_required store={store} {self} expected_payment={payment}",)


class Tools:
    def __init__(self, data: Dataset) -> None:
        from refund_agent.release.agent_evaluation_resources import _bind_constructed_execution

        self._data = _validated(Dataset, data)
        _bind_constructed_execution(self)
        self._approvals: dict[str, c.Approval] = {}
        self._proposals: dict[str, c.Proposal] = {}
        self._reviews: dict[str, c.ReviewItem] = {}
        self._audits: dict[str, c.AuditRecord] = {}
        self._ledger: dict[str, c.Receipt] = {}
        self._expected: dict[str, dict[str, c.Record]] = {
            name: {} for name in ("ledger", "audits", "proposals", "reviews", "approvals")
        }
        # Public handouts are detached; these exact built-in stores stay private.
        self._owned = {name: getattr(self, f"_{name}") for name in self._expected}
        self._receipts: dict[str, c.Receipt] = {}
        self._request_receipts: dict[str, c.Receipt] = {}
        self._paid_by_order: dict[str, int] = {}
        self._blocked: dict[str, tuple[str, object, object, str | None]] = {}
        self._request_id: str | None = None
        self._context: tuple[c.State, str] | None = None
        self._lock = RLock()
        for order in self._data.orders:
            self._import_prior(order.order_id, order.prior_refunds)

    @property
    def data(self) -> Dataset:
        return _detach(self._data)

    @data.setter
    def data(self, value: Dataset) -> None:
        with self._lock:
            self._data = _validated(Dataset, value)

    @property
    def ledger(self) -> dict[str, c.Receipt]:
        with self._lock:
            return _detach(self._read("ledger", self._ledger))

    @property
    def audits(self) -> dict[str, c.AuditRecord]:
        with self._lock:
            return _detach(self._read("audits", self._audits))

    @property
    def proposals(self) -> dict[str, c.Proposal]:
        with self._lock:
            return _detach(self._read("proposals", self._proposals))

    @property
    def reviews(self) -> dict[str, c.ReviewItem]:
        with self._lock:
            return _detach(self._read("reviews", self._reviews))

    def _capture(self, observed: dict[str, object], name: str, read: Callable[[], object]) -> None:
        try:
            observed[name] = read()
        except BaseException as error:
            observed["error"] = type(error).__name__

    def _checked_record(self, name: str, row: object) -> c.Record:
        model = {
            "ledger": c.Receipt,
            "audits": c.AuditRecord,
            "proposals": c.Proposal,
            "reviews": c.ReviewItem,
            "approvals": c.Approval,
        }[name]
        if type(row) is not model:
            raise TypeError("invalid_store_record")
        return _validated(model, cast(c.Record, row))

    def _scan(self, name: str, rows: list[object]) -> dict[str, c.Record]:
        field = {
            "ledger": "idempotency_key",
            "audits": "audit_id",
            "proposals": "proposal_id",
            "reviews": "review_id",
            "approvals": "approval_id",
        }[name]
        records: dict[str, c.Record] = {}
        for row in rows:
            checked = self._checked_record(name, row)
            records[getattr(checked, field)] = checked
        return records

    def _observe(
        self, name: str, store: dict[str, _Record], expected: dict[str, _Record], key: str = ""
    ) -> dict[str, object]:
        observed: dict[str, object] = {}
        self._capture(observed, "items", lambda: dict(store.items()))
        self._capture(observed, "values", lambda: list(store.values()))
        try:
            values = cast(list[object], observed.get("values", []))
            observed["count"] = len(values)
            self._capture(observed, "scan", lambda: self._scan(name, values))
            items = cast(dict[object, object], observed.get("items", {}))
            keys = set(expected) | set(items) | ({key} if key else set())
            if any(type(k) is not str for k in keys):
                raise TypeError("invalid_store_key")
            checked_keys = cast(set[str], keys)
            self._capture(
                observed, "keyed", lambda: {k: store.get(k) for k in sorted(checked_keys)}
            )
        except BaseException as error:
            observed["error"] = type(error).__name__
        self._capture(observed, "contents", lambda: dict(dict.items(store)))
        self._capture(observed, "records", lambda: self._validate_views(name, observed))
        return observed

    def _validate_views(self, name: str, observed: dict[str, object]) -> dict[str, object]:
        records: dict[str, object] = {}
        for view in ("items", "contents", "keyed", "scan"):
            rows = cast(dict[object, object], observed.get(view, {}))
            if any(type(k) is not str for k in rows):
                raise TypeError("invalid_store_key")
            records[view] = {
                k: None
                if row is None and view == "keyed"
                else self._checked_record(name, row).model_dump()
                for k, row in rows.items()
            }
        return records

    def _matches(self, expected: dict[str, _Record], observed: dict[str, object]) -> bool:
        try:
            fields = {k: _validated(type(row), row).model_dump() for k, row in expected.items()}
            records = cast(dict[str, object], observed.get("records", {}))
            return (
                "error" not in observed
                and records["items"] == fields
                and records["scan"] == fields
                and records["contents"] == fields
                and observed["count"] == len(expected)
                and records["keyed"]
                == {k: fields.get(k) for k in cast(dict[str, object], records["keyed"])}
            )
        except BaseException as error:
            observed["error"] = type(error).__name__
            return False

    def _reconcile(self, name: str, expected: object, observed: object) -> ReconciliationError:
        ids = _request_ids(expected) | _request_ids(observed)
        if self._request_id is not None:
            ids.add(self._request_id)
        facts = _detach((name, expected, observed, self._request_id))
        for rid in sorted(ids):
            self._blocked.setdefault(rid, facts)
        return self._reconciliation_error(self._blocked.get(self._request_id or "", facts))

    def _reconciliation_error(
        self, facts: tuple[str, object, object, str | None]
    ) -> ReconciliationError:
        ledger = cast(dict[str, c.Receipt], self._expected["ledger"])
        observed = self._observe("ledger", self._ledger, ledger)
        receipt = None
        if self._matches(ledger, observed):
            receipt = next(
                (p for p in ledger.values() if p.request and p.request.request_id == facts[3]), None
            )
        error = ReconciliationError(*facts[:3], receipt)
        error._facts = _detach(facts)
        error.ledger_expected, error.ledger_observed = _detach(ledger), _detach(observed)
        error._owner = self
        return error

    def _read(self, name: str, store: dict[str, _Record]) -> dict[str, _Record]:
        expected = cast(dict[str, _Record], self._expected[name])
        if store is self._owned.get(name):
            return expected
        self._owned.pop(name, None)
        observed = self._observe(name, store, expected)
        if not self._matches(expected, observed):
            if "error" in observed and name != "ledger" and self._context:
                state, operation = self._context
                if self.payment_for(state.request.request_id) is None:
                    try:
                        self._decision(state, f"{operation}:{name}_read_error:{observed['error']}")
                    except BaseException:
                        pass  # An unavailable audit store cannot record the read failure.
            raise self._reconcile(name, expected, observed)
        return expected

    def _write(self, name: str, key: str, record: c.Record | None, keep: bool = False) -> None:
        execution = getattr(self, "_constructed_execution", None)
        if execution is not None and name == "ledger":
            execution.check(record.request if record is not None else None)
        store = cast(dict[str, c.Record], getattr(self, f"_{name}"))
        previous = self._expected[name]
        before = previous.get(key)
        wanted = _detach(previous.get(key, record) if keep and record is not None else record)
        error = None
        try:
            if record is None:
                store.pop(key, None)
            elif keep:
                store.setdefault(key, _detach(record))
            else:
                store[key] = _detach(record)
        except BaseException as cause:
            error = self._store_error(cause)
        expected = self._readback(name, store, previous, key, wanted, error)
        self._expected[name] = expected
        if name == "ledger":
            self._index_ledger(
                cast(c.Receipt | None, before), cast(c.Receipt | None, expected.get(key))
            )
        if error is not None:
            raise error

    def _readback(
        self,
        name: str,
        store: dict[str, c.Record],
        previous: dict[str, c.Record],
        key: str,
        record: c.Record | None,
        error: BaseException | None,
    ) -> dict[str, c.Record]:
        # Built-in dict views cannot disagree; replacement stores still verify all views.
        owned = store is self._owned.get(name) and error is None and store.get(key) == record
        expected = previous if owned else dict(previous)
        if record is None:
            expected.pop(key, None)
        else:
            expected[key] = record
        if owned:
            return expected
        self._owned.pop(name, None)
        observed = self._observe(name, store, expected, key)
        if self._matches(expected, observed):
            return expected
        if record is not None and key not in previous and self._matches(previous, observed):
            raise error or _Absent(f"{name}_write_not_recorded")
        raise self._reconcile(name, expected.get(key), observed) from error

    def _index_ledger(self, before: c.Receipt | None, after: c.Receipt | None) -> None:
        if before is not None:
            self._receipts.pop(before.receipt_id, None)
            if before.request is not None:
                self._request_receipts.pop(before.request.request_id, None)
            self._paid_by_order[before.order_id] -= before.amount_cents
        if after is not None:
            self._receipts[after.receipt_id] = after
            if after.request is not None:
                self._request_receipts[after.request.request_id] = after
            self._paid_by_order[after.order_id] = (
                self._paid_by_order.get(after.order_id, 0) + after.amount_cents
            )

    def _import_prior(self, order_id: str, prior: list[c.PriorRefund]) -> None:
        for payment in prior:
            self._read("ledger", self._ledger)
            existing = self._receipts.get(payment.payment_id)
            facts = (order_id, payment.amount_cents, payment.at)
            if existing and (existing.order_id, existing.amount_cents, existing.at) != facts:
                raise ValueError("prior_payment_conflict")
            if existing:
                continue
            key = f"prior-{payment.payment_id}"
            receipt = c.Receipt(
                receipt_id=payment.payment_id,
                idempotency_key=key,
                order_id=order_id,
                **payment.model_dump(include={"amount_cents", "at"}),
            )
            if key in self._read("ledger", self._ledger):
                raise ValueError("prior_payment_conflict")
            self._write("ledger", key, receipt)

    @property
    def payments(self) -> list[c.Receipt]:
        with self._lock:
            return _detach([p for p in self._read("ledger", self._ledger).values() if p.request])

    def paid_cents(self, order_id: str, at: datetime) -> int:
        if type(order_id) is not str:
            raise ValueError("order identifier must be an exact str")
        c._utc_time(at)
        with self._lock:
            self._read("ledger", self._ledger)
            return self._paid_by_order.get(order_id, 0)

    def payment_for(self, request_id: str) -> c.Receipt | None:
        """The only lookup used to report a request's payment facts."""
        if type(request_id) is not str:
            raise ValueError("request identifier must be an exact str")
        with self._lock:
            self._guard(request_id)
            previous, self._request_id = self._request_id, request_id
            try:
                self._read("ledger", self._ledger)
                return _detach(self._request_receipts.get(request_id))
            finally:
                self._request_id = previous

    def completed_state(self, request: c.Request) -> c.State | None:
        request = _validated(c.Request, request)
        query = c.State(request=request, trace_id=f"trace-{request.request_id}")
        with self._boundary(query, "completed_state"):
            receipt = query.receipt
            rid = request.request_id
            proposals = self._read("proposals", self._proposals).values() if receipt is None else ()
            record = receipt or next(
                (p for p in proposals if p.request_id == rid and p.request != request), None
            )
            if record is None:
                return None
            assert record.request is not None and record.evidence is not None
            state = c.State(
                **record.evidence.model_dump(),
                **record.model_dump(include={"request", "decision", "approvals", "proposal_id"}),
                execution_at=receipt.at if receipt else request.at,
                receipt=receipt,
            )
            state.policy_version = record.evidence.policy.version
            if record.request != request:
                state.failure_detail = "request_content_conflict"
                if receipt is None:
                    state.decision = decision(state, c.Outcome.REFER, state.failure_detail)
            return state

    def request_problem(self, request: c.Request) -> str | None:
        stored = self.completed_state(request)
        return stored.failure_detail if stored else None

    def run_time(self, request: c.Request) -> datetime:
        request = _validated(c.Request, request)
        state = c.State(request=request, trace_id=f"trace-{request.request_id}")
        with self._boundary(state, "run_time"):
            rid = request.request_id
            dates = [
                a.approved_at
                for a in self._read("approvals", self._approvals).values()
                if a.request_id == rid
            ]
            return max(request.at, *dates) + timedelta(microseconds=1) if dates else request.at

    def assess(self, state: c.State) -> c.Decision:
        state = _validated(c.State, state)
        self._guard(state.request.request_id)
        return _detach(assess(self._data, state))

    def audit_for(self, audit_id: str) -> c.AuditRecord | None:
        if type(audit_id) is not str:
            raise ValueError("audit identifier must be an exact str")
        with self._lock:
            return _detach(self._read("audits", self._audits).get(audit_id))

    def approval_for(self, state: c.State) -> c.Approval | None:
        state = _validated(c.State, state)
        with self._boundary(state, "approval_for"):
            proposal = self._read("proposals", self._proposals).get(state.proposal_id or "")
            if proposal is None or state.receipt:
                return None
            for _, approval in sorted(self._read("approvals", self._approvals).items()):
                if approval_problem(self._data, state, proposal, approval) is None:
                    return _detach(approval)
            return None

    def changed_approval(self, state: c.State) -> str | None:
        state = _validated(c.State, state)
        with self._boundary(state, "changed_approval"):
            if state.receipt:
                return None
            problems = [
                proposal_problem(
                    state,
                    self._read("proposals", self._proposals).get(a.proposal_id),
                    state.request.amount_cents,
                )
                for _, a in sorted(self._read("approvals", self._approvals).items())
                if a.request_id == state.request.request_id
            ]
            return None if None in problems else next((p for p in problems if p), None)

    def record_approval(self, approval: c.Approval) -> c.ToolResult:
        rid = getattr(approval, "request_id", None)
        if isinstance(rid, str):
            self._guard(rid)
        request = next((r for r in self._data.requests if r.request_id == rid), None)
        if request is None:
            request = next(
                (
                    p.request
                    for p in self._read("proposals", self._proposals).values()
                    if p.request_id == rid
                ),
                None,
            )
        if request is None:
            return c.ToolResult(ok=False, reason="invalid_approval")
        state = c.State(request=request, trace_id=f"trace-{request.request_id}")
        with self._boundary(state, "approval_intake"):
            try:
                approval = _validated(c.Approval, approval)
            except (ValidationError, AttributeError, TypeError):
                return self._decision(state, "invalid_approval")
            if state.receipt:
                return self._decision(state, "request_already_executed")
            try:
                return self._admit_approval(state, approval)
            except _Absent as error:
                return self._decision(state, str(error))

    def _admit_approval(self, state: c.State, approval: c.Approval) -> c.ToolResult:
        previous = self._read("approvals", self._approvals).get(approval.approval_id)
        if previous == approval:
            return c.ToolResult(ok=True, record_id=approval.approval_id)
        if previous:
            return self._decision(state, "approval_conflict")
        proposal = self._read("proposals", self._proposals).get(approval.proposal_id)
        review = self._read("reviews", self._reviews).get(f"review-{approval.proposal_id}")
        if proposal is None or review is None or review.proposal is None or not review.open:
            return self._decision(state, "open_review_required")
        state.__dict__.update(_detach(vars(proposal.evidence)))
        state.request, state.proposal_id = _detach(proposal.request), proposal.proposal_id
        state.approvals, state.execution_at = [approval], approval.approved_at
        problem = approval_problem(self._data, state, proposal, approval)
        if problem:
            return self._decision(state, problem)
        state.decision = decision(state, c.Outcome.REFER, "approval_recorded")
        return self._approve(state, review, approval)

    def _approve(self, state: c.State, review: c.ReviewItem, approval: c.Approval) -> c.ToolResult:
        try:
            closed = review.model_copy(update={"open": False})
            self._write("reviews", review.review_id, closed)
            self._checked_append(self._audit_record(state, approval.approved_at, "approval"))
            self._write("approvals", approval.approval_id, approval)
        except ReconciliationError:
            raise
        except BaseException:
            self._write("approvals", approval.approval_id, None)
            self._write("reviews", review.review_id, review)
            raise
        return c.ToolResult(ok=True, record_id=approval.approval_id)

    def call(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        if type(state) is not c.State or state.model_config.get("frozen"):
            return c.ToolResult(ok=False, reason="mutable_state_required")
        try:
            request = _validated(c.Request, state.request)
        except (ValidationError, AttributeError, TypeError):
            return c.ToolResult(ok=False, reason="invalid_tool_arguments")
        working = c.State(request=request, trace_id=f"trace-{request.request_id}")
        try:
            with self._boundary(working, f"tool_error:{getattr(call, 'name', 'unknown_tool')}"):
                try:
                    validated = _validated(c.State, state)
                    working.__dict__.update(vars(validated), receipt=working.receipt)
                    call = _validated(c.ToolCall, call)
                except (ValidationError, AttributeError, TypeError):
                    return self._decision(working, "invalid_tool_arguments")
                stored = self.completed_state(working.request)
                if stored and stored.failure_detail:
                    working.__dict__.update(vars(stored))
                    return self._decision(working, stored.failure_detail)
                result = self._dispatch(call, working)
                return _detach(result) if result.ok else self._decision(working, result.reason)
        finally:
            state.__dict__.update(_detach(vars(working)))
            state.__pydantic_fields_set__.update(vars(working))

    def _guard(self, request_id: str) -> None:
        if facts := self._blocked.get(request_id):
            raise self._reconciliation_error(facts)

    @contextmanager
    def _boundary(self, state: c.State, operation: str) -> Iterator[None]:
        # ponytail: one process lock; deployments need durable ledger/audit transactions.
        with self._lock:
            self._guard(state.request.request_id)
            previous = self._request_id, self._context
            parent = previous[1][0] if previous[1] else None
            state.receipt = (
                _detach(parent.receipt)
                if parent and parent.request.request_id == state.request.request_id
                else None
            )
            self._request_id, self._context = state.request.request_id, (state, operation)
            try:
                state.receipt = self.payment_for(state.request.request_id)
                yield
            except BaseException as error:
                error = self._store_error(error)
                if previous[1] and not isinstance(error, (AuditWriteError, PaymentError)):
                    state.receipt = self.payment_for(state.request.request_id)
                    if state.receipt is None:
                        raise error
                self._failed_boundary(state, operation, error)
            finally:
                self._request_id, self._context = previous

    def _failed_boundary(self, state: c.State, operation: str, error: BaseException) -> None:
        if isinstance(error, ReconciliationError):
            error = self._reconciliation_error(error._facts)
            state.receipt = error.receipt if error.confirmed else None
            state.decision = state.receipt.decision if state.receipt else None
            raise error
        state.receipt, state.decision = None, None
        state.receipt = self.payment_for(state.request.request_id)
        try:
            if not isinstance(error, AuditWriteError):
                name = type(error.__cause__ or error).__name__
                self._decision(state, f"{operation}:{name}")
        except ReconciliationError:
            raise
        except BaseException as failure:
            state.receipt, state.decision = None, None
            state.receipt = self.payment_for(state.request.request_id)
            error = self._store_error(failure)
        if isinstance(error, AuditWriteError):
            receipt = self.payment_for(error._record.decision.request_id)
            failure: AuditWriteError | PaymentError = AuditWriteError(error._record, receipt)
        elif state.receipt:
            failure = PaymentError(state.receipt, error)
        else:
            raise error
        failure._owner = self
        raise failure from error

    def _store_error(self, error: BaseException) -> BaseException:
        owned = getattr(error, "_owner", None) is self
        if isinstance(error, (AuditWriteError, PaymentError)) and not owned:
            return RuntimeError(type(error).__name__)
        return error

    def _dispatch(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        if state.execution_at is not None and state.execution_at < state.request.at:
            return c.ToolResult(ok=False, reason="execution_before_request")
        handlers = {
            "order.lookup": self._order,
            "shipment.status": self._shipment,
            "policy.retrieve": self._policy,
            "refund.propose": self._propose,
            "approval.request": self._review,
            "refund.execute": self._execute,
            "audit.append": lambda _call, state: self._decision(state),
        }
        if call.name not in handlers:
            return c.ToolResult(ok=False, reason="unknown_tool")
        if call.arguments.order_id is not None and (
            state.order is None or call.arguments.order_id != state.order.order_id
        ):
            return c.ToolResult(ok=False, reason="order_mismatch")
        try:
            return handlers[call.name](call, state)
        except _Absent as error:
            return c.ToolResult(ok=False, reason=str(error))

    def _order(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        # ponytail: scan 2,000 rows; index references if the corpus grows materially.
        reference = state.request.order_reference
        orders = [o for o in self._data.orders if reference in (o.order_id, o.reference)]
        if len(orders) != 1:
            return c.ToolResult(ok=False, reason="ambiguous_order" if orders else "order_not_found")
        state.order = _detach(orders[0])
        at = state.execution_at or state.request.at
        if any(p.at > at for p in state.order.prior_refunds):
            return c.ToolResult(ok=False, reason="future_prior_refund")
        self._import_prior(state.order.order_id, state.order.prior_refunds)
        state.refunded_cents = self.paid_cents(state.order.order_id, at)
        refresh_summaries(self._data, state)
        return c.ToolResult(ok=True, order=state.order, age_seconds=state.freshness["order"])

    def _shipment(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        if state.order is None:
            return c.ToolResult(ok=False, reason="missing_order")
        shipment = sources(self._data, state)[1]
        if shipment is None:
            return c.ToolResult(ok=False, reason="shipment_not_found")
        state.shipment_events = _detach(shipment.events)
        refresh_summaries(self._data, state)
        return c.ToolResult(ok=True, shipment=shipment, age_seconds=state.freshness["shipment"])

    def _policy(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        snapshot = sources(self._data, state)[2]
        policy = next(
            (p for p in self._data.policies if snapshot and p.version == snapshot.version), None
        )
        if snapshot is None or policy is None:
            return c.ToolResult(ok=False, reason="policy_not_found")
        state.policy = _detach(policy)
        refresh_summaries(self._data, state)
        return c.ToolResult(ok=True, policy=state.policy, age_seconds=state.freshness["policy"])

    def _propose(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        if state.order is None or state.policy is None:
            return c.ToolResult(ok=False, reason="missing_evidence")
        if call.arguments.amount_cents != state.request.amount_cents:
            return c.ToolResult(ok=False, reason="amount_mismatch")
        if call.arguments.reason != state.request.reason:
            return c.ToolResult(ok=False, reason="reason_mismatch")
        return self._save_proposal(state)

    def _save_proposal(self, state: c.State) -> c.ToolResult:
        refresh_summaries(self._data, state)
        assert state.order is not None and state.policy is not None and state.risk_score is not None
        bound = [state.request.model_dump(mode="json"), state.order.order_id, state.policy.version]
        digest = sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()[:24]
        proposal = c.Proposal(
            proposal_id=f"proposal-{digest}",
            **state.request.model_dump(include={"request_id", "amount_cents", "reason"}),
            request=state.request,
            created_at=state.execution_at or state.request.at,
            evidence=c.Evidence.model_validate(state, from_attributes=True),
        )
        self._write("proposals", proposal.proposal_id, proposal, keep=True)
        state.proposal_id = proposal.proposal_id
        return c.ToolResult(ok=True, record_id=proposal.proposal_id)

    def _request_review(self, state: c.State, reason: str) -> c.ToolResult:
        from refund_agent.agent import _open_named_review

        caller = sys._getframe(1)
        if (
            caller.f_code is not _open_named_review.__code__
            or caller.f_globals is not _open_named_review.__globals__
        ):
            raise ValueError("request-only review requires core finalization")
        if state.receipt or state.order is not None or state.request.amount_cents <= 50000:
            raise ValueError("request-only review requires an unresolved tier 3 request")
        policy = self._data.policy_at(state.execution_at or state.request.at)
        bound = [state.request.model_dump(mode="json"), policy.version]
        digest = sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()[:24]
        review = c.ReviewItem(
            review_id=f"review-request-{digest}",
            request=_detach(state.request),
            named_approver=policy.named_approver,
            required_role="named_approver",
            policy_reason=reason,
        )
        return self._store_review(review, state)

    def _named_reviews(self) -> dict[str, c.ReviewItem]:
        return self._read("reviews", self._reviews)

    def _store_review(self, review: c.ReviewItem, state: c.State) -> c.ToolResult:
        caller = sys._getframe(1)
        issuers = (Tools._request_review, Tools._review)
        while caller is not None and not any(
            caller.f_code is issuer.__code__ and caller.f_globals is issuer.__globals__
            for issuer in issuers
        ):
            caller = caller.f_back
        if caller is None:
            raise ValueError("review storage requires core issuance")
        self._write("reviews", review.review_id, review)
        return c.ToolResult(ok=True, record_id=review.review_id)

    def _review(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        if state.receipt:
            return c.ToolResult(ok=False, reason="request_already_executed")
        proposal = self._read("proposals", self._proposals).get(state.proposal_id or "")
        if proposal is None or proposal.request != state.request:
            return c.ToolResult(ok=False, reason="proposal_required")
        tier = tier_for(proposal.amount_cents)
        role = "named_approver" if tier == c.Tier.NAMED_APPROVER else "reviewer"
        review = c.ReviewItem(
            review_id=f"review-{proposal.proposal_id}",
            proposal=_detach(proposal),
            required_role=role,
            policy_reason=self.assess(state).reason,
        )
        if not self.approval_for(state):
            return self._store_review(review, state)
        return c.ToolResult(ok=True, record_id=review.review_id)

    def _audit_record(self, state: c.State, at: datetime, event: c.AuditEvent) -> c.AuditRecord:
        receipt = state.receipt = self.payment_for(state.request.request_id)
        if receipt:
            assert receipt.decision and receipt.evidence and receipt.request
            state.decision = receipt.decision
            state.__dict__.update(_detach(vars(receipt.evidence)))
            state.approvals = _detach(list(receipt.approvals))
            state.proposal_id, state.request = receipt.proposal_id, _detach(receipt.request)
            state.policy_version = receipt.evidence.policy.version
        assert state.decision is not None
        record = c.AuditRecord(
            audit_id="",
            **state.model_dump(include=set(c.AuditRecord.model_fields) - {"decision"}),
            at=at,
            decision=state.decision,
            approval_ids=[a.approval_id for a in state.approvals],
            event=event,
            money_moved=receipt is not None,
            paid_amount_cents=receipt.amount_cents if receipt else 0,
            receipt_id=receipt.receipt_id if receipt else None,
        )
        encoded = json.dumps(record.model_dump(mode="json"), sort_keys=True).encode()
        digest = sha256(encoded).hexdigest()[:24]
        return record.model_copy(update={"audit_id": f"audit-{digest}"})

    def _store_audit(self, record: c.AuditRecord) -> c.ToolResult:
        self._write("audits", record.audit_id, record, keep=True)
        return c.ToolResult(ok=True, record_id=record.audit_id)

    def _checked_append(self, record: c.AuditRecord) -> c.ToolResult:
        try:
            return self._store_audit(record)
        except ReconciliationError:
            raise
        except BaseException as error:
            failure = AuditWriteError(record)
            failure._owner = self
            raise failure from error

    def record_decision(self, state: c.State) -> c.ToolResult:
        return self.call(c.ToolCall(name="audit.append"), state)

    def _decision(self, state: c.State, reason: str | None = None) -> c.ToolResult:
        receipt = state.receipt
        chosen = receipt.decision if receipt else state.decision
        if reason:
            chosen = receipt.decision if receipt else decision(state, c.Outcome.REFER, reason)
            state.failure_detail = reason
        if chosen is None or chosen.request_id != state.request.request_id:
            return c.ToolResult(ok=False, reason="decision_required")
        if chosen.executed and receipt is None:
            return c.ToolResult(ok=False, reason="execution_not_recorded")
        event: c.AuditEvent = "confirmation" if receipt else "decision"
        state.decision = chosen
        record = self._audit_record(state, state.execution_at or state.request.at, event)
        result = self._checked_append(record)
        return result.model_copy(update={"ok": False, "reason": reason}) if reason else result

    def _authorize(self, state: c.State, amount: int, approval: c.Approval | None) -> str | None:
        assert state.order is not None
        proposal = self._read("proposals", self._proposals).get(state.proposal_id or "")
        refunded = self.paid_cents(state.order.order_id, state.execution_at or state.request.at)
        return gate(self._data, state, amount, proposal, approval, refunded)

    def _execute(self, call: c.ToolCall, state: c.State) -> c.ToolResult:
        args = call.arguments
        if args.idempotency_key is None or not args.idempotency_key.strip():
            return c.ToolResult(ok=False, reason="idempotency_key_required")
        if args.amount_cents is None or state.order is None:
            return c.ToolResult(ok=False, reason="missing_execution_arguments")
        if previous := self._read("ledger", self._ledger).get(args.idempotency_key):
            approved = previous.approvals[0].approval_id if previous.approvals else None
            bound = (previous.request, previous.order_id, previous.amount_cents, approved)
            supplied = (state.request, state.order.order_id, args.amount_cents, args.approval_id)
            if bound != supplied:
                return c.ToolResult(ok=False, reason="idempotency_conflict")
            assert state.receipt is not None
            return c.ToolResult(ok=True, record_id=state.receipt.receipt_id)
        if state.receipt:
            return c.ToolResult(ok=False, reason="request_already_executed")
        approval = self._read("approvals", self._approvals).get(args.approval_id or "")
        if args.approval_id is not None and approval is None:
            return c.ToolResult(ok=False, reason="invalid_approval")
        state.execution_at = execution_time(state, approval)
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            result = self._dispatch(c.ToolCall(name=name), state)
            if not result.ok:
                return result
        problem = self._authorize(state, args.amount_cents, approval)
        if problem:
            if problem.startswith("proposal_changed:"):
                self._save_proposal(state)
                self._review(c.ToolCall(name="approval.request"), state)
            return c.ToolResult(ok=False, reason=problem)
        return self._commit(state, args.idempotency_key, args.amount_cents, approval)

    # region book:fnd.action-guardrails.commit-01
    def _commit(
        self, state: c.State, key: str, amount: int, approval: c.Approval | None
    ) -> c.ToolResult:
        assert state.order is not None and state.execution_at is not None
        state.approvals = [approval] if approval else []
        state.decision = decision(state, c.Outcome.APPROVE, "eligible").model_copy(
            update={"amount_cents": amount, "tier": tier_for(amount)}
        )
        self._checked_append(self._audit_record(state, state.execution_at, "intent"))
        receipt = c.Receipt(
            receipt_id=f"refund-{state.request.request_id}",
            **state.model_dump(include={"request", "proposal_id", "approvals"}),
            order_id=state.order.order_id,
            amount_cents=amount,
            idempotency_key=key,
            at=state.execution_at,
            decision=state.decision.model_copy(update={"executed": True}),
            evidence=c.Evidence.model_validate(state, from_attributes=True),
        )
        self._read("ledger", self._ledger)
        if receipt.receipt_id in self._receipts:
            raise ValueError("payment_identity_conflict")
        self._write("ledger", key, receipt)
        state.receipt = self.payment_for(state.request.request_id)
        self._decision(state)
        assert state.receipt is not None
        return c.ToolResult(ok=True, record_id=state.receipt.receipt_id)

    # endregion book:fnd.action-guardrails.commit-01
