"""Chapter 34's tracing mixin. Durations are constructed, never wall latency.

The wrapped operation executes once and its value is returned unchanged. Spans
carry record identifiers and sources; source/receipt contents stay in their stores.
"""

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from refund_agent import contracts as c
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools

# Constructed virtual seconds; nested operations run serially on this clock.
DURATIONS = {"step": 0.02, "tool": 0.10, "gate": 0.03, "store": 0.01}
Attempt = Annotated[int, Field(strict=True, gt=0)]


class SpanValues(TypedDict, total=False):
    trace_id: str
    run_id: str
    attempt: int
    bundle: str | None
    span_id: str
    parent_id: str | None
    name: str
    start: datetime
    end: datetime
    status: Literal["ok", "held", "refused", "error"]
    reason: str
    links: tuple[tuple[str, str], ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    requester_id: str
    sources: tuple[str, ...]
    tokens: int | None


# region book:fnd.observability-reliability.spans-01
class Span(c.Record):
    trace_id: str
    run_id: str
    attempt: Attempt
    bundle: str | None = None
    span_id: str
    parent_id: str | None
    name: str
    start: datetime
    end: datetime
    status: Literal["ok", "held", "refused", "error"]
    reason: str = ""
    links: tuple[tuple[str, str], ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    requester_id: str
    sources: tuple[str, ...] = ()
    tokens: c.Nonnegative | None = None

class TracedTools(Tools):
    def _trace_call(self, name, operation):
        with self._operation(name) as span:
            value = operation()
            self._result(span, value)
            return value
        # endregion book:fnd.observability-reliability.spans-01

    shadow_log: list[dict]
    shadow_reviews: dict[str, c.ReviewItem]

    def __init__(
        self, data, *, span_store, traced_mode, budget_monitor=None, **kwargs
    ):
        self.span_store = span_store
        self.budget_monitor = budget_monitor
        self._trace_request: c.Request | None = None
        self._trace_root: SpanValues | None = None
        self._trace_stack: list[SpanValues] = []
        self._trace_clock: datetime | None = None
        self._trace_links = ()
        self._trace_step = None
        self._trace_step_context = None
        self.traced_mode = traced_mode
        self._trace_bundle = kwargs["bundle"].fingerprint if "bundle" in kwargs else None
        self._span_sequence = len(span_store.entries)
        self._trace_listeners: list[Callable[[Span], None]] = []
        super().__init__(data, **kwargs)

    def begin_trace(self, request, at=None, *, links=()):
        if self._trace_request is not None:
            raise ValueError("trace run already open")
        self._trace_request = request
        self._trace_clock = c._utc_time(at or request.at)
        attempt = 1 + sum(
            s.name == "run"
            for s in self.span_store.trace(f"trace-{request.request_id}")
        )
        self._trace_root = {
            "trace_id": f"trace-{request.request_id}",
            "run_id": f"run-{request.request_id}-{attempt}",
            "attempt": attempt,
            "bundle": self._trace_bundle,
            "span_id": self._next_id(),
            "parent_id": None,
            "name": "run",
            "start": self._trace_clock,
            "requester_id": request.requester_id,
            "links": tuple(links),
        }

    def _next_id(self):
        self._span_sequence = (
            max(self._span_sequence, len(self.span_store.entries)) + 1
        )
        return f"observability-reliability-span-{self._span_sequence}"

    def _emit(self, span: Span):
        self.span_store.append(span)
        for listener in self._trace_listeners:
            listener(span)

    def on_span(self, listener: Callable[[Span], None]):
        self._trace_listeners.append(listener)

    def begin_step(self):
        if self._trace_step is not None:
            raise ValueError("trace step already open")
        self._trace_step_context = self._operation("step")
        self._trace_step = self._trace_step_context.__enter__()

    def finish_step(self, result=None):
        if self._trace_step is not None:
            if result is not None:
                self._result(self._trace_step, result)
            context = self._trace_step_context
            self._trace_step, self._trace_step_context = None, None
            context.__exit__(None, None, None)

    @contextmanager
    def _operation(self, name):
        if self._trace_request is None:
            raise ValueError(
                "tracing needs the before_run hook or begin_trace"
            )
        assert self._trace_root is not None and self._trace_clock is not None
        span: SpanValues = {
            **self._trace_root,
            "name": name,
            "span_id": self._next_id(),
            "parent_id": self._trace_stack[-1]["span_id"]
            if self._trace_stack
            else self._trace_root["span_id"],
            "start": self._trace_clock,
            "links": (),
            "status": "ok",
        }
        self._trace_stack.append(span)
        self._trace_clock += timedelta(
            seconds=DURATIONS[name.split(":", 1)[0]]
        )
        try:
            yield span
        except BaseException:
            span["status"] = "error"
            raise
        finally:
            self._trace_stack.pop()
            span["end"] = self._trace_clock
            self._emit(Span(**span))

    def _result(self, span: SpanValues, result):
        if isinstance(result, c.ToolResult):
            span["reason"] = result.reason
            span["status"] = (
                "ok"
                if result.ok
                else "held"
                if result.reason == "shipment_incomplete"
                else "refused"
            )
            for source, record in (
                ("order", result.order),
                ("shipment", result.shipment),
                ("policy", result.policy),
            ):
                if record is not None:
                    span["reads"] = (
                        fingerprint(record.model_dump(mode="json")),
                    )
                    span["sources"] = (source,)
            if result.record_id:
                span["writes"] = (result.record_id,)
        elif isinstance(result, c.Decision):
            span["reason"] = result.reason
            span["status"] = (
                "held" if result.reason == "shipment_incomplete" else "ok"
            )
        elif isinstance(result, str):
            span["reason"], span["status"] = result, "refused"

    def finish_trace(self, trace):
        if self._trace_request is None or self._trace_stack:
            raise ValueError("no run open or child operation still open")
        assert self._trace_root is not None and self._trace_clock is not None
        root: SpanValues = {
            **self._trace_root,
            "end": self._trace_clock,
            "status": "held"
            if trace.decision.reason == "shipment_incomplete"
            else "ok",
            "reason": trace.decision.reason,
        }
        self._emit(Span(**root))
        self._trace_request, self._trace_root, self._trace_clock = (
            None,
            None,
            None,
        )

    def _dispatch(self, call, state):
        if self._trace_request is None:
            return super()._dispatch(call, state)
        # Execution-time rereads belong to the executing tool's current step.
        if self._trace_stack and self._trace_stack[-1]["name"] != "step":
            return self._dispatch_traced(call, state)
        # Audit and named-review finalization have no preceding model reply.
        if call.name == "audit.append":
            self.finish_step(state.decision)
        if self._trace_step is None:
            self.begin_step()
        try:
            result = self._dispatch_traced(call, state)
            if self.budget_monitor is not None:
                from refund_agent.batch_stream.replay import PaymentRecord

                for mode, payments in (
                    ("executed", self.payments),
                    ("shadow", self.shadow_payments),
                ):
                    for payment in payments:
                        if payment.request is not None:
                            self.budget_monitor.record_payment(
                                PaymentRecord(payment, mode)
                            )
            self._result(self._trace_step, result)
            return result
        finally:
            self.finish_step()

    def assess(self, state):
        if self._trace_request is None or (
            self._trace_stack and self._trace_stack[-1]["name"] != "step"
        ):
            return super().assess(state)
        if self._trace_step is None:
            self.begin_step()
        try:
            if self.budget_monitor is not None:
                self.budget_monitor.tick(
                    max(
                        self.budget_monitor.now,
                        state.execution_at or state.request.at,
                    )
                )
            result = super().assess(state)
            self._result(self._trace_step, result)
            return result
        finally:
            self.finish_step()

    def _execute(self, call, state):
        if self.budget_monitor is not None:
            self.budget_monitor.tick(
                max(
                    self.budget_monitor.now,
                    state.execution_at or state.request.at,
                )
            )
        return super()._execute(call, state)

    def _dispatch_traced(self, call, state):
        before = len(self.shadow_log) if self.traced_mode == "shadow" else 0
        # A refused unknown call invokes none of the seven tools. Its failed
        # step remains observable without inventing a tool in the closed set.
        if call.name in (*c.PHASE_TO_TOOL.values(), "audit.append"):
            result = self._trace_call(
                "tool:" + call.name,
                lambda: super(TracedTools, self)._dispatch(call, state),
            )
        else:
            result = super()._dispatch(call, state)
        # Chapter 29 binds the shadow _write/_commit itself. Observe its public
        # recorded effects; never replace that payment or review binding.
        if self.traced_mode == "shadow":
            for entry in self.shadow_log[before:]:
                if entry["kind"] == "payment":
                    assert state.receipt is not None
                    with self._operation("store:ledger") as span:
                        span["writes"] = (state.receipt.receipt_id,)
                elif entry["kind"] == "review":
                    rows = tuple(
                        r
                        for r in self.shadow_reviews.values()
                        if r.proposal.proposal_id == entry["proposal_id"]
                    )
                    for row in rows:
                        with self._operation("store:reviews") as span:
                            span["writes"] = (row.review_id,)
        return result

    def _authorize(self, state, amount, approval):
        if self._trace_request is None:
            return super()._authorize(state, amount, approval)
        return self._trace_call(
            "gate",
            lambda: super(TracedTools, self)._authorize(
                state, amount, approval
            ),
        )

    def _write(self, name, key, record, keep=False):
        # Constructor imports of prior receipts precede any run and are not run work.
        if self._trace_request is None:
            return super()._write(name, key, record, keep=keep)
        with self._operation("store:" + name) as span:
            super()._write(name, key, record, keep=keep)
            if record is not None:
                span["writes"] = (record_id(record),)

    def _store_audit(self, record):
        if self.traced_mode != "shadow":
            return super()._store_audit(record)
        return self._trace_call(
            "store:audits",
            lambda: super(TracedTools, self)._store_audit(record),
        )

    def _save_proposal(self, state):
        if self.traced_mode != "shadow":
            return super()._save_proposal(state)
        return self._trace_call(
            "store:proposals",
            lambda: super(TracedTools, self)._save_proposal(state),
        )


def record_id(record):
    if isinstance(record, c.Receipt):
        return record.receipt_id
    if isinstance(record, c.AuditRecord):
        return record.audit_id
    if isinstance(record, c.Proposal):
        return record.proposal_id
    if isinstance(record, c.Approval):
        return record.approval_id
    if isinstance(record, c.ReviewItem):
        return record.review_id
    raise ValueError("unknown trace store record")


class TracedModel:
    """Open steps before replies, including replies that never call a tool."""

    def __init__(self, model, tools):
        self.model, self.tools = model, tools

    def reply(self, state):
        self.tools.finish_step(state.decision)
        self.tools.begin_step()
        try:
            return self.model.reply(state)
        except BaseException:
            self.tools._trace_step["status"] = "error"
            self.tools.finish_step()
            raise


class SpanStore:
    def __init__(self, path: Path | None = None):
        self.path, self._entries, self._ids = path, [], set()
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._add(Span.model_validate_json(line))

    def _add(self, span: Span):
        self._check(span)
        self._ids.add(span.span_id)
        self._entries.append(span)

    def _check(self, span: Span):
        if (
            span.span_id in self._ids
            or span.end < span.start
            or span.attempt < 1
        ):
            raise ValueError("duplicate span or invalid span timing/attempt")
        if span.name not in (
            "run",
            "step",
            "gate",
            "watermark_advance",
            "correction",
        ) and span.name not in (
            "tool:order.lookup",
            "tool:shipment.status",
            "tool:policy.retrieve",
            "tool:refund.propose",
            "tool:approval.request",
            "tool:refund.execute",
            "tool:audit.append",
            "store:ledger",
            "store:audits",
            "store:proposals",
            "store:reviews",
            "store:approvals",
            "store:lineage",
        ):
            raise ValueError("unknown span operation")
        if any(
            relation not in ("released_by", "corrects")
            for _, relation in span.links
        ):
            raise ValueError("unknown causal relation")

    def append(self, span: Span):
        span = c._validated(Span, span)
        self._check(span)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(span.model_dump_json() + "\n")
                stream.flush()
        self._add(span)

    @property
    def entries(self):
        return tuple(self._entries)

    def trace(self, trace_id: str):
        return tuple(s for s in self._entries if s.trace_id == trace_id)

    def by_record(self, record_id: str):
        return tuple(
            s for s in self._entries if record_id in s.reads + s.writes
        )

    def causal(self, request, at, name, *, reads=(), writes=(), links=()):
        roots = tuple(
            s
            for s in self.trace(f"trace-{request.request_id}")
            if s.name == "run"
        )
        attempt = max((s.attempt for s in roots), default=1)
        span = Span(
            trace_id=f"trace-{request.request_id}",
            run_id=f"run-{request.request_id}-{attempt}",
            attempt=attempt,
            bundle=roots[-1].bundle if roots else None,
            span_id=f"observability-reliability-causal-{len(self.entries) + 1}",
            parent_id=None,
            name=name,
            start=at,
            end=at,
            status="ok",
            reads=tuple(reads),
            writes=tuple(writes),
            links=tuple(links),
            requester_id=request.requester_id,
        )
        self.append(span)
        return span


def audit_coverage(spans, audits) -> dict:
    expected = {a.audit_id for a in audits}
    stored = {
        key for s in spans if s.name == "store:audits" for key in s.writes
    }
    return {
        "audits": len(expected),
        "covered": len(expected & stored),
        "missing": sorted(expected - stored),
        "pass": expected <= stored,
    }
