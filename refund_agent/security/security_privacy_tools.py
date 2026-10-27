"""Outermost scope boundary, including the wrapped tools' run-time snapshots.

Issuer, resolver, verifier and tool code are trusted in-process teaching code.
Operating-system isolation and authenticated intake are deployment controls.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from refund_agent import contracts as c
from refund_agent.rules import decision
from refund_agent.security.security_privacy_accounts import AccountRecords, ScopeResolver
from refund_agent.security.security_privacy_check import reviewer_problem
from refund_agent.security.security_privacy_credentials import (
    Credential,
    Revocations,
    Verifier,
    credential_problem,
    identity_problem,
)
from refund_agent.security.security_privacy_log import CredentialEntry, CredentialLog
from refund_agent.tools import Tools


@dataclass(frozen=True)
class SecurityRuntime:
    accounts: AccountRecords
    resolver: ScopeResolver
    verifier: Verifier
    revocations: Revocations
    log: CredentialLog


class ScopedTools(Tools):
    version = "security-privacy.scoped@1"

    def __init__(
        self,
        data,
        *,
        security_privacy: SecurityRuntime,
        security_credential: Credential | None = None,
        **kwargs,
    ):
        self.security_privacy = security_privacy
        self.security_credential = security_credential
        self.security_run_id: str | None = None
        self.security_clock: datetime | None = None
        self._security_approval_ids: set[str] = set()
        self._security_rereading = False
        super().__init__(data, **kwargs)

    def _allowed_orders(self, request):
        credential = self.security_credential
        if (
            credential is None
            or credential.subject != request.requester_id
            or credential.request_id != request.request_id
        ):
            return ()
        at = self.security_clock or request.at
        if identity_problem(credential, at, self.security_privacy.verifier):
            return ()
        if self.security_privacy.revocations.contains(credential, "order.lookup", at):
            return ()
        return tuple(
            o
            for o in credential.order_ids
            if self.security_privacy.accounts.may_act(credential.subject, o)
        )

    def bind_run_record(self, record):
        """Consume chapter 34's serialized run Span, without copying payloads."""
        credential = self.security_credential
        if (
            credential is None
            or record["name"] != "run"
            or (
                record["trace_id"] != f"trace-{credential.request_id}"
                or record["requester_id"] != credential.subject
                or not record["run_id"]
            )
        ):
            raise ValueError("security_privacy_run_context_differs")
        self.security_run_id = record["run_id"]

    @contextmanager
    def _scoped_data(self, state, view):
        saved = self._data
        execution = self._constructed_execution if hasattr(self, "_constructed_execution") else None
        if execution is not None:
            execution.check(state.request)
            snapshot = execution.snapshot
        self._data = view
        if execution is not None:
            # The constructed run reads this scoped view until it is restored.
            execution.data, execution.snapshot = view, view.model_dump()
        try:
            yield
        finally:
            try:
                if execution is not None:
                    execution.check(state.request)
            finally:
                self._data = saved
                if execution is not None:
                    execution.data, execution.snapshot = saved, snapshot
                    execution.check(state.request)

    @contextmanager
    def _boundary(self, state, operation):
        # A1: chapter 30 snapshots at run entry, before a tool is dispatched.
        with self._lock:
            saved = self._data
            ids = self._allowed_orders(state.request)
            view = saved.model_copy(
                update={
                    "orders": [o for o in saved.orders if o.order_id in ids],
                    "shipments": [s for s in saved.shipments if s.order_id in ids],
                    "requests": [
                        r for r in saved.requests if r.request_id == state.request.request_id
                    ],
                    "retrievals": [
                        s for s in saved.retrievals if s.request_id == state.request.request_id
                    ],
                    "correct": [],
                }
            )
            with self._scoped_data(state, view):
                with super()._boundary(state, operation):
                    yield

    def _problem(self, call, state):
        runtime = self.security_privacy
        return credential_problem(
            self.security_credential,
            call,
            state,
            state.execution_at or state.request.at,
            runtime.verifier,
            runtime.revocations,
        )

    def _record_credential(self, call, state, reason="", *, credential=None, use_run=True):
        credential = self.security_credential if use_run else credential
        self.security_privacy.log.append(
            CredentialEntry(
                credential_id=credential.credential_id if credential else None,
                kind=credential.kind if credential else None,
                subject=credential.subject if credential else None,
                request_id=state.request.request_id,
                trace_id=state.trace_id,
                run_id=self.security_run_id,
                tool=call.name,
                outcome="refused" if reason else "allowed",
                reason=reason,
                at=state.execution_at or state.request.at,
            )
        )

    def call(self, call, state):
        call, checked = c._validated(c.ToolCall, call), c._validated(c.State, state)
        # Preserve the base's order_mismatch before checking a supplied target.
        common = call.model_copy(
            update={
                "arguments": call.arguments.model_copy(
                    update={"order_id": None, "amount_cents": None}
                )
            }
        )
        problem = self._problem(common, checked)
        if (
            problem is None
            and call.arguments.order_id is not None
            and (checked.order is None or checked.order.order_id != call.arguments.order_id)
        ):
            problem = "order_mismatch"
        if problem is None and call.name == "order.lookup":
            _, problem = self.security_privacy.resolver.resolve(
                checked.request.order_reference, self._allowed_orders(checked.request)
            )
        if (
            problem is None
            and checked.order is not None
            and checked.order.order_id not in self._allowed_orders(checked.request)
        ):
            problem = "order_not_found"
        if problem is None:
            problem = self._problem(
                call.model_copy(
                    update={"arguments": call.arguments.model_copy(update={"order_id": None})}
                ),
                checked,
            )
        self._record_credential(call, checked, problem or "")
        if problem:
            with self._boundary(state, "security-privacy-refusal"):
                return self._decision(state, problem)
        with self._boundary(state, "security-privacy-call"):
            return super().call(call, state)

    def completed_state(self, request):
        query = c.State(
            request=request,
            trace_id=f"trace-{request.request_id}",
            execution_at=self.security_clock,
        )
        if self._problem(c.ToolCall(name="order.lookup"), query):
            return None
        _, problem = self.security_privacy.resolver.resolve(
            request.order_reference, self._allowed_orders(request)
        )
        if problem:
            return None
        # Stored continuations are also disclosures, even before order.lookup.
        self._read("ledger", self._ledger)
        receipt = self._request_receipts.get(request.request_id)
        proposals = self._read("proposals", self._proposals).values()
        records = (
            [receipt] if receipt else [p for p in proposals if p.request_id == request.request_id]
        )
        if any(
            row.request.requester_id != request.requester_id
            or (row.order_id if isinstance(row, c.Receipt) else row.evidence.order.order_id)
            not in self._allowed_orders(request)
            for row in records
            if row.request is not None
        ):
            return None
        return super().completed_state(request)

    def payment_for(self, request_id):
        receipt = super().payment_for(request_id)
        credential = self.security_credential
        if receipt and (
            credential is None
            or credential.request_id != request_id
            or receipt.request is None
            or credential.subject != receipt.request.requester_id
            or receipt.order_id not in self._allowed_orders(receipt.request)
        ):
            return None
        return receipt

    def assess(self, state):
        with self._boundary(state, "security-privacy-assess"):
            chosen = super().assess(state)
            if chosen.outcome != c.Outcome.APPROVE:
                return chosen
            call = c.ToolCall(
                name="refund.propose",
                arguments=c.ToolArguments(amount_cents=state.request.amount_cents),
            )
            problem = self._problem(call, state)
            return decision(state, c.Outcome.REFER, problem) if problem else chosen

    def _authorize(self, state, amount, approval):
        problem = super()._authorize(state, amount, approval)
        if problem:
            return problem
        if approval and approval.approval_id not in self._security_approval_ids:
            return "credential_reviewer"
        return self._problem(
            c.ToolCall(name="refund.execute", arguments=c.ToolArguments(amount_cents=amount)), state
        )

    def _execute(self, call, state):
        with self._lock:
            previous, self._security_rereading = self._security_rereading, True
            try:
                return super()._execute(call, state)
            finally:
                self._security_rereading = previous

    def _dispatch(self, call, state):
        # The gate dispatches its live reads internally rather than through call.
        if self._security_rereading and call.name in (
            "order.lookup",
            "shipment.status",
            "policy.retrieve",
        ):
            problem = self._problem(call, state)
            if problem is None and call.name == "order.lookup":
                _, problem = self.security_privacy.resolver.resolve(
                    state.request.order_reference, self._allowed_orders(state.request)
                )
            self._record_credential(call, state, problem or "")
            if problem:
                return c.ToolResult(ok=False, reason=problem)
        return super()._dispatch(call, state)

    def record_approval(self, approval, credential=None, at=None):
        approval = c._validated(c.Approval, approval)
        at = approval.approved_at if at is None else c._utc_time(at)
        runtime = self.security_privacy
        problem = reviewer_problem(credential, approval, at, runtime.verifier, runtime.revocations)
        proposal = self.proposals.get(approval.proposal_id)
        if proposal is None:
            return c.ToolResult(ok=False, reason=problem or "open_review_required")
        state = c.State(
            request=proposal.request, trace_id=proposal.evidence.trace_id, execution_at=at
        )
        self._record_credential(
            c.ToolCall(name="approval.intake"),
            state,
            problem or "",
            credential=credential,
            use_run=False,
        )
        if problem:
            with self._boundary(state, "security-privacy-approval-refusal"):
                return self._decision(state, problem)
        result = super().record_approval(approval)
        if result.ok:
            self._security_approval_ids.add(approval.approval_id)
        return result


class ScopeRemovedTools(Tools):
    """Offline control: all chapter 35 checks are deliberately removed."""


class ScopeSkipsOrder(ScopedTools):
    """Offline mutation control, never selected by the shipped component."""

    def _allowed_orders(self, request):
        return tuple(o.order_id for o in self._data.orders)

    def _problem(self, call, state):
        return super()._problem(call, state.model_copy(update={"order": None}))
