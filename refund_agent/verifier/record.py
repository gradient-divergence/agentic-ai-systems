"""Proposal-bound verdicts, detached read tools and the verifier's one write."""

import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from refund_agent import contracts as c
from refund_agent.contracts import _detach, _utc_time, _validated
from refund_agent.selection import fingerprint

READ_NAMES = ("order.lookup", "shipment.status", "policy.retrieve")


def evidence_fingerprint(evidence):
    return fingerprint(
        evidence.model_dump(mode="json", include={"order", "shipment_events", "policy"})
    )


class Read(c.Record):
    tool: str
    observed_at: datetime
    read_at: datetime
    sha256: str
    words: int = Field(ge=0, strict=True)


class Verdict(c.Record):
    proposal_id: str
    request_id: str
    request_sha256: str
    amount_cents: c.Cents
    reason: str
    policy_version: str
    evidence_sha256: str
    reads: tuple[Read, ...]
    verifier: str
    answer: Literal["agree", "object"]
    objection_reason: str | None = None
    completed_at: datetime
    words_read: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def objection_has_reason(self):
        if (self.answer == "object") != bool(self.objection_reason):
            raise ValueError("only an objection carries an objection reason")
        if not self.proposal_id or not self.request_id or not self.verifier:
            raise ValueError("proposal, request and verifier identities required")
        return self


def make_verdict(proposal, evidence, reads, name, answer, reason, completed_at, words):
    return Verdict(
        proposal_id=proposal.proposal_id,
        request_id=proposal.request_id,
        request_sha256=fingerprint(proposal.request.model_dump(mode="json")),
        amount_cents=proposal.amount_cents,
        reason=proposal.reason,
        policy_version=evidence.policy.version,
        evidence_sha256=evidence_fingerprint(evidence),
        reads=tuple(reads),
        verifier=name,
        answer=answer,
        objection_reason=reason,
        completed_at=completed_at,
        words_read=words,
    )


def verdict_problem(verdict, proposal, state, deadline):
    expected = (
        proposal.proposal_id,
        state.request.request_id,
        fingerprint(state.request.model_dump(mode="json")),
        state.request.amount_cents,
        state.request.reason,
        state.policy.version,
        evidence_fingerprint(state),
    )
    received = (
        verdict.proposal_id,
        verdict.request_id,
        verdict.request_sha256,
        verdict.amount_cents,
        verdict.reason,
        verdict.policy_version,
        verdict.evidence_sha256,
    )
    if received != expected or verdict.completed_at < proposal.created_at:
        return "verifier_verdict_void"
    if verdict.completed_at > deadline:
        return "verifier_timeout"
    return None


class VerdictLog:
    def __init__(self):
        self.__entries = ()

    @property
    def entries(self):
        return self.__entries

    def append(self, verdict):
        verdict = _validated(Verdict, verdict)
        if any(row.proposal_id == verdict.proposal_id for row in self.__entries):
            raise ValueError("one verdict per proposal")
        self.__entries += (verdict,)


class ReadOnlyTools:
    """Read the same sources without holding a paying tools instance or its state."""

    names = READ_NAMES
    __slots__ = ("__data", "at", "__reads", "evidence_state")

    def __init__(self, data, at):
        self.__data = data
        self.at = _utc_time(at)
        self.__reads = ()
        self.evidence_state = None

    @property
    def reads(self):
        return self.__reads

    @property
    def words(self):
        return sum(read.words for read in self.reads)

    def _record(self, name, row):
        payload = json.dumps(row.model_dump(mode="json"), sort_keys=True)
        observed = row.observed_at if hasattr(row, "observed_at") else self.at
        self.__reads += (
            Read(
                tool=name,
                observed_at=observed,
                read_at=self.at,
                sha256=fingerprint(row.model_dump(mode="json")),
                words=len(payload.split()),
            ),
        )

    def call(self, call, state):
        if call.name not in self.names:
            raise ValueError("read-only tools refuse every write")
        return {
            "order.lookup": self._order,
            "shipment.status": self._shipment,
            "policy.retrieve": self._policy,
        }[call.name](state)

    def _order(self, state):
        rows = [
            o
            for o in self.__data.orders
            if state.request.order_reference in (o.order_id, o.reference)
        ]
        if len(rows) != 1:
            return c.ToolResult(ok=False, reason="ambiguous_order" if rows else "order_not_found")
        state.order = _detach(rows[0])
        self._record("order.lookup", state.order)
        return c.ToolResult(ok=True, order=_detach(state.order))

    def _shipment(self, state):
        row = next(
            (
                s
                for s in self.__data.shipments
                if state.order and s.order_id == state.order.order_id
            ),
            None,
        )
        if row is None:
            return c.ToolResult(ok=False, reason="shipment_not_found")
        row = _detach(row)
        state.shipment_events = _detach(row.events)
        self._record("shipment.status", row)
        return c.ToolResult(ok=True, shipment=row)

    def _policy(self, state):
        snapshot = next(
            (s for s in self.__data.retrievals if s.request_id == state.request.request_id), None
        )
        policy = next(
            (p for p in self.__data.policies if snapshot and p.version == snapshot.version), None
        )
        if policy is None or snapshot is None:
            return c.ToolResult(ok=False, reason="policy_not_found")
        state.policy = _detach(policy)
        self._record("policy.retrieve", state.policy)
        self.__reads = self.__reads[:-1] + (
            self.__reads[-1].model_copy(update={"observed_at": snapshot.observed_at}),
        )
        return c.ToolResult(ok=True, policy=_detach(state.policy))

    def evidence(self, proposal):
        state = c.State(
            request=_detach(proposal.request), trace_id=f"verify-{proposal.proposal_id}"
        )
        for name in self.names:
            result = self.call(c.ToolCall(name=name), state)
            if not result.ok:
                raise ValueError(result.reason)
        self.evidence_state = state
        return state
