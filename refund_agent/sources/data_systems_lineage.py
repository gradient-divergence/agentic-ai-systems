"""Content-addressed, immutable inputs and a deterministic offline decision replay."""

import json
from datetime import datetime

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset
from refund_agent.labels import LABEL_POLICY
from refund_agent.model import Exchange
from refund_agent.selection import fingerprint


class RecordStore:
    def __init__(self):
        self._records: dict[str, str] = {}
        self._decisions: dict[str, str] = {}

    def put(self, record: dict) -> str:
        key = fingerprint(record)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key in self._records and self._records[key] != encoded:
            raise ValueError("record store never overwrites")
        self._records.setdefault(key, encoded)
        return key

    def get(self, key: str) -> dict:
        record = json.loads(self._records[key])
        if fingerprint(record) != key:
            raise ValueError("stored record fingerprint mismatch")
        return record

    def append(self, lineage: "LineageRecord") -> None:
        key = self.put(lineage.model_dump(mode="json"))
        self._decisions.setdefault(key, key)

    @property
    def decisions(self) -> tuple["LineageRecord", ...]:
        return tuple(
            LineageRecord.model_validate_json(self._records[key])
            for key in self._decisions
            if self.get(key)
        )

    def decisions_using(self, record_fingerprint: str) -> tuple["LineageRecord", ...]:
        return tuple(
            row
            for row in self.decisions
            if any(ref.fingerprint == record_fingerprint and ref.used for ref in row.inputs)
        )

    def export(self) -> dict:
        return {key: self.get(key) for key in sorted(self._records)}


class InputRecord(c.Record):
    source: str
    record_kind: str
    contract_version: str
    fingerprint: str
    observed_at: datetime | None
    captured_at: datetime
    age_seconds: float | None
    used: bool


class RejectedReply(c.Record):
    request_id: str
    phase: c.Phase
    reply_type: str
    payload: str | None


class ModelError(c.Record):
    request_id: str
    phase: c.Phase
    error_type: str


class LineageRecord(c.Record):
    request: c.Request
    trace_id: str
    decision_at: datetime
    decision: c.Decision
    inputs: tuple[InputRecord, ...]
    derivations: dict
    code_version: str
    replay_fingerprint: str
    registry_version: str
    registry: dict
    served_versions: dict[str, str]
    candidate: dict
    max_steps: c.Cents
    model_exchanges: tuple[Exchange, ...]
    rejected_replies: tuple[RejectedReply, ...]
    model_errors: tuple[ModelError, ...]
    model_proposal: c.Decision | None
    approvals: tuple[c.Approval, ...]
    proposal: c.Proposal | None
    reviewers: tuple[c.Reviewer, ...]
    label_policy_version: str
    mode: str
    # Chapter 33 fills this field with its stamp; all other facts stay as read.
    bundle: str | None = None


def make_lineage(tools, state, snapshots: dict) -> LineageRecord:
    from refund_agent.sources.data_systems_release import replay_fingerprint

    at = state.execution_at or state.request.at
    eligible = [p for p in tools._data.policies if p.effective_at <= at]
    current = max(eligible, key=lambda p: p.effective_at).version if eligible else None
    policy_versions = {current, state.policy.version if state.policy else None}
    refs = []
    for source, kind in (
        ("order", "orders"),
        ("shipment", "shipments"),
        ("policy", "policies"),
        ("policy", "retrievals"),
    ):
        for record in snapshots[source][kind]:
            observed = snapshots["observations"][source]
            if "observed_at" in record:
                observed = datetime.fromisoformat(record["observed_at"])
            used = source in tools._read_sources[state.trace_id]
            if kind == "policies":
                used = used and record["version"] in policy_versions
            refs.append(
                InputRecord(
                    source=source,
                    record_kind=kind,
                    contract_version=tools.served_versions[source],
                    fingerprint=tools.record_store.put(record),
                    observed_at=observed,
                    captured_at=at,
                    age_seconds=(at - observed).total_seconds() if observed else None,
                    used=used,
                )
            )
    refs.append(
        InputRecord(
            source="ledger",
            record_kind="ledger",
            contract_version=tools.served_versions["ledger"],
            fingerprint=tools.record_store.put(snapshots["ledger"]),
            observed_at=at,
            captured_at=at,
            age_seconds=0,
            used=True,
        )
    )
    order = state.order
    receipts = snapshots["ledger"]["receipts"]
    refunded = sum(p["amount_cents"] for p in receipts if order and p["order_id"] == order.order_id)
    delivered = [e.at for e in state.shipment_events if e.status == "delivered"]
    exchanges, rejected = [], []
    # Classify only after the agent has validated and processed each reply.
    for request_id, phase, reply in tools._model_exchanges.get(state.trace_id, ()):
        try:
            if type(reply) not in (c.ToolCall, c.Decision):
                raise TypeError("unsupported reply type")
            exchanges.append(Exchange(request_id=request_id, phase=phase, reply=reply))
        except (ValidationError, TypeError, ValueError):
            payload = None
            if type(reply) in (c.ToolCall, c.Decision):
                try:
                    payload = json.dumps(reply.model_dump(mode="json", warnings=False))
                except (TypeError, ValueError):
                    pass  # Preserve the rejection even if its payload is not JSON.
            rejected.append(RejectedReply(
                request_id=request_id, phase=phase,
                reply_type=type(reply).__name__, payload=payload,
            ))
    model_proposal = next(
        (
            e.reply
            for e in reversed(exchanges)
            if e.phase == c.Phase.ASSESS and isinstance(e.reply, c.Decision)
        ),
        None,
    )
    proposal = tools._read("proposals", tools._proposals).get(state.proposal_id or "")
    return LineageRecord(
        request=state.request,
        trace_id=state.trace_id,
        decision_at=at,
        decision=state.decision,
        inputs=tuple(refs),
        derivations={
            "balance": {
                "paid_cents": order.paid_cents if order else None,
                "refunded_cents": refunded,
                "receipt_ids": [p["receipt_id"] for p in receipts],
                "value_cents": order.paid_cents - refunded if order else None,
            },
            "delivery_age_seconds": (at - max(delivered)).total_seconds() if delivered else None,
        },
        code_version=tools.code_version,
        replay_fingerprint=replay_fingerprint(),
        registry_version=tools.registry.version,
        registry=tools.registry.record(),
        served_versions=tools.served_versions,
        candidate=tools.candidate_record,
        max_steps=tools.max_steps,
        model_exchanges=tuple(exchanges),
        rejected_replies=tuple(rejected),
        model_errors=tuple(
            ModelError(request_id=rid, phase=phase, error_type=error)
            for rid, phase, error in tools._model_errors.get(state.trace_id, ())
        ),
        model_proposal=model_proposal,
        approvals=tuple(state.approvals),
        proposal=proposal,
        reviewers=tuple(tools._data.reviewers),
        label_policy_version=LABEL_POLICY.version,
        mode=tools.lineage_mode,
    )


def rebuild(lineage: LineageRecord, store: RecordStore) -> dict:
    """Replay into a separate ledger. No fresh model call, payment or store mutation."""
    from refund_agent.release.candidate import Candidate, build_tools, run_request
    from refund_agent.sources.data_systems_registry import ContractVersion, Registry
    from refund_agent.sources.data_systems_release import replay_fingerprint

    if lineage.replay_fingerprint != replay_fingerprint():
        raise ValueError("recorded replay code is not available in this checkout")

    at = lineage.decision_at
    records = {kind: [] for kind in ("orders", "shipments", "policies", "retrievals")}
    ledger = None
    for ref in lineage.inputs:
        record = store.get(ref.fingerprint)
        # Capture is simulator metadata; observed_at is an untrusted source claim.
        if ref.captured_at > at:
            raise ValueError("lineage input captured after decision")
        rejected_future = (
            ref.source in ("order", "shipment", "policy")
            and lineage.decision.outcome == c.Outcome.REFER
            and not lineage.decision.executed
            and lineage.decision.reason == f"future_{ref.source}"
        )
        if ref.observed_at and ref.observed_at > at and not rejected_future:
            raise ValueError("lineage input observed after decision")
        if "observed_at" in record:
            observed = datetime.fromisoformat(record["observed_at"])
            if observed != ref.observed_at:
                raise ValueError("lineage observation does not match the input")
        if ref.observed_at and (at - ref.observed_at).total_seconds() != ref.age_seconds:
            raise ValueError("lineage age does not match the decision time")
        if ref.record_kind == "ledger":
            ledger = record
        else:
            records[ref.record_kind].append(record)
    if ledger is None:
        raise ValueError("lineage requires the ledger read")
    data = Dataset.model_validate_json(
        json.dumps(
            {
                **records,
                "requests": [lineage.request.model_dump(mode="json")],
                "correct": [],
                "reviewers": [r.model_dump(mode="json") for r in lineage.reviewers],
            }
        )
    )
    registry = Registry(
        tuple(
            ContractVersion.model_validate_json(json.dumps(e)) for e in lineage.registry["entries"]
        ),
        lineage.registry["pins"],
        lineage.registry_version,
    )
    candidate = Candidate(**lineage.candidate)
    if candidate.fingerprint != lineage.code_version:
        raise ValueError("recorded candidate fingerprint mismatch")
    tools = build_tools(data, candidate,
                        "shadow" if lineage.mode == "shadow" else "execute")
    tools.registry, tools.served_versions = registry, dict(lineage.served_versions)
    for row in ledger["receipts"]:
        receipt = c.Receipt.model_validate_json(json.dumps(row))
        if receipt.idempotency_key not in tools._read("ledger", tools._ledger):
            tools._write("ledger", receipt.idempotency_key, receipt)
    if lineage.approvals:
        if lineage.proposal is None:
            raise ValueError("approval replay requires the recorded proposal")
        tools._write("proposals", lineage.proposal.proposal_id, lineage.proposal)
        for approval in lineage.approvals:
            tools._write("approvals", approval.approval_id, approval)
    replies = {(e.request_id, e.phase): e.reply for e in lineage.model_exchanges}
    rejected = {(e.request_id, e.phase): e for e in lineage.rejected_replies}
    errors = {(e.request_id, e.phase): e for e in lineage.model_errors}

    class RecordedModel:
        def reply(self, state):
            key = state.request.request_id, state.phase
            if key in errors:
                raise type(errors[key].error_type, (Exception,), {})()
            if key in rejected:
                row = rejected[key]
                model = {"ToolCall": c.ToolCall, "Decision": c.Decision}.get(row.reply_type)
                if model and row.payload:
                    return model.model_construct(**json.loads(row.payload))
                return None
            if key not in replies:
                raise ValueError("unrecorded model output")
            return replies[key]

    model = RecordedModel()
    replay = "recorded model outputs; deterministic tools and rules"
    trace = run_request(candidate, tools, lineage.request, model=model,
                        at=at, max_steps=lineage.max_steps)
    return {
        "decision": trace.decision.model_dump(mode="json"),
        "equal": trace.decision == lineage.decision,
        "replay": replay,
        "fresh_model_limit": "a fresh model call may answer differently",
    }


def decisions_using(record_fingerprint: str, store: RecordStore):
    return store.decisions_using(record_fingerprint)
