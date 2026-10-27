"""Store faults through public calls, including work after an earlier payment.

An after-payment case first closes a paid request, then faults the next request.
This reaches mutations that completed requests correctly cannot perform. The
confirmation rows also fail after the tested request's own payment. Intake uses
record_approval: approval recording is deliberately not an eighth tool.
"""

from collections.abc import Callable
from datetime import timedelta
from itertools import product

import pytest
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for
from test_tenth_pass import corrupt_error, error_facts

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.model import Reply, ScriptedModel
from refund_agent.tools import AuditWriteError, ReconciliationError, Tools

# Store | every mutation used by the package | every read used for verification.
STORE_OPERATIONS = {
    "ledger": ("payment", "prior", "get", "items", "values"),
    "proposals": ("write", "get", "items", "values"),
    "reviews": ("open", "close", "reopen", "get", "items", "values"),
    "approvals": ("write", "delete", "get", "items", "values"),
    "audits": ("intent", "confirmation", "decision", "approval", "get", "items", "values"),
}
FAILURES = ("raise_before", "raise_after", "drop", "alter")
READS = ("get", "items", "values")
POSITIONS = ("before_payment", "after_payment")
ENTRIES = ("tools", "agent")


def pairs(store: str, operation: str) -> tuple[str, ...]:
    extra = tuple(f"read:{read}:{kind}" for read, kind in product(READS, FAILURES))
    audit = tuple(f"audit:{kind}" for kind in FAILURES)
    return (
        ("alone",)
        + (extra if operation not in READS else ())
        + (audit if (store, operation) == ("ledger", "payment") else ())
    )


ATTACK_CASES = [
    (store, operation, kind, position, entry, pair)
    for store, operations in STORE_OPERATIONS.items()
    for operation, kind, position, entry in product(operations, FAILURES, POSITIONS, ENTRIES)
    for pair in pairs(store, operation)
]
# Masked writes, malformed keys and records, and failing comparisons at every mutation.
ATTACK_CASES += [
    (store, operation, kind, position, entry, "alone")
    for store, operations in STORE_OPERATIONS.items()
    for operation, kind, position, entry in product(
        operations,
        (
            "mask",
            "malformed",
            "wrong_key",
            "bad_record",
            "loose_money",
            "bad_identity",
            "comparison",
        ),
        POSITIONS,
        ENTRIES,
    )
    if operation not in READS
]
ATTACK_CASES += [
    ("audits", "confirmation", kind, position, entry, "alone")
    for kind, position, entry in product(("raise_once", "drop_once"), POSITIONS, ENTRIES)
]
assert len(ATTACK_CASES) == 3144


def altered(record: c.Record) -> c.Record:
    if isinstance(record, c.Receipt):
        request = (
            record.request.model_copy(update={"request_id": "ghost"}) if record.request else None
        )
        return record.model_copy(
            update={"request": request, "amount_cents": record.amount_cents + 1}
        )
    if isinstance(record, c.Proposal):
        return record.model_copy(
            update={
                "request_id": "ghost",
                "request": record.request.model_copy(update={"request_id": "ghost"}),
            }
        )
    if isinstance(record, c.ReviewItem):
        return record.model_copy(update={"proposal": altered(record.proposal)})
    if isinstance(record, c.Approval):
        return record.model_copy(update={"request_id": "ghost"})
    assert isinstance(record, c.AuditRecord)
    return record.model_copy(
        update={
            "decision": record.decision.model_copy(update={"request_id": "ghost"}),
            "paid_amount_cents": record.paid_amount_cents + 1,
        }
    )


def malformed(record: c.Record) -> c.Record:
    if isinstance(record, c.Receipt):
        request: dict[str, object] = {"request_id": "ghost"}
        request["cycle"] = request
        return record.model_copy(update={"request": request})
    if isinstance(record, c.Proposal):
        return record.model_copy(update={"request_id": 0, "request": {"request_id": "ghost"}})
    if isinstance(record, c.ReviewItem):
        return record.model_copy(update={"proposal": malformed(record.proposal)})
    if isinstance(record, c.Approval):
        return record.model_copy(update={"request_id": 0})
    assert isinstance(record, c.AuditRecord)
    return record.model_copy(
        update={"decision": record.decision.model_copy(update={"request_id": 0})}
    )


def incomparable(record: c.Record) -> c.Record:
    def compare(self, other):
        raise TypeError("record comparison unavailable")

    model = type("Incomparable", (type(record),), {"__eq__": compare})
    return model(**record.model_dump())


def loose_money(record: c.Record) -> c.Record:
    if isinstance(record, c.ReviewItem):
        return record.model_copy(update={"proposal": loose_money(record.proposal)})
    field = "paid_amount_cents" if isinstance(record, c.AuditRecord) else "amount_cents"
    return record.model_copy(update={field: float(getattr(record, field))})


class BadIdentity(str):
    def __hash__(self):
        raise TypeError("malformed identity hash")


def bad_identity(record: c.Record) -> c.Record:
    identity = BadIdentity("ghost")
    if isinstance(record, c.Receipt):
        return (
            record.model_copy(
                update={"request": record.request.model_copy(update={"request_id": identity})}
            )
            if record.request
            else record.model_copy(update={"request": {"request_id": identity}})
        )
    if isinstance(record, c.Proposal):
        return record.model_copy(
            update={
                "request_id": identity,
                "request": record.request.model_copy(update={"request_id": identity}),
            }
        )
    if isinstance(record, c.ReviewItem):
        return record.model_copy(update={"proposal": bad_identity(record.proposal)})
    if isinstance(record, c.Approval):
        return record.model_copy(update={"request_id": identity})
    assert isinstance(record, c.AuditRecord)
    return record.model_copy(
        update={"decision": record.decision.model_copy(update={"request_id": identity})}
    )


def same_records(left, right) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


class FaultStore(dict[str, c.Record]):
    def __init__(
        self, records, operation: str, kind: str, active: Callable[[], bool] = lambda: True
    ):
        super().__init__(records)
        self.operation, self.kind, self.active = operation, kind, active
        self.enabled, self.attempted, self.retain_then_raise = True, False, False
        self.hits: list[str] = []
        self.read_pair: tuple[str, str] | None = None
        self.mask_records = dict(records)

    def targets(self, operation: str, value: c.Record | None) -> bool:
        if not self.enabled or not self.active():
            return False
        if self.kind in ("raise_once", "drop_once") and self.attempted:
            return False
        if operation == "delete":
            return self.operation == "delete"
        if operation not in ("write", "setdefault"):
            return False
        if self.operation == "write":
            return True
        if isinstance(value, c.Receipt):
            return self.operation == ("payment" if value.request else "prior")
        if isinstance(value, c.ReviewItem):
            return self.operation in ("open", "reopen") if value.open else self.operation == "close"
        if isinstance(value, c.AuditRecord):
            return value.event == self.operation or (
                self.operation == "outcome" and value.event != "intent"
            )
        return False

    def mutate(self, operation: str, key: str, value: c.Record | None):
        if operation == "write" and self.retain_then_raise:
            dict.__setitem__(self, key, value)
            raise OSError("injected retained approval")
        hit = self.targets(operation, value)
        if hit:
            was_attempted = self.attempted
            self.attempted = True
            self.hits.append(operation)
            if self.kind in ("raise_before", "raise_once"):
                raise OSError("injected mutation failure")
            if self.kind in ("drop", "drop_once"):
                return value
            if self.kind in (
                "wrong_key",
                "bad_record",
                "loose_money",
                "bad_identity",
                "comparison",
            ):
                original = dict.get(self, key) if operation == "delete" else value
                assert isinstance(original, c.Record)
                if operation == "delete":
                    dict.pop(self, key, None)
                found = (
                    {"request_id": "ghost"}
                    if self.kind == "bad_record"
                    else incomparable(original)
                    if self.kind == "comparison"
                    else original
                )
                if self.kind == "loose_money":
                    found = loose_money(original)
                if self.kind == "bad_identity":
                    found = bad_identity(original)
                dict.__setitem__(self, 17 if self.kind == "wrong_key" else key, found)
                return value
            if self.kind in ("alter", "mask", "malformed"):
                if self.kind == "alter" or not was_attempted:
                    self.mask_records = dict(dict.items(self))
                if operation == "delete":
                    self.mask_records.pop(key, None)
                else:
                    self.mask_records[key] = value
                original = dict.get(self, key) if operation == "delete" else value
                assert isinstance(original, c.Record)
                dict.__setitem__(
                    self,
                    key,
                    malformed(original) if self.kind == "malformed" else altered(original),
                )
                return value
        if operation == "delete":
            result = dict.pop(self, key, None)
        elif operation == "setdefault":
            assert value is not None
            result = super().setdefault(key, value)
        else:
            dict.__setitem__(self, key, value)
            result = value
        if hit and self.kind == "raise_after":
            raise OSError("injected mutation acknowledgement failure")
        return result

    def __setitem__(self, key, value):
        self.mutate("write", key, value)

    def setdefault(self, key, default=None):
        return self.mutate("setdefault", key, default)

    def pop(self, key, default=None):
        return self.mutate("delete", key, None)

    def read(self, operation: str, value):
        kind = self.kind if self.operation == operation else None
        if self.read_pair and self.attempted and self.read_pair[0] == operation:
            kind = self.read_pair[1]
        if not self.enabled or not self.active() or kind is None:
            return value
        if not self and kind in ("drop", "alter"):
            return value
        self.hits.append(operation)
        if kind.startswith("raise"):
            raise OSError("injected read failure")
        if kind == "drop":
            return None if operation == "get" else []
        if operation == "get":
            return altered(value) if isinstance(value, c.Record) else value
        if operation == "items":
            return [(key, altered(row)) for key, row in value]
        return [altered(row) for row in value]

    def get(self, key, default=None):
        records = self.mask_records if self.kind == "mask" and self.attempted else self
        return self.read("get", dict.get(records, key, default))

    def items(self):
        records = self.mask_records if self.kind == "mask" and self.attempted else self
        return self.read("items", list(dict.items(records)))

    def values(self):
        records = self.mask_records if self.kind == "mask" and self.attempted else self
        return self.read("values", list(dict.values(records)))


class IntakeFailure(dict):
    def __init__(self, retained: bool):
        super().__init__()
        self.retained = retained

    def __setitem__(self, key, value):
        if self.retained:
            dict.__setitem__(self, key, value)
        raise OSError("injected approval intake failure")


def prepare(position: str):
    data = small_case(5000, 100000)
    data.reviewers.append(c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"))
    data.orders[0].prior_refunds.append(
        c.PriorRefund(payment_id="old", amount_cents=100, at=data.requests[0].at)
    )
    paid = data.requests[0].model_copy(update={"request_id": "paid", "amount_cents": 2499})
    data.requests.append(paid)
    data.retrievals.append(data.retrievals[0].model_copy(update={"request_id": "paid"}))
    tools = Tools(data)
    if position == "after_payment":
        assert Agent(tools).run(paid).decision.executed
        closed = tools.completed_state(paid)
        assert closed is not None
        assert not tools.call(c.ToolCall(name="approval.request"), closed).ok
        assert tools.record_approval(approval_for(closed)).reason == "request_already_executed"
    state = Agent(tools).run(data.requests[0]).steps[-1].state
    return tools, state, approval_for(state)


def action_for(tools: Tools, state: c.State, approval: c.Approval, store: str, operation: str):
    state.execution_at = state.request.at + timedelta(seconds=2)
    if store == "proposals" and operation == "write":
        tools._write("proposals", state.proposal_id or "", None)
        return lambda: tools.call(
            c.ToolCall(
                name="refund.propose",
                arguments=execute(state).arguments.model_copy(
                    update={"reason": state.request.reason}
                ),
            ),
            state,
        )
    if store == "reviews" and operation == "open":
        tools._write("reviews", f"review-{state.proposal_id}", None)
        return lambda: tools.call(c.ToolCall(name="approval.request"), state)
    if store in ("reviews", "approvals") and operation not in READS:
        return lambda: tools.record_approval(approval)
    if store == "audits" and operation == "approval":
        return lambda: tools.record_approval(approval)
    if store == "audits" and operation == "decision":
        state.failure_detail = "matrix_decision"
        return lambda: tools.call(c.ToolCall(name="audit.append"), state)
    if store == "ledger" and operation == "prior":
        data = tools.data
        data.orders[0].prior_refunds.append(
            c.PriorRefund(payment_id="new-history", amount_cents=100, at=state.request.at)
        )
        tools.data = data
        return lambda: tools.call(c.ToolCall(name="order.lookup"), state)
    if store == "reviews" and operation in READS:
        return lambda: tools.call(c.ToolCall(name="approval.request"), state)
    assert tools.record_approval(approval).ok
    return lambda: tools.call(execute(state, approval_id=approval.approval_id), state)


def run_entry(tools: Tools, state: c.State, action, entry: str):
    if entry == "tools":
        return action()

    class ActingModel(ScriptedModel):
        acted = False

        def reply(self, state: c.State) -> Reply:
            if not self.acted:
                self.acted = True
                action()
            return super().reply(state)

    return Agent(tools, ActingModel()).run(
        state.request, at=state.request.at + timedelta(seconds=2)
    )


def physical(tools: Tools, name: str) -> dict[str, c.Record]:
    return dict(dict.items(getattr(tools, f"_{name}")))


def assert_report_matches_ledger(tools: Tools, result, error):
    ledger = physical(tools, "ledger")
    receipt = result.receipt if isinstance(result, c.Trace) else getattr(error, "receipt", None)
    if receipt:
        assert receipt == ledger.get(receipt.idempotency_key)
    if isinstance(result, c.Trace) and result.decision.executed:
        assert receipt is not None and result.decision == receipt.decision
    if (
        isinstance(result, c.ToolResult)
        and result.ok
        and (result.record_id or "").startswith("refund-")
    ):
        assert any(
            isinstance(r, c.Receipt) and r.receipt_id == result.record_id for r in ledger.values()
        )


def assert_at_most_one_payment(tools: Tools):
    ledger = physical(tools, "ledger")
    identities = [r.receipt_id for r in ledger.values() if isinstance(r, c.Receipt)]
    assert len(identities) == len(set(identities))
    requests = [
        str(r.request.request_id)
        for r in ledger.values()
        if isinstance(r, c.Receipt) and isinstance(r.request, c.Request)
    ]
    assert len(requests) == len(set(requests))
    assert (
        len([r for r in ledger.values() if isinstance(r, c.Receipt) and r.receipt_id == "refund-r"])
        <= 1
    )


def assert_audits_match(tools: Tools, initial: set[str], error):
    ledger = physical(tools, "ledger")
    for key, record in physical(tools, "audits").items():
        if not same_records(record, tools._expected["audits"].get(key)):
            assert isinstance(error, ReconciliationError) and error.observed is not None
            if isinstance(record, c.AuditRecord) and isinstance(record.decision.request_id, str):
                assert str(record.decision.request_id) in tools._blocked
            continue  # Altered records are quarantined, never accepted as an outcome.
        assert isinstance(record, c.AuditRecord)
        if key in initial or record.event in ("intent", "approval"):
            continue  # These records precede the tested payment.
        receipt = next(
            (
                r
                for r in ledger.values()
                if isinstance(r, c.Receipt)
                and isinstance(r.request, c.Request)
                and r.request.request_id == record.decision.request_id
            ),
            None,
        )
        assert record.money_moved == (receipt is not None)
        assert record.paid_amount_cents == (receipt.amount_cents if receipt else 0)
        assert record.receipt_id == (receipt.receipt_id if receipt else None)
        assert record.decision.executed == (receipt is not None)


def assert_blocked(tools: Tools, state: c.State, approval: c.Approval, error):
    if not isinstance(error, ReconciliationError):
        return
    assert error.expected is not None or error.store == "approvals"
    assert error.observed is not None and state.request.request_id in tools._blocked
    facts = corrupt_error(error)
    for request_id in tools._blocked:
        request = state.request.model_copy(update={"request_id": request_id})
        for name in ("refund.execute", "refund.propose", "approval.request"):
            fresh = c.State(request=request, trace_id=f"trace-{request_id}")
            with pytest.raises(ReconciliationError) as caught:
                tools.call(c.ToolCall(name=name, arguments=execute(state).arguments), fresh)
            assert error_facts(caught.value) == facts
            corrupt_error(caught.value)
        with pytest.raises(ReconciliationError) as caught:
            Agent(tools).run(request)
        assert error_facts(caught.value) == facts
        corrupt_error(caught.value)
    with pytest.raises(ReconciliationError) as caught:
        tools.record_approval(approval)
    assert error_facts(caught.value) == facts


@pytest.mark.parametrize("store,operation,kind,position,entry,pair", ATTACK_CASES)
def test_store_attack_table(store, operation, kind, position, entry, pair, monkeypatch):
    tools, state, approval = prepare(position)
    action = action_for(tools, state, approval, store, operation)
    initial = set(physical(tools, "audits"))
    fault = FaultStore(physical(tools, store), operation, kind)
    monkeypatch.setattr(tools, f"_{store}", fault)
    if (store, operation) == ("reviews", "reopen"):
        monkeypatch.setattr(tools, "_approvals", IntakeFailure(False))
    if (store, operation) == ("approvals", "delete"):
        fault.retain_then_raise = True
    secondary = None
    if pair.startswith("read:"):
        _, read, read_kind = pair.split(":")
        fault.read_pair = (read, read_kind)
    if pair.startswith("audit:"):
        secondary = FaultStore(
            physical(tools, "audits"), "outcome", pair.split(":")[1], lambda: fault.attempted
        )
        monkeypatch.setattr(tools, "_audits", secondary)
    result, error = None, None
    try:
        result = run_entry(tools, state, action, entry)
    except Exception as caught:
        error = caught
    assert fault.hits, (store, operation, kind, position, entry, pair)
    for name in STORE_OPERATIONS:
        if not same_records(physical(tools, name), tools._expected[name]):
            assert isinstance(error, ReconciliationError), (name, error)
    if kind in ("bad_record", "bad_identity") or (
        kind == "malformed" and store in ("ledger", "proposals", "reviews")
    ):
        assert "ghost" in tools._blocked
    assert_report_matches_ledger(tools, result, error)
    if (store, operation) == ("audits", "confirmation"):
        assert isinstance(error, (AuditWriteError, ReconciliationError))
        assert error.receipt is not None and error.receipt.receipt_id in str(error)
    assert_at_most_one_payment(tools)
    assert_audits_match(tools, initial, error)
    fault.enabled = False
    if secondary:
        secondary.enabled = False
    assert_blocked(tools, state, approval, error)
    before = physical(tools, "ledger")
    initial = set(physical(tools, "audits"))
    if not isinstance(error, ReconciliationError):
        for key in ("retry-one", "retry-two"):
            try:
                trace = Agent(tools).run(state.request)
                assert_report_matches_ledger(tools, trace, None)
                tools.call(
                    execute(trace.steps[-1].state, key, approval.approval_id), trace.steps[-1].state
                )
            except Exception as retry:
                assert_report_matches_ledger(tools, None, retry)
    else:
        assert physical(tools, "ledger") == before
    assert_at_most_one_payment(tools)
    assert_audits_match(tools, initial, error)
