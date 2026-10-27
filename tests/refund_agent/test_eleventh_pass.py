import pytest
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence
from test_store_matrix import physical, run_entry

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.model import ScriptedModel
from refund_agent.tools import ReconciliationError


class WrongKey(dict):
    def __setitem__(self, key, value):
        dict.__setitem__(self, 17, value)

    def setdefault(self, key, value=None):
        self[key] = value
        return value


class IntermittentRead(dict):
    def __init__(self, records, name, reads, fail_at=None):
        super().__init__(records)
        self.name, self.reads, self.fail_at = name, reads, fail_at

    def read(self, operation, value):
        self.reads.append((self.name, operation))
        if len(self.reads) == self.fail_at:
            raise OSError("intermittent read unavailable")
        return value

    def get(self, key, default=None):
        return self.read("get", dict.get(self, key, default))

    def items(self):
        return self.read("items", list(dict.items(self)))

    def values(self):
        return self.read("values", list(dict.values(self)))


def read_case(entry, fail_at=None):
    data = small_case(5000, 100000)
    data.reviewers.append(c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"))
    tools, state = evidence(data, data.requests[0])
    reads = []
    for name in ("ledger", "proposals", "reviews", "approvals", "audits"):
        setattr(tools, f"_{name}", IntermittentRead(physical(tools, name), name, reads, fail_at))

    def pay():
        approval = approval_for(state)
        assert tools.record_approval(approval).ok
        return tools.call(execute(state, approval_id=approval.approval_id), state)

    return tools, state, reads, lambda: run_entry(tools, state, pay, entry)


def read_positions():
    cases = []
    for entry in ("tools", "agent"):
        tools, _, reads, action = read_case(entry)
        action()
        payments = list(physical(tools, "ledger").values())
        assert len(payments) == 1 and isinstance(payments[0], c.Receipt)
        assert payments[0].amount_cents == 5000
        cases.extend(
            (entry, index, name, operation) for index, (name, operation) in enumerate(reads, 1)
        )
    return cases


@pytest.mark.parametrize("entry,position,store,operation", read_positions())
def test_intermittent_execution_read_table(entry, position, store, operation):
    tools, state, reads, action = read_case(entry, position)
    with pytest.raises(ReconciliationError) as caught:
        action()
    error = caught.value
    assert reads[position - 1] == (store, operation)
    assert error.expected is not None and error.observed is not None
    ledger = physical(tools, "ledger")
    assert len(ledger) <= 1
    payment = ledger.get("payment")
    assert payment is None or isinstance(payment, c.Receipt)
    if error.confirmed:
        assert error.receipt is not None and error.receipt.amount_cents == 5000
        assert error.receipt == ledger["payment"]
        assert "payment_recorded payment=refund-r cents=5000" in str(error)
    else:
        assert error.receipt is None and "payment_recorded" not in str(error)
    for audit in physical(tools, "audits").values():
        assert isinstance(audit, c.AuditRecord)
        if audit.event == "confirmation":
            assert payment is not None and payment.amount_cents == audit.paid_amount_cents == 5000
    before = dict(ledger)
    for retry in (
        lambda: tools.call(execute(state, "retry"), state),
        lambda: Agent(tools).run(state.request),
    ):
        with pytest.raises(ReconciliationError):
            retry()
    assert physical(tools, "ledger") == before


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_confirmation_ledger_loss_never_confirms_cached_payment(entry, monkeypatch):
    class LateRead(dict):
        reads = 0

        def values(self):
            if self:
                self.reads += 1
                if self.reads >= 3:
                    raise OSError("confirmation ledger unavailable")
            return super().values()

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    ledger = LateRead()
    monkeypatch.setattr(tools, "_ledger", ledger)
    with pytest.raises(ReconciliationError) as caught:
        run_entry(tools, state, lambda: tools.call(execute(state), state), entry)
    error = caught.value
    assert not error.confirmed and error.receipt is None
    assert "payment_outcome_unknown" in str(error) and "payment_recorded" not in str(error)
    payment = physical(tools, "ledger")["payment"]
    assert isinstance(payment, c.Receipt) and payment.amount_cents == 2499
    assert "refund-r" in repr(error.expected) + repr(error.observed)
    assert [r.event for r in tools.audits.values()] == ["intent"]


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_confirmation_audit_and_ledger_loss_never_confirm_cached_payment(entry, monkeypatch):
    class Ledger(dict):
        unavailable = False

        def get(self, key, default=None):
            if self.unavailable:
                raise OSError("confirmation keyed ledger unavailable")
            return super().get(key, default)

    ledger = Ledger()

    class Audit(dict):
        def setdefault(self, key, value=None):
            assert value is not None
            if value.event == "confirmation":
                ledger.unavailable = True
                value = value.model_copy(update={"trace_id": "altered"})
            return super().setdefault(key, value)

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_ledger", ledger)
    monkeypatch.setattr(tools, "_audits", Audit())
    with pytest.raises(ReconciliationError) as caught:
        run_entry(tools, state, lambda: tools.call(execute(state), state), entry)
    error = caught.value
    assert error.store == "audits" and not error.confirmed and error.receipt is None
    assert "payment_outcome_unknown" in str(error) and "payment_recorded" not in str(error)
    payment = physical(tools, "ledger")["payment"]
    assert isinstance(payment, c.Receipt) and payment.amount_cents == 2499
    assert isinstance(error.expected, c.AuditRecord) and error.expected.paid_amount_cents == 2499
    assert error.observed is not None
    with pytest.raises(ReconciliationError) as retry:
        tools.payment_for(state.request.request_id)
    assert not retry.value.confirmed and retry.value.receipt is None
    assert retry.value.ledger_expected["payment"] == payment
    assert retry.value.ledger_observed["error"] == "OSError"
    ledger.unavailable = False
    with pytest.raises(ReconciliationError) as recovered:
        tools.payment_for(state.request.request_id)
    assert recovered.value.confirmed and recovered.value.receipt == payment
    assert "payment_recorded payment=refund-r cents=2499" in str(recovered.value)


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_reconciliation_checks_contents_after_keyed_reads(entry, monkeypatch):
    class Ledger(dict):
        armed, reads = False, 0

        def get(self, key, default=None):
            value = dict.get(self, key, default)
            if self.armed:
                self.reads += 1
                if self.reads == (3 if entry == "agent" else 2):
                    dict.__setitem__(self, key, value.model_copy(update={"amount_cents": 2500}))
            return value

    ledger = Ledger()

    class Audit(dict):
        def setdefault(self, key, value=None):
            assert value is not None
            if value.event == "confirmation":
                ledger.armed = True
                value = value.model_copy(update={"trace_id": "altered"})
            return super().setdefault(key, value)

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_ledger", ledger)
    monkeypatch.setattr(tools, "_audits", Audit())
    with pytest.raises(ReconciliationError) as caught:
        if entry == "agent":
            Agent(tools).run(state.request)
        else:
            tools.call(execute(state), state)
    assert not caught.value.confirmed and caught.value.receipt is None
    assert "payment_outcome_unknown" in str(caught.value)
    payment = next(iter(physical(tools, "ledger").values()))
    assert isinstance(payment, c.Receipt) and payment.amount_cents == 2500


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize("store", ["ledger", "proposals", "reviews", "approvals", "audits"])
def test_wrong_key_quarantines_every_store(store, entry, monkeypatch):
    data = small_case(5000 if store in ("reviews", "approvals") else 2499)
    data.reviewers.append(c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"))
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state)
    fault = WrongKey(physical(tools, store))
    monkeypatch.setattr(tools, f"_{store}", fault)
    calls = {
        "ledger": execute(state),
        "audits": execute(state),
        "proposals": c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=2499, reason="damaged"),
        ),
        "reviews": c.ToolCall(name="approval.request"),
    }
    action = (
        (lambda: tools.record_approval(approval))
        if store == "approvals"
        else (lambda: tools.call(calls[store], state))
    )
    with pytest.raises(ReconciliationError) as caught:
        run_entry(tools, state, action, entry)
    assert caught.value.store == store and caught.value.observed is not None
    assert 17 in caught.value.observed["contents"]
    before = physical(tools, "ledger")
    for retry in (
        lambda: tools.call(execute(state, "retry"), state),
        lambda: Agent(tools).run(state.request),
        lambda: tools.record_approval(approval),
    ):
        with pytest.raises(ReconciliationError):
            retry()
    assert physical(tools, "ledger") == before


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_frozen_caller_state_is_rejected_before_payment(entry):
    class FrozenState(c.State):
        model_config = {"extra": "forbid", "frozen": True}

    data = small_case()
    tools, prepared = evidence(data, data.requests[0])
    frozen = FrozenState(**prepared.model_dump())
    before = frozen.model_dump()
    results = []

    def attempt():
        results.append(tools.call(execute(frozen), frozen))

    class Model(ScriptedModel):
        def reply(self, state):
            attempt()
            return c.ToolCall(name="refund.execute")

    if entry == "agent":
        assert not Agent(tools, Model()).run(prepared.request).decision.executed
    else:
        attempt()
        assert not tools.audits
    assert len(results) == 1 and not results[0].ok
    assert results[0].reason == "mutable_state_required"
    assert not tools.payments and frozen.model_dump() == before
