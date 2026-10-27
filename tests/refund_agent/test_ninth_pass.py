import pytest
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence
from test_tenth_pass import corrupt_error, error_facts

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.evaluation import _positive_data, _score_safety
from refund_agent.tools import AuditWriteError, ReconciliationError, Tools


@pytest.mark.parametrize("through_agent", [False, True])
@pytest.mark.parametrize("fault", ["request", "amount", "hidden", "scan"])
def test_review_ledger_reproductions(fault, through_agent, monkeypatch):
    class BadLedger(dict):
        def __setitem__(self, key, value):
            if fault == "request":
                value = value.model_copy(
                    update={"request": value.request.model_copy(update={"request_id": "ghost"})}
                )
            if fault in ("amount", "hidden"):
                value = value.model_copy(update={"amount_cents": 100})
            super().__setitem__(key, value)

        def get(self, key, default=None):
            return default if fault == "hidden" else super().get(key, default)

        def values(self):
            if self and fault == "scan":
                raise OSError("ledger read failed")
            return super().values()

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    ledger = BadLedger()
    monkeypatch.setattr(tools, "_ledger", ledger)
    with pytest.raises(ReconciliationError) as caught:
        Agent(tools).run(state.request) if through_agent else tools.call(execute(state), state)
    error = caught.value
    assert error.expected is not None and error.observed is not None
    assert "refund-r" in str(error)
    assert [r.event for r in tools.audits.values()] == ["intent"]
    facts = corrupt_error(error)
    for key in ("one", "two"):
        with pytest.raises(ReconciliationError) as retry:
            tools.call(execute(state, key), state)
        assert error_facts(retry.value) == facts
    with pytest.raises(ReconciliationError) as retry:
        Agent(tools).run(state.request)
    assert error_facts(retry.value) == facts and len(ledger) == 1
    if fault == "request":
        with pytest.raises(ReconciliationError) as ghost:
            Agent(tools).run(state.request.model_copy(update={"request_id": "ghost"}))
        assert error_facts(ghost.value) == facts


@pytest.mark.parametrize("through_agent", [False, True])
def test_review_dropped_ledger_write(through_agent, monkeypatch):
    class Dropped(dict):
        def __setitem__(self, key, value):
            pass

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_ledger", Dropped())
    result = Agent(tools).run(state.request) if through_agent else tools.call(execute(state), state)
    assert not (result.decision.executed if isinstance(result, c.Trace) else result.ok)
    assert not tools.payments
    assert all(r.event != "confirmation" and not r.money_moved for r in tools.audits.values())
    assert len([r for r in tools.audits.values() if r.event == "decision"]) == 1


@pytest.mark.parametrize("alter", [False, True])
@pytest.mark.parametrize("after_payment", [False, True])
def test_audit_write_verification(alter, after_payment, monkeypatch):
    class BadAudit(dict):
        def setdefault(self, key, value=None):
            assert value is not None
            if value.event == ("confirmation" if after_payment else "intent"):
                if alter:
                    super().__setitem__(key, value.model_copy(update={"trace_id": "other"}))
                return value
            return super().setdefault(key, value)

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_audits", BadAudit())
    with pytest.raises(ReconciliationError if alter else AuditWriteError) as caught:
        tools.call(execute(state), state)
    assert len(tools.payments) == int(after_payment)
    receipt = caught.value.receipt
    assert (receipt is not None) == after_payment
    if receipt:
        assert (
            receipt == tools.payment_for(state.request.request_id)
            if not alter
            else receipt.amount_cents == 2499
        )
        assert receipt.receipt_id in str(caught.value)
    if alter:
        assert isinstance(caught.value, ReconciliationError)
        facts = corrupt_error(caught.value)
        with pytest.raises(ReconciliationError) as retry:
            Agent(tools).run(state.request)
        assert error_facts(retry.value) == facts


def test_review_unverified_approval_rollback_blocks(monkeypatch):
    class FailedApproval(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise OSError("approval write failed")

        def pop(self, key, default=None):
            return self.get(key, default)

    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools, state = evidence(data, request)
    approval = approval_for(state)
    monkeypatch.setattr(tools, "_approvals", FailedApproval())
    with pytest.raises(ReconciliationError) as caught:
        tools.record_approval(approval)
    assert caught.value.expected is None and caught.value.observed is not None
    facts = corrupt_error(caught.value)
    for name in ("refund.execute", "refund.propose", "approval.request"):
        with pytest.raises(ReconciliationError) as retry:
            tools.call(c.ToolCall(name=name, arguments=execute(state).arguments), state)
        assert error_facts(retry.value) == facts
    with pytest.raises(ReconciliationError) as retry:
        tools.record_approval(approval)
    assert error_facts(retry.value) == facts and not tools.payments


def test_nested_paid_read_failure_writes_no_confirmation(monkeypatch):
    data = small_case()
    tools, prepared = evidence(data, data.requests[0])
    counts = []

    class LateRead(dict):
        def get(self, key, default=None):
            if tools._ledger:
                raise OSError("proposal read failed after payment")
            return super().get(key, default)

    monkeypatch.setattr(tools, "_proposals", LateRead(tools.proposals))

    class Model:
        def reply(self, state: c.State) -> c.ToolCall:
            assert tools.call(execute(prepared), prepared).ok
            counts.append(len(tools.audits))
            tools.approval_for(prepared)
            return c.ToolCall(name="order.lookup")

    with pytest.raises(ReconciliationError) as caught:
        Agent(tools, Model()).run(prepared.request)
    assert caught.value.receipt is not None and caught.value.receipt.amount_cents == 2499
    assert len(tools.audits) == counts[0]


def test_safety_rejects_an_extra_contradictory_audit():
    class ContradictoryAudit(Tools):
        @property
        def audits(self):
            records = super().audits
            for key, record in list(records.items()):
                records[f"bad-{key}"] = record.model_copy(update={"paid_amount_cents": 999})
            return records

    score = _score_safety(generate(), ContradictoryAudit)
    assert score.correct == score.audited == score.unauthorized_cents == 0
