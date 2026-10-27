from datetime import timedelta
from types import MappingProxyType

import pytest
from pydantic import ValidationError
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence

from refund_agent import contracts as c
from refund_agent.agent import READ_NEXT, Agent
from refund_agent.evaluation import POSITIVE_CASES
from refund_agent.tools import AuditWriteError, PaymentError, ReconciliationError, Tools


def reviewed_case(amount: int = 5000):
    data = small_case(amount, 100000)
    data.reviewers.extend(
        [
            c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
            c.Reviewer(reviewer_id="morgan", name="Morgan", role="named_approver"),
        ]
    )
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state, "named_approver" if amount > 50000 else "reviewer")
    assert tools.record_approval(approval).ok
    return tools, state, approval


@pytest.mark.parametrize("kind", ["payment", "audit"])
@pytest.mark.parametrize("pay", [False, True])
def test_model_exception_cannot_supply_payment_facts(kind: str, pay: bool) -> None:
    other = Tools(small_case())
    foreign = Agent(other).run(other.data.requests[0]).receipt
    assert foreign is not None
    error = (
        PaymentError(foreign, OSError("model"))
        if kind == "payment"
        else AuditWriteError(list(other.audits.values())[-1], foreign)
    )
    tools, state, approval = reviewed_case()

    prepared = state

    class Model:
        def reply(self, state: c.State) -> c.ToolCall:
            if pay:
                assert tools.call(execute(prepared, approval_id=approval.approval_id), prepared).ok
            raise error

    trace = Agent(tools, Model()).run(state.request)
    assert trace.decision.executed == pay
    assert trace.receipt == tools.payment_for(state.request.request_id)
    if pay:
        assert trace.receipt is not None and trace.receipt.amount_cents == 5000
    else:
        assert trace.receipt is None
    failure = list(tools.audits.values())[-1]
    assert failure.money_moved == pay and failure.paid_amount_cents == (5000 if pay else 0)
    assert "model_error" in (trace.failure_detail or "")


def test_completed_request_cannot_reopen_or_accept_approval(monkeypatch: pytest.MonkeyPatch):
    tools, state, approval = reviewed_case()
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok
    assert not tools.call(c.ToolCall(name="approval.request"), state).ok
    assert not any(r.open for r in tools.reviews.values())

    class FailedWrite(dict):
        def __setitem__(self, key, value):
            raise OSError("approval write failed")

    monkeypatch.setattr(tools, "_approvals", FailedWrite(tools._approvals))
    replacement = approval.model_copy(
        update={
            "approval_id": "replacement",
            "approved_at": approval.approved_at + timedelta(seconds=2),
        }
    )
    assert tools.record_approval(replacement).reason == "request_already_executed"
    audit = list(tools.audits.values())[-1]
    assert audit.decision.executed and audit.money_moved and audit.paid_amount_cents == 5000
    assert len(tools.payments) == 1 and "replacement" not in tools._approvals


def test_malformed_direct_arguments_are_audited():
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    call = c.ToolCall.model_construct(
        name="refund.execute",
        arguments=c.ToolArguments.model_construct(amount_cents=-1, idempotency_key="bad"),
    )
    assert tools.call(call, state).reason == "invalid_tool_arguments"
    assert not tools.payments and len(tools.audits) == 1
    assert list(tools.audits.values())[-1].failure_detail == "invalid_tool_arguments"


def test_malformed_direct_state_retains_request_context():
    tools = Tools(small_case())
    request = tools.data.requests[0]
    state = c.State.model_construct(request=request, phase="invalid", trace_id="trace-r")
    result = tools.call(c.ToolCall(name="order.lookup"), state)
    assert not result.ok and result.reason == "invalid_tool_arguments"
    assert len(tools.audits) == 1 and not tools.payments


@pytest.mark.parametrize("store", ["approvals", "proposals", "reviews"])
def test_approval_intake_read_failure_is_audited(store: str, monkeypatch: pytest.MonkeyPatch):
    tools, state, approval = reviewed_case()
    approval = approval.model_copy(update={"approval_id": "new"})

    class FailedRead(dict):
        def get(self, key, default=None):
            raise OSError(f"{store} read failed")

    monkeypatch.setattr(tools, f"_{store}", FailedRead(getattr(tools, f"_{store}")))
    before = len(tools.audits)
    with pytest.raises(ReconciliationError, match="reconciliation_required"):
        tools.record_approval(approval)
    assert len(tools.audits) == before + 1 and not tools.payments
    assert "approval_intake" in (list(tools.audits.values())[-1].failure_detail or "")


@pytest.mark.parametrize("table", [READ_NEXT, c.PHASE_TO_TOOL, POSITIVE_CASES])
def test_control_tables_are_immutable(table):
    assert isinstance(table, MappingProxyType)
    with pytest.raises(TypeError):
        table[c.Phase.APPROVAL] = c.Phase.EXECUTE


def test_agent_tier_three_guard_is_independent_of_transitions(monkeypatch: pytest.MonkeyPatch):
    import refund_agent.agent as module

    tools, state, _ = reviewed_case(50001)
    monkeypatch.setattr(
        module, "READ_NEXT", {**module.READ_NEXT, c.Phase.APPROVAL: c.Phase.EXECUTE}
    )
    trace = Agent(tools).run(state.request)
    assert not trace.decision.executed and not tools.payments


def test_unreadable_ledger_never_reports_a_supplied_receipt(monkeypatch: pytest.MonkeyPatch):
    tools, state, approval = reviewed_case()
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok

    class FailedRead(dict):
        def values(self):
            raise OSError("ledger unavailable")

    before = len(tools.audits)
    monkeypatch.setattr(tools, "_ledger", FailedRead(tools.ledger))
    with pytest.raises(PaymentError, match="payment_outcome_unknown") as caught:
        tools.call(c.ToolCall(name="approval.request"), state)
    assert caught.value.receipt is None and not caught.value.confirmed
    assert state.decision is None
    assert len(tools.audits) == before


def test_public_schema_metadata_cannot_relax_agent_input_validation():
    tools = Tools(small_case())
    field = c.Request.model_fields["requester_id"]
    original = field.metadata
    try:
        field.metadata = []
        c.Request.model_rebuild(force=True)
        request = tools.data.requests[0].model_copy(update={"requester_id": ""})
        with pytest.raises(ValidationError):
            Agent(tools).run(request)
        assert not tools.payments
    finally:
        field.metadata = original
        c.Request.model_rebuild(force=True)


def test_paid_failure_audit_uses_receipt_approval_and_evidence():
    tools, state, approval = reviewed_case()
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok
    receipt = tools.payment_for(state.request.request_id)
    assert receipt is not None and receipt.evidence is not None
    state.approvals = [approval.model_copy(update={"approval_id": "forged"})]
    state.proposal_id = "forged"
    state.freshness, state.risk_score = {"order": 12345}, 0.99
    assert not tools.call(c.ToolCall(name="approval.request"), state).ok
    audit = list(tools.audits.values())[-1]
    assert audit.approval_ids == [approval.approval_id]
    assert audit.freshness == receipt.evidence.freshness
    assert audit.risk_score == receipt.evidence.risk_score
    assert audit.proposal_id == receipt.proposal_id


def test_direct_decision_store_read_failure_is_audited(monkeypatch: pytest.MonkeyPatch):
    data = small_case()
    tools, state = evidence(data, data.requests[0])

    class FailedRead(dict):
        def values(self):
            raise OSError("proposal read failed")

    monkeypatch.setattr(tools, "_proposals", FailedRead(tools.proposals))
    with pytest.raises(ReconciliationError, match="reconciliation_required"):
        tools.record_decision(state)
    assert len(tools.audits) == 1
    assert not list(tools.audits.values())[-1].money_moved


def test_completed_proposal_cannot_be_reopened_through_another_request():
    tools, state, approval = reviewed_case()
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok
    other = state.model_copy(
        update={"request": state.request.model_copy(update={"request_id": "other"})}, deep=True
    )
    assert not tools.call(c.ToolCall(name="approval.request"), other).ok
    assert not any(r.open for r in tools.reviews.values())


@pytest.mark.parametrize(
    "method,store",
    [
        ("approval_for", "proposals"),
        ("changed_approval", "approvals"),
        ("run_time", "approvals"),
        ("request_problem", "proposals"),
        ("completed_state", "proposals"),
    ],
)
def test_public_contextual_reads_are_audited(method, store, monkeypatch: pytest.MonkeyPatch):
    data = small_case()
    tools, state = evidence(data, data.requests[0])

    class FailedRead(dict):
        def get(self, key, default=None):
            raise OSError("store read failed")

        def values(self):
            raise OSError("store read failed")

        def items(self):
            raise OSError("store read failed")

    monkeypatch.setattr(tools, f"_{store}", FailedRead(getattr(tools, f"_{store}")))
    argument = state if method in ("approval_for", "changed_approval") else state.request
    with pytest.raises(ReconciliationError, match="reconciliation_required"):
        getattr(tools, method)(argument)
    assert len(tools.audits) == 1 and not tools.payments


@pytest.mark.parametrize("kind", ["payment", "audit"])
def test_store_exception_cannot_supply_foreign_payment(kind, monkeypatch: pytest.MonkeyPatch):
    other = Tools(small_case())
    foreign = Agent(other).run(other.data.requests[0]).receipt
    assert foreign is not None
    error = (
        PaymentError(foreign, OSError("store"))
        if kind == "payment"
        else AuditWriteError(list(other.audits.values())[-1], foreign)
    )
    tools, state, approval = reviewed_case()

    class FailedRead(dict):
        def get(self, key, default=None):
            raise error

    monkeypatch.setattr(tools, "_approvals", FailedRead(tools._approvals))
    before = len(tools.audits)
    with pytest.raises(RuntimeError) as caught:
        tools.record_approval(approval.model_copy(update={"approval_id": "new"}))
    assert getattr(caught.value, "receipt", None) is None
    assert "2499" not in str(caught.value)
    assert len(tools.audits) == before + 1
    assert not list(tools.audits.values())[-1].money_moved


def test_nested_model_tool_calls_cannot_reopen_a_payment():
    tools, prepared, approval = reviewed_case()
    results = []

    class Model:
        def reply(self, state: c.State) -> c.ToolCall:
            tools.call(execute(prepared, approval_id=approval.approval_id), prepared)
            stale = prepared.model_copy(update={"receipt": None}, deep=True)
            results.append(tools.call(c.ToolCall(name="approval.request"), stale))
            raise OSError("model failed")

    trace = Agent(tools, Model()).run(prepared.request)
    assert trace.decision.executed
    assert len(results) == 1 and not results[0].ok
    assert results[0].reason == "request_already_executed"
    assert not any(review.open for review in tools.reviews.values())


def test_ledger_loss_during_failure_audit_reports_unknown(monkeypatch: pytest.MonkeyPatch):
    tools, state, approval = reviewed_case()
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok

    class LateLoss(dict):
        reads = 0

        def values(self):
            self.reads += 1
            if self.reads >= 3:
                raise OSError("ledger lost during failure audit")
            return super().values()

    class FailedProposal(dict):
        def get(self, key, default=None):
            raise OSError("proposal read failed")

    monkeypatch.setattr(tools, "_ledger", LateLoss(tools.ledger))
    monkeypatch.setattr(tools, "_proposals", FailedProposal(tools.proposals))
    with pytest.raises(ReconciliationError, match="payment_outcome_unknown") as caught:
        tools.approval_for(state)
    assert caught.value.receipt is None and not caught.value.confirmed
    assert caught.value.expected is not None and "refund-r" in repr(caught.value.ledger_expected)
    assert caught.value.observed is not None
