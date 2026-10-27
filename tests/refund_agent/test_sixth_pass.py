from datetime import timedelta

import pytest
from pydantic import ValidationError
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence

from refund_agent.agent import Agent
from refund_agent.contracts import Phase, PriorRefund, ToolArguments, ToolCall
from refund_agent.evaluation import _positive_data
from refund_agent.model import ScriptedModel, Transcript
from refund_agent.rules import approval_problem
from refund_agent.tools import PaymentError, Tools


@pytest.mark.parametrize("entry", ["constructor", "setter"])
def test_dataset_entry_rejects_crafted_reviewer_role(entry: str) -> None:
    data = _positive_data()
    bad = data.model_copy(
        update={"reviewers": [data.reviewers[0].model_copy(update={"role": "customer"})]}
    )
    tools = Tools(data)
    with pytest.raises(ValidationError):
        if entry == "constructor":
            Tools(bad)
        else:
            tools.data = bad
    assert tools.data == data and not tools.payments


def test_approval_role_is_independent_of_dataset() -> None:
    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools, state = evidence(data, request)
    approval = approval_for(state).model_copy(update={"role": "customer"})
    bad = data.model_copy(
        update={"reviewers": [data.reviewers[0].model_copy(update={"role": "customer"})]}
    )
    proposal = tools.proposals[state.proposal_id or ""]
    assert approval_problem(bad, state, proposal, approval) == "invalid_approval"


@pytest.mark.parametrize("method", ["completed_state", "request_problem", "run_time"])
def test_request_handling_validates_crafted_models(method: str) -> None:
    data = small_case()
    tools = Tools(data)
    invalid = data.requests[0].model_copy(update={"amount_cents": True})
    with pytest.raises(ValidationError):
        getattr(tools, method)(invalid)


@pytest.mark.parametrize("method", ["assess", "approval_for", "changed_approval"])
def test_state_handling_validates_crafted_models(method: str) -> None:
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    invalid = state.model_copy(update={"refunded_cents": -1})
    with pytest.raises(ValidationError):
        getattr(tools, method)(invalid)


def test_scripted_replies_are_validated_on_entry() -> None:
    bad = ToolCall(name="refund.execute").model_copy(
        update={"arguments": ToolArguments.model_construct(amount_cents=True)}
    )
    with pytest.raises(ValidationError):
        ScriptedModel(replies={("r", Phase.EXECUTE): bad})


def test_transcript_intake_validates_trace() -> None:
    data = small_case()
    trace = Agent(Tools(data)).run(data.requests[0])
    bad = trace.model_copy(
        update={"decision": trace.decision.model_copy(update={"amount_cents": True})}
    )
    with pytest.raises(ValidationError):
        Transcript.from_trace(bad)


def test_echoed_history_does_not_count_payment_twice() -> None:
    data = small_case(2499, 5000)
    tools = Tools(data)
    receipt = Agent(tools).run(data.requests[0]).receipt
    assert receipt is not None
    history = [
        PriorRefund.model_construct(payment_id=receipt.receipt_id, amount_cents=2499, at=receipt.at)
    ]
    tools.data = data.model_copy(
        update={"orders": [data.orders[0].model_copy(update={"prior_refunds": history})]}
    )
    second = data.requests[0].model_copy(update={"request_id": "second", "amount_cents": 2501})
    tools.data = tools.data.model_copy(
        update={
            "retrievals": tools.data.retrievals
            + [data.retrievals[0].model_copy(update={"request_id": "second"})]
        }
    )
    trace = Agent(tools).run(second)
    assert trace.decision.executed
    assert tools.paid_cents("o", receipt.at + timedelta(seconds=1)) == 5000
    assert len(tools.ledger) == 2


class RaisingApprovals(dict[str, object]):
    def __init__(self, after: bool) -> None:
        super().__init__()
        self.after = after

    def __setitem__(self, key: str, value: object) -> None:
        if self.after:
            super().__setitem__(key, value)
        raise OSError("approval store failed")


@pytest.mark.parametrize("after", [False, True])
def test_approval_write_exception_is_audited(after: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools = Tools(data)
    state = Agent(tools).run(request).steps[-1].state
    before = len(tools.audits)
    monkeypatch.setattr(tools, "_approvals", RaisingApprovals(after))
    with pytest.raises(OSError, match="approval store failed"):
        tools.record_approval(approval_for(state))
    assert not tools._approvals and all(r.open for r in tools.reviews.values())
    assert len(tools.audits) == before + 2
    record = list(tools.audits.values())[-1]
    assert record.failure_detail and "OSError" in record.failure_detail
    assert not record.money_moved and record.paid_amount_cents == 0


@pytest.mark.parametrize(
    "store,tool",
    [
        ("proposals", "refund.propose"),
        ("reviews", "approval.request"),
        ("ledger", "refund.execute"),
    ],
)
@pytest.mark.parametrize("after", [False, True])
def test_direct_write_exceptions_are_audited(
    store: str, tool: str, after: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RaisingWrite(RaisingApprovals):
        def setdefault(self, key: str, default: object = None) -> object:
            if key not in self:
                self[key] = default
            return self[key]

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    if store == "proposals":
        tools._write("proposals", state.proposal_id or "", None)
    failing = RaisingWrite(after)
    failing.update(getattr(tools, f"_{store}"))
    monkeypatch.setattr(tools, f"_{store}", failing)
    call = ToolCall(
        name=tool, arguments=execute(state).arguments.model_copy(update={"reason": "damaged"})
    )
    with pytest.raises(PaymentError if store == "ledger" and after else OSError):
        tools.call(call, state)
    recorded = list(tools.audits.values())[-1]
    assert recorded.failure_detail and "OSError" in recorded.failure_detail
    assert recorded.money_moved == (store == "ledger" and after)
    assert recorded.paid_amount_cents == (2499 if recorded.money_moved else 0)
