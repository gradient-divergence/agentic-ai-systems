from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel
from test_fourth_pass import mutate_handout
from test_second_pass import execute
from test_stage1 import approval_for, evidence

from refund_agent.agent import Agent
from refund_agent.contracts import Outcome, Phase, State, ToolArguments, ToolCall
from refund_agent.evaluation import _positive_data
from refund_agent.model import ReplayModel, ScriptedModel, Transcript
from refund_agent.tools import Tools


def propose(tools: Tools, state: State) -> list[object]:
    returned: list[object] = []
    for name in ("order.lookup", "shipment.status", "policy.retrieve", "refund.propose"):
        call = ToolCall(
            name=name,
            arguments=ToolArguments(
                amount_cents=state.request.amount_cents, reason=state.request.reason
            ),
        )
        result = tools.call(call, state)
        assert result.ok
        returned.extend((call, result))
    return returned


def test_direct_execution_approval_cannot_be_rebound(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _positive_data()
    first = next(r for r in data.requests if r.request_id == "positive-5000")
    second = next(r for r in data.requests if r.request_id == "positive-50000")
    tools, first_state = evidence(data, first)
    approval = approval_for(first_state).model_copy(update={"approval_id": "a"})
    assert tools.record_approval(approval).ok
    assert tools.call(execute(first_state, "first", "a"), first_state).ok
    second_state = State(request=second, trace_id="second")
    propose(tools, second_state)
    exposed = first_state.approvals[0]
    monkeypatch.setattr(type(exposed), "__pydantic_setattr_handlers__", {})
    monkeypatch.setitem(cast(dict[str, object], exposed.model_config), "frozen", False)
    exposed.request_id = second.request_id
    exposed.proposal_id = second_state.proposal_id or ""
    exposed.amount_cents = 50000
    exposed.reason = "changed_mind"
    result = tools.call(execute(second_state, "second", "a"), second_state)
    assert not result.ok and result.reason == "invalid_approval"
    assert [p.amount_cents for p in tools.payments] == [5000]
    assert tools._approvals["a"] == approval


@pytest.mark.parametrize("paid", [False, True])
def test_changed_request_reproduction_is_audited(paid: bool) -> None:
    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == (2499 if paid else 5000))
    tools, state = evidence(data, request)
    if paid:
        assert tools.call(execute(state), state).ok
    else:
        assert tools.record_approval(approval_for(state)).ok
    before = deepcopy((tools.ledger, tools.proposals, tools.reviews, tools._approvals))
    audit_count = len(tools.audits)
    changed = request.model_copy(update={"text": "changed"} if paid else {"amount_cents": 6000})
    trace = Agent(tools).run(changed)
    assert len(tools.audits) == audit_count + 1
    assert (tools.ledger, tools.proposals, tools.reviews, tools._approvals) == before
    record = list(tools.audits.values())[-1]
    assert record.failure_detail == "request_content_conflict"
    assert record.decision == trace.decision
    assert record.money_moved == paid
    if paid:
        assert trace.receipt == tools.payments[0]
        assert trace.decision.executed and trace.decision.amount_cents == 2499
        assert trace.failure_detail == "request_content_conflict"
    else:
        assert trace.decision.outcome == Outcome.REFER
        assert trace.decision.reason == "request_content_conflict"
        assert trace.receipt is None and record.paid_amount_cents == 0


def stored_facts(tools: Tools) -> object:
    return deepcopy(
        (tools.data, tools.ledger, tools.audits, tools.proposals, tools.reviews, tools._approvals)
    )


def variants(value: object) -> list[object]:
    if not isinstance(value, BaseModel):
        return [value]
    fields = {name: getattr(value, name) for name in type(value).model_fields}
    updates = {
        name: "crafted" if isinstance(field, str) else field for name, field in fields.items()
    }
    return [
        value,
        value.model_dump(),
        value.model_copy(update=updates),
        type(value).model_construct(_fields_set=set(fields), **updates),
    ]


def test_every_public_handout_is_detached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools = Tools(data)
    tools.data = data.model_copy()
    state = State.model_construct(request=request, trace_id="trace-positive-5000")
    handouts = propose(tools, state)
    handouts.append(tools.assess(state))
    handouts.append(tools.call(ToolCall(name="approval.request"), state))
    first_trace = Agent(tools).run(request)
    handouts.append(first_trace)
    approval = approval_for(state)
    handouts.extend((approval, tools.record_approval(approval.model_copy(update={}))))
    handouts.append(tools.approval_for(state))
    handouts.append(tools.call(execute(state, approval_id=approval.approval_id), state))
    handouts.extend((state, tools.record_decision(state)))
    model_call = ToolCall(name="order.lookup")
    replies = {(request.request_id, Phase.ORDER): model_call}
    model = ScriptedModel(replies=replies)
    agent = Agent(tools, model)
    trace = agent.run(request)
    transcript = Transcript.from_trace(first_trace)
    transcript.write(tmp_path / "replay.json")
    replay = ReplayModel(tmp_path / "replay.json")
    handouts.extend(
        (
            transcript,
            cast(ScriptedModel, agent.model).replies,
            agent.model.reply(State(request=request, trace_id="model")),
            model_call,
            replies,
            replay.replies,
            replay.reply(State(request=request, trace_id="replay")),
        )
    )
    handouts.extend(
        (
            data,
            trace,
            agent.tools.data,
            tools.ledger,
            tools.audits,
            tools.proposals,
            tools.reviews,
            tools.payments,
            tools.payment_for(request.request_id),
            tools.completed_state(request),
        )
    )
    handouts.extend(tools.audit_for(key) for key in tools.audits)
    before = stored_facts(tools)
    model_before = deepcopy((model.replies, replay.replies))
    for handout in handouts:
        for exposed in variants(handout):
            with monkeypatch.context() as patch:
                mutate_handout(exposed, patch)
            assert stored_facts(tools) == before
            assert (model.replies, replay.replies) == model_before
