from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_second_pass import execute
from test_stage1 import approval_for, evidence, with_observation

from refund_agent.agent import Agent
from refund_agent.contracts import (
    Approval,
    AuditRecord,
    Item,
    Order,
    Policy,
    PolicySnapshot,
    PriorRefund,
    Receipt,
    Request,
    Shipment,
    ShipmentEvent,
    State,
    ToolCall,
    ToolResult,
)
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools

T = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)


def small_case(amount: int = 2499, paid: int = 3000) -> Dataset:
    request = Request(
        request_id="r",
        requester_id="customer",
        order_reference="o",
        at=T,
        amount_cents=amount,
        reason="damaged",
    )
    order = Order(
        order_id="o",
        reference="o",
        purchased_at=T - timedelta(days=2),
        observed_at=T,
        items=[Item(name="lamp", quantity=1, unit_cents=paid)],
        paid_cents=paid,
        payment_id="p",
    )
    policy = Policy(
        version="2026-01-20",
        effective_at=T.replace(hour=0),
        return_days=90,
        cap_cents=100000,
        text="Fictional refunds, in cents.",
    )
    return Dataset(
        orders=[order],
        requests=[request],
        policies=[policy],
        correct=[],
        shipments=[
            Shipment(
                order_id="o",
                observed_at=T,
                events=[ShipmentEvent(at=T - timedelta(days=1), status="delivered")],
            )
        ],
        retrievals=[PolicySnapshot(request_id="r", version=policy.version, observed_at=T)],
    )


def refresh(data: Dataset, at: datetime) -> Dataset:
    for source in ("order", "shipment", "policy"):
        data = with_observation(data, data.requests[0], source, at)
    return data


def record(tools: Tools, state: State, approval: Approval) -> ToolResult:
    assert tools.call(ToolCall(name="approval.request"), state).ok
    intake = getattr(tools, "record_approval", None)
    assert callable(intake), "approvals must enter through a validated intake operation"
    return intake(approval)


def test_execution_counts_refund_after_request() -> None:
    data = small_case()
    order = data.orders[0].model_copy(
        update={
            "prior_refunds": [
                PriorRefund(payment_id="prior-o", amount_cents=2000, at=T + timedelta(seconds=60))
            ]
        }
    )
    data = data.model_copy(update={"orders": [order]})
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0], at=T + timedelta(seconds=120))
    assert not trace.decision.executed
    assert trace.decision.reason == "amount_over_policy"


def test_execution_ages_all_sources_at_execution() -> None:
    data = small_case()
    trace = Agent(Tools(data)).run(data.requests[0], at=T + timedelta(days=31))
    assert not trace.decision.executed
    assert trace.decision.reason == "stale_policy"
    assert trace.steps[-1].state.freshness == dict.fromkeys(
        ("order", "shipment", "policy"), 2678400
    )


def test_execution_uses_newly_effective_policy() -> None:
    data = small_case()
    later = T + timedelta(days=1)
    policy = data.policies[0].model_copy(
        update={"version": "2026-01-21", "effective_at": later, "cap_cents": 2000}
    )
    data = data.model_copy(update={"policies": data.policies + [policy]})
    trace = Agent(Tools(data)).run(data.requests[0], at=later)
    assert not trace.decision.executed
    assert trace.decision.reason == "non_current_policy"


def test_execution_rereads_changed_evidence() -> None:
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    tools.data = data.model_copy(
        update={"orders": [data.orders[0].model_copy(update={"paid_cents": 2000})]}
    )
    result = tools.call(execute(state), state)
    assert not result.ok and result.reason == "amount_over_policy"
    assert state.order is not None and state.order.paid_cents == 2000


def test_old_approval_does_not_authorize_stale_evidence(corpus: Dataset) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state)
    assert record(tools, state, approval).ok
    trace = Agent(tools).run(state.request, at=T + timedelta(days=31))
    assert not trace.decision.executed
    assert trace.decision.reason == "stale_policy"


def test_changed_policy_opens_review_for_new_proposal(corpus: Dataset) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools = Tools(data)
    first = Agent(tools).run(data.requests[0])
    approval = approval_for(first.steps[-1].state)
    assert tools.record_approval(approval).ok
    later = T + timedelta(days=1)
    policy = data.policies[0].model_copy(update={"version": "2026-01-21", "effective_at": later})
    tools.data = refresh(data, later).model_copy(
        update={
            "policies": data.policies + [policy],
            "retrievals": [
                PolicySnapshot(request_id="r", version=policy.version, observed_at=later)
            ],
        }
    )
    trace = Agent(tools).run(data.requests[0], at=later)
    assert not trace.decision.executed
    assert trace.decision.reason == "proposal_changed:policy_version"
    current = tools.proposals[trace.steps[-1].state.proposal_id or ""]
    assert current.created_at == later and current.proposal_id != approval.proposal_id
    assert any(r.open and r.proposal == current for r in tools.reviews.values())


def test_ledger_survives_legacy_balance_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailsOnce(dict[str, int]):
        def __setitem__(self, key: str, value: int) -> None:
            if not self:
                super().__setitem__("failed", 0)
                raise OSError("balance write failed")
            super().__setitem__(key, value)

    data = small_case()
    second = data.requests[0].model_copy(update={"request_id": "second"})
    data = data.model_copy(
        update={
            "requests": data.requests + [second],
            "retrievals": data.retrievals
            + [data.retrievals[0].model_copy(update={"request_id": "second"})],
        }
    )
    tools = Tools(data)
    monkeypatch.setattr(tools, "refunded_cents", FailsOnce(), raising=False)
    Agent(tools).run(data.requests[0])
    assert list(tools.audits.values())[-1].money_moved
    trace = Agent(tools).run(second)
    assert not trace.decision.executed and trace.decision.reason == "amount_over_policy"
    assert sum(p.amount_cents for p in tools.ledger.values()) == 2499


def test_preloaded_approval_cannot_pay(corpus: Dataset) -> None:
    data = small_case(5000, 10000)
    approval = Approval(
        approval_id="a",
        request_id="r",
        order_id="o",
        proposal_id="proposal-r",
        amount_cents=5000,
        reason="damaged",
        policy_version="2026-01-20",
        reviewer="Morgan",
        approved_at=T,
    )
    try:
        tools = getattr(Tools, "__call__")(data, [approval])
    except TypeError:
        return
    trace = Agent(tools).run(data.requests[0])
    assert not trace.decision.executed, "a constructor cannot preload approvals"


def test_request_identifier_cannot_review_its_request() -> None:
    data = small_case(5000, 10000)
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state).model_copy(update={"reviewer": state.request.request_id})
    assert not record(tools, state, approval).ok
    result = tools.call(execute(state, approval_id=approval.approval_id), state)
    assert not result.ok, "a requester string is not a listed reviewer"


def test_requester_cannot_be_reviewer_even_if_listed(corpus: Dataset) -> None:
    data = small_case(5000, 10000)
    data = data.model_copy(
        update={
            "requests": [data.requests[0].model_copy(update={"requester_id": "alex"})],
            "reviewers": corpus.reviewers,
        }
    )
    tools, state = evidence(data, data.requests[0])
    assert not tools.record_approval(approval_for(state)).ok
    assert not tools.payments


def test_approval_intake_requires_existing_open_review(corpus: Dataset) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state)
    intake = getattr(tools, "record_approval", None)
    assert callable(intake), "approval intake must check an existing open review"
    assert not intake(approval.model_copy(update={"proposal_id": "guessed"})).ok
    for key in tools.reviews:
        tools._write("reviews", key, None)
    assert not intake(approval).ok
    assert record(tools, state, approval).ok
    assert not intake(approval.model_copy(update={"approval_id": "second"})).ok


@pytest.mark.parametrize(
    "reviewer,role",
    [
        ("r", "reviewer"),
        ("stranger", "reviewer"),
        ("refund-agent", "reviewer"),
        ("alex", "named_approver"),
    ],
)
def test_reviewer_must_be_a_listed_person(corpus: Dataset, reviewer: str, role: str) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state).model_copy(update={"reviewer": reviewer, "role": role})
    assert not record(tools, state, approval).ok
    assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok


@pytest.mark.parametrize("mode", ["raise", "append_then_raise"])
def test_intent_failure_is_named_and_never_pays(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools(small_case())
    original = tools._store_audit

    def unavailable(record: AuditRecord) -> ToolResult:
        if mode == "append_then_raise":
            original(record)
        raise OSError("audit store unavailable")

    monkeypatch.setattr(tools, "_store_audit", unavailable)
    with pytest.raises(RuntimeError, match="audit_intent_failed") as caught:
        Agent(tools).run(tools.data.requests[0])
    assert getattr(caught.value, "receipt", None) is None
    assert getattr(caught.value, "trace", None) is not None
    assert not tools.payments
    assert all(not r.money_moved and r.receipt_id is None for r in tools.audits.values())


@pytest.mark.parametrize("mode", ["raise", "append_then_raise"])
def test_confirmation_failure_names_reachable_receipt(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = Tools(small_case())
    original = tools._store_audit

    def unavailable(record: AuditRecord) -> ToolResult:
        if record.money_moved:
            if mode == "append_then_raise":
                original(record)
            raise OSError("audit store unavailable")
        return original(record)

    monkeypatch.setattr(tools, "_store_audit", unavailable)
    with pytest.raises(RuntimeError, match="audit_confirmation_failed.*refund-r") as caught:
        Agent(tools).run(tools.data.requests[0])
    receipt = getattr(caught.value, "receipt", None)
    assert receipt is not None and receipt.amount_cents == 2499
    assert receipt in tools.ledger.values()
    assert getattr(caught.value, "trace", None) is not None
    assert tools.completed_state(tools.data.requests[0]) is not None
    assert len(tools.payments) == 1
    monkeypatch.setattr(tools, "_store_audit", original)
    assert Agent(tools).run(tools.data.requests[0]).decision.executed
    assert len(tools.payments) == 1


def test_default_scores_exact_public_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from refund_agent.data.generate import generate
    from refund_agent.run import main

    main(["--out", str(tmp_path)])
    score = json.loads((tmp_path / "scorecard.json").read_text())
    assert score["agent"]["total"] == score["baseline"]["total"] == 40
    assert {p.stem for p in (tmp_path / "traces").glob("*.json")} == set(
        generate().public_scenarios
    )
    output = capsys.readouterr().out
    assert "agreement with generated labels" in output
    assert "positive_payments: 5/5" in output


def test_never_pay_fails_positive_ledger_cases() -> None:
    import refund_agent.evaluation as evaluation

    class NeverPays(Tools):
        def _execute(self, call: ToolCall, state: State) -> ToolResult:
            return ToolResult(ok=True, record_id="fake")

    evaluate = getattr(evaluation, "positive_evaluation", None)
    assert callable(evaluate), "refusal safety alone cannot detect an agent that never pays"
    guarded = evaluate()
    abstaining = evaluate(NeverPays)
    assert guarded.total == guarded.correct == 5
    assert guarded.paid_cents == 62598
    assert abstaining.correct == abstaining.paid_cents == 0


def test_proposal_time_is_creation_time(corpus: Dataset) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0], at=T + timedelta(seconds=60))
    state = trace.steps[-1].state
    proposal = tools.proposals[state.proposal_id or ""]
    assert proposal.created_at == T + timedelta(seconds=60)
    approval = approval_for(state).model_copy(update={"approved_at": T})
    assert not tools.record_approval(approval).ok


def test_approval_has_no_expiry_with_fresh_matching_evidence(corpus: Dataset) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    tools = Tools(data)
    state = Agent(tools).run(data.requests[0]).steps[-1].state
    approval = approval_for(state)
    assert tools.record_approval(approval).ok
    later = T + timedelta(days=31)
    tools.data = refresh(data, later)
    trace = Agent(tools).run(data.requests[0], at=later)
    assert trace.decision.executed
    assert tools.payments[0].approvals == (approval,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "A revised request"),
        ("order_reference", "replacement"),
        ("amount_cents", 6000),
        ("reason", "changed_mind"),
    ],
)
def test_changed_request_requires_new_identity(corpus: Dataset, field: str, value: object) -> None:
    data = small_case(5000, 10000).model_copy(update={"reviewers": corpus.reviewers})
    replacement = data.orders[0].model_copy(
        update={"order_id": "replacement", "reference": "replacement"}
    )
    data = data.model_copy(
        update={
            "orders": data.orders + [replacement],
            "shipments": data.shipments
            + [data.shipments[0].model_copy(update={"order_id": "replacement"})],
        }
    )
    tools = Tools(data)
    state = Agent(tools).run(data.requests[0]).steps[-1].state
    approval = approval_for(state)
    assert tools.record_approval(approval).ok
    request = state.request.model_copy(update={field: value})
    before = tools.proposals
    trace = Agent(tools).run(request, at=T + timedelta(seconds=60))
    assert trace.decision.reason == "request_content_conflict"
    assert list(tools.audits.values())[-1].failure_detail == "request_content_conflict"
    assert not tools.payments and tools.proposals == before
    request = request.model_copy(update={"request_id": "revised"})
    tools.data = tools.data.model_copy(
        update={
            "retrievals": tools.data.retrievals
            + [data.retrievals[0].model_copy(update={"request_id": "revised"})]
        }
    )
    state = Agent(tools).run(request).steps[-1].state
    renewed = approval_for(state).model_copy(update={"approval_id": "renewed"})
    assert tools.record_approval(renewed).ok
    assert Agent(tools).run(request).decision.executed
    assert len(tools.payments) == 1


def test_conflicting_prior_payment_stops_execution() -> None:
    data = small_case()
    first = data.orders[0].model_copy(
        update={"prior_refunds": [PriorRefund(payment_id="prior-o", amount_cents=500, at=T)]}
    )
    data = data.model_copy(update={"orders": [first]})
    tools, state = evidence(data, data.requests[0])
    updated = first.model_copy(
        update={"prior_refunds": [PriorRefund(payment_id="prior-o", amount_cents=2000, at=T)]}
    )
    tools.data = data.model_copy(update={"orders": [updated]})
    trace = Agent(tools).run(state.request)
    assert not trace.decision.executed
    assert trace.decision.reason == "tool_error"
    assert trace.steps[-1].state.failure_detail == "tool_error:order.lookup:ValueError"
    assert not tools.payments


def test_failed_ledger_write_does_not_mark_state_executed() -> None:
    class RejectsPayment(dict[str, Receipt]):
        def __setitem__(self, key: str, value: Receipt) -> None:
            raise OSError("ledger write failed")

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    tools._ledger = RejectsPayment(tools.ledger)
    with pytest.raises(OSError, match="ledger write failed"):
        tools.call(execute(state), state)
    assert not tools.payments
    assert state.decision is None or not state.decision.executed
    assert all(not r.money_moved for r in tools.audits.values())


def test_confirmation_decision_comes_from_ledger() -> None:
    from refund_agent.contracts import Outcome
    from refund_agent.rules import decision

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    state.decision = decision(state, Outcome.APPROVE, "eligible")

    class ChangesDisplay(dict[str, Receipt]):
        def __setitem__(self, key: str, value: Receipt) -> None:
            super().__setitem__(key, value)
            assert state.decision is not None
            state.decision = state.decision.model_copy(update={"amount_cents": 1})

    tools._ledger = ChangesDisplay(tools.ledger)
    assert tools.call(execute(state), state).ok
    confirmation = next(r for r in tools.audits.values() if r.event == "confirmation")
    assert confirmation.decision.amount_cents == 2499
    assert confirmation.decision.executed and confirmation.money_moved
    assert confirmation.receipt_id == tools.payments[0].receipt_id
    assert state.decision is not None and state.execution_at is not None
    state.decision = state.decision.model_copy(update={"amount_cents": 1})
    confirmation = tools._audit_record(state, state.execution_at, "confirmation")
    assert confirmation.decision.amount_cents == 2499


def test_default_selection_uses_public_markers_not_position(corpus: Dataset) -> None:
    from refund_agent.run import _run

    ids = [r.request_id for r in corpus.requests[100:140]]
    data = corpus.model_copy(update={"public_scenarios": ids})
    _, score, traces, _ = _run(data, None)
    assert score.total == 40
    assert [t.decision.request_id for t in traces] == ids
