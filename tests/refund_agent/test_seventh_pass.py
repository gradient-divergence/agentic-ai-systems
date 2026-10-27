import traceback
from datetime import timedelta

import pytest
from pydantic import ValidationError
from test_fifth_pass import propose
from test_review_regressions import T, small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence

from refund_agent.agent import Agent
from refund_agent.contracts import PriorRefund, Receipt, Reviewer, ToolCall
from refund_agent.data.generate import generate
from refund_agent.evaluation import _score_safety
from refund_agent.rules import approval_problem
from refund_agent.tools import PaymentError, Tools


def test_every_receipt_counts_and_future_history_is_refused() -> None:
    data = small_case(2499, 5000)
    data.orders[0].prior_refunds.append(
        PriorRefund(payment_id="old", amount_cents=4999, at=T + timedelta(seconds=1))
    )
    tools = Tools(data)
    assert tools.paid_cents("o", T) == 4999
    trace = Agent(tools).run(data.requests[0], at=T)
    assert trace.decision.reason == "future_prior_refund"
    assert not trace.decision.executed and not tools.payments
    later = Agent(tools).run(data.requests[0], at=T + timedelta(seconds=2))
    assert later.decision.reason == "amount_over_policy"


@pytest.mark.parametrize("through_agent", [False, True])
def test_failed_ledger_read_after_payment_reports_unknown(
    through_agent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedRead(dict[str, Receipt]):
        def values(self):
            if self:
                raise OSError("ledger read failed")
            return super().values()

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    ledger = FailedRead()
    monkeypatch.setattr(tools, "_ledger", ledger)
    with pytest.raises(PaymentError, match="payment_outcome_unknown") as caught:
        if through_agent:
            Agent(tools).run(data.requests[0])
        else:
            tools.call(execute(state), state)
    assert caught.value.receipt is None and not caught.value.confirmed
    assert next(iter(ledger.items()))[1].amount_cents == 2499


@pytest.mark.parametrize("entry", ["constructor", "setter"])
def test_agent_cannot_enter_reviewer_dataset(entry: str) -> None:
    data = small_case(5000, 10000)
    bad = data.model_copy(
        update={"reviewers": [Reviewer(reviewer_id="refund-agent", name="Agent", role="reviewer")]}
    )
    tools = Tools(data)
    with pytest.raises(ValidationError, match="agent_cannot_review"):
        if entry == "constructor":
            Tools(bad)
        else:
            tools.data = bad
    assert tools.data == data


def test_approval_rule_excludes_agent_even_with_staff_preparer() -> None:
    data = small_case(5000, 10000)
    request = data.requests[0].model_copy(update={"preparer_id": "staff"})
    tools, state = evidence(data, request)
    bad = data.model_copy(
        update={"reviewers": [Reviewer(reviewer_id="refund-agent", name="Agent", role="reviewer")]}
    )
    approval = approval_for(state).model_copy(update={"reviewer": "refund-agent"})
    proposal = tools.proposals[state.proposal_id or ""]
    assert approval_problem(bad, state, proposal, approval) == "invalid_approval"


@pytest.mark.parametrize(
    "amount,updates,reason",
    [
        (2499, {"idempotency_key": None}, "idempotency_key_required"),
        (2499, {"amount_cents": 100}, "amount_mismatch"),
        (5000, {}, "approval_required"),
    ],
)
def test_direct_execution_refusal_is_audited_once(amount, updates, reason) -> None:
    data = small_case(amount, 10000)
    tools, state = evidence(data, data.requests[0])
    call = execute(state)
    call = call.model_copy(update={"arguments": call.arguments.model_copy(update=updates)})
    result = tools.call(call, state)
    assert not result.ok and result.reason == reason
    assert len(tools.audits) == 1 and not tools.payments
    audit = next(iter(tools.audits.values()))
    assert audit.decision == state.decision and audit.failure_detail == reason
    assert tools.record_decision(state).record_id == audit.audit_id
    assert len(tools.audits) == 1


@pytest.mark.parametrize("field,value", [("paid_amount_cents", 999), ("receipt_id", "ghost")])
def test_safety_score_rejects_contradictory_audit(field: str, value: object) -> None:
    class ContradictoryAudit(Tools):
        @property
        def audits(self):
            return {
                key: record.model_copy(update={field: value})
                for key, record in super().audits.items()
            }

    score = _score_safety(generate(), ContradictoryAudit)
    assert score.correct == 0 and score.audited == 0
    assert score.unauthorized_cents == 0


def test_late_ledger_read_failure_cannot_audit_zero_payment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LateRead(dict[str, Receipt]):
        reads = 0

        def values(self):
            if self:
                self.reads += 1
                if self.reads == 5:
                    raise OSError("late ledger read failed")
            return super().values()

    data = small_case()
    tools = Tools(data)
    ledger = LateRead()
    monkeypatch.setattr(tools, "_ledger", ledger)
    with pytest.raises(PaymentError, match="refund-r") as caught:
        Agent(tools).run(data.requests[0])
    assert caught.value.receipt is not None and caught.value.receipt.amount_cents == 2499
    assert len(ledger) == 1
    assert all(r.money_moved for r in tools.audits.values() if r.event != "intent")


def test_raising_approval_rollback_cannot_leave_unaudited_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedAudit(dict):
        def setdefault(self, key, default=None):
            raise OSError("approval audit failed")

    class FailedRollback(dict):
        def pop(self, key, default=None):
            raise OSError("approval rollback failed")

    data = small_case(5000, 10000).model_copy(
        update={"reviewers": [Reviewer(reviewer_id="alex", name="Alex", role="reviewer")]}
    )
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_audits", FailedAudit())
    monkeypatch.setattr(tools, "_approvals", FailedRollback())
    with pytest.raises(RuntimeError, match="audit_decision_failed") as caught:
        tools.record_approval(approval_for(state))
    assert caught.value.__cause__ is not None
    assert "approval rollback failed" in "".join(traceback.format_exception(caught.value))
    monkeypatch.setattr(tools, "_audits", {})
    assert not Agent(tools).run(data.requests[0]).decision.executed
    assert not tools.payments


def test_caller_cannot_supply_a_receipt_to_audit() -> None:
    data = small_case()
    paid = Agent(Tools(data)).run(data.requests[0]).receipt
    tools, state = evidence(data, data.requests[0])
    assert paid is not None
    state.receipt, state.decision = paid, paid.decision
    assert not tools.call(ToolCall(name="audit.append"), state).ok
    assert not tools.payments and len(tools.audits) == 1
    assert list(tools.audits.values())[-1].paid_amount_cents == 0


@pytest.mark.parametrize("after", [False, True])
def test_payment_write_and_outcome_read_can_both_raise(
    after: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnavailableOutcome(dict[str, Receipt]):
        attempted = False

        def __setitem__(self, key: str, receipt: Receipt) -> None:
            self.attempted = True
            if after:
                super().__setitem__(key, receipt)
            raise OSError("payment write failed")

        def values(self):
            if self.attempted:
                raise OSError("payment outcome read failed")
            return super().values()

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    ledger = UnavailableOutcome()
    monkeypatch.setattr(tools, "_ledger", ledger)
    with pytest.raises(PaymentError, match="payment_outcome_unknown") as caught:
        tools.call(execute(state), state)
    assert caught.value.receipt is None
    assert not caught.value.confirmed and caught.value.trace is None
    assert len(ledger) == int(after)
    assert [r.event for r in tools.audits.values()] == ["intent"]


def test_public_tool_names_cannot_make_the_balance_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    data = small_case(2499, 3000)
    tools, state = evidence(data, data.requests[0])
    other = data.requests[0].model_copy(update={"request_id": "other", "amount_cents": 2000})
    tools.data = data.model_copy(
        update={
            "retrievals": data.retrievals
            + [data.retrievals[0].model_copy(update={"request_id": "other"})]
        }
    )
    other_state = state.model_copy(update={"request": other, "proposal_id": None}, deep=True)
    propose(tools, other_state)
    assert tools.call(execute(other_state, "other"), other_state).ok
    monkeypatch.setattr("refund_agent.tools.TOOL_NAMES", ("approval.request",) * 3)
    result = tools.call(execute(state, "first"), state)
    assert not result.ok and result.reason == "amount_over_policy"
    assert sum(p.amount_cents for p in tools.payments) == 2000


@pytest.mark.parametrize("identity", ["", " "])
@pytest.mark.parametrize("field", ["reviewer_id", "named_approver", "reviewer"])
def test_approval_identities_cannot_be_blank(field: str, identity: str) -> None:
    data = small_case(50001, 100000)
    tools, state = evidence(data, data.requests[0])
    if field == "reviewer_id":
        bad = data.model_copy(
            update={
                "reviewers": [
                    Reviewer.model_construct(
                        reviewer_id=identity, name="Person", role="named_approver"
                    )
                ]
            }
        )
    elif field == "named_approver":
        bad = data.model_copy(
            update={"policies": [data.policies[0].model_copy(update={field: identity})]}
        )
    else:
        approval = approval_for(state, "named_approver").model_copy(update={field: identity})
        assert tools.record_approval(approval).reason == "invalid_approval"
        return
    with pytest.raises(ValidationError):
        Tools(bad)
