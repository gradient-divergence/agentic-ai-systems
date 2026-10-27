from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence
from test_tenth_pass import corrupt_error, error_facts

from refund_agent.agent import Agent
from refund_agent.baseline import score_decisions
from refund_agent.contracts import PriorRefund, Receipt, State, ToolCall, ToolResult
from refund_agent.data.generate import write_data
from refund_agent.evaluation import _positive_data, _score_safety
from refund_agent.tools import ReconciliationError, Tools


def test_unreadable_payment_after_write_names_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unreadable(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            if key in self:
                raise OSError("readback unavailable")
            return default

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_ledger", Unreadable())
    with pytest.raises(ReconciliationError, match="refund-r") as caught:
        tools.call(execute(state), state)
    assert caught.value.receipt is None and isinstance(caught.value.expected, Receipt)
    assert caught.value.expected.amount_cents == 2499
    assert len(tools._ledger) == 1
    facts = corrupt_error(caught.value)
    monkeypatch.setattr(tools, "_ledger", dict(tools._ledger))
    with pytest.raises(ReconciliationError) as retry:
        Agent(tools).run(state.request)
    assert error_facts(retry.value) == facts


def test_history_identity_cannot_be_paid_again() -> None:
    data = small_case(2499, 10000)
    data.orders[0].prior_refunds.append(
        PriorRefund(payment_id="refund-r", amount_cents=2499, at=data.requests[0].at)
    )
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0])
    assert not trace.decision.executed
    assert len(tools.ledger) == 1 and tools.paid_cents("o", data.requests[0].at) == 2499


def test_history_reorder_and_repeat_preserve_balance() -> None:
    data = small_case(100, 10000)
    at = data.requests[0].at
    history = [
        PriorRefund(payment_id="old-a", amount_cents=100, at=at),
        PriorRefund(payment_id="old-b", amount_cents=200, at=at),
    ]
    data.orders[0].prior_refunds.extend(history)
    tools = Tools(data)
    tools.data = data.model_copy(
        update={
            "orders": [
                data.orders[0].model_copy(
                    update={"prior_refunds": [history[1], history[0], history[1]]}
                )
            ]
        }
    )
    assert Agent(tools).run(data.requests[0]).decision.executed
    assert tools.paid_cents("o", at) == 400
    assert len(tools.ledger) == 3


@pytest.mark.parametrize(
    "field,value", [("amount_cents", 5001), ("at", "later"), ("order", "other")]
)
def test_echoed_history_conflict_refuses(field: str, value: object) -> None:
    data = small_case(100, 10000)
    receipt = Agent(Tools(data)).run(data.requests[0]).receipt
    assert receipt is not None
    tools = Tools(data)
    assert Agent(tools).run(data.requests[0]).decision.executed
    payment = PriorRefund(
        payment_id=receipt.receipt_id, amount_cents=receipt.amount_cents, at=receipt.at
    )
    if field == "order":
        order = data.orders[0].model_copy(
            update={"order_id": "other", "reference": "other", "prior_refunds": [payment]}
        )
    else:
        payment = payment.model_copy(
            update={field: receipt.at + timedelta(seconds=1) if field == "at" else value}
        )
        order = data.orders[0].model_copy(update={"prior_refunds": [payment]})
    second = data.requests[0].model_copy(
        update={"request_id": "second", "order_reference": order.order_id}
    )
    tools.data = data.model_copy(
        update={
            "orders": [order],
            "retrievals": data.retrievals
            + [data.retrievals[0].model_copy(update={"request_id": "second"})],
        }
    )
    trace = Agent(tools).run(second)
    assert not trace.decision.executed
    assert len(tools.payments) == 1


@pytest.mark.parametrize("entry", ["constructor", "setter"])
@pytest.mark.parametrize(
    "field,value", [("paid_cents", True), ("observed_at", "naive"), ("items", "bad")]
)
def test_other_dataset_fields_are_validated(entry: str, field: str, value: object) -> None:
    data = small_case()
    bad = data.model_copy(
        update={
            "orders": [
                data.orders[0].model_copy(
                    update={
                        field: data.requests[0].at.replace(tzinfo=None)
                        if field == "observed_at"
                        else value
                    }
                )
            ]
        }
    )
    tools = Tools(data)
    with pytest.raises(ValidationError):
        if entry == "constructor":
            Tools(bad)
        else:
            tools.data = bad
    assert tools.data == data


def test_invalid_export_is_rejected(tmp_path: Path) -> None:
    data = small_case()
    bad = data.model_copy(
        update={"requests": [data.requests[0].model_copy(update={"amount_cents": True})]}
    )
    with pytest.raises(ValidationError):
        write_data(bad, tmp_path)
    assert not (tmp_path / "data.json").exists()


def test_failed_approval_rollback_still_leaves_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    class CannotRestore(dict[str, object]):
        def __setitem__(self, key: str, value: object) -> None:
            raise OSError("review restore failed")

    class CannotApprove(dict[str, object]):
        def __setitem__(self, key: str, value: object) -> None:
            raise OSError("approval write failed")

    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools, state = evidence(data, request)
    before = len(tools.audits)
    monkeypatch.setattr(tools, "_reviews", CannotRestore(tools.reviews))
    monkeypatch.setattr(tools, "_approvals", CannotApprove())
    with pytest.raises(ReconciliationError) as caught:
        tools.record_approval(approval_for(state))
    assert len(tools.audits) == before
    assert caught.value.expected is not None and caught.value.observed is not None
    facts = corrupt_error(caught.value)
    with pytest.raises(ReconciliationError) as retry:
        Agent(tools).run(request)
    assert error_facts(retry.value) == facts
    assert not tools.payments and not tools._approvals


@pytest.mark.parametrize("side", ["prediction", "label"])
def test_score_intake_validates_copied_decisions(side: str) -> None:
    data = small_case(100)
    decision = Agent(Tools(data)).run(data.requests[0]).decision
    invalid = decision.model_copy(update={"amount_cents": 100.0})
    with pytest.raises(ValidationError):
        score_decisions(
            [invalid if side == "prediction" else decision],
            [invalid if side == "label" else decision],
        )


@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_retries_have_one_payment(same_key: bool) -> None:
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    barrier = Barrier(2)

    def pay(key: str) -> ToolResult:
        barrier.wait(timeout=5)
        return tools.call(execute(state, key), state.model_copy(deep=True))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(pay, ["same", "same" if same_key else "other"]))
    assert sum(r.ok for r in results) == (2 if same_key else 1)
    assert len(tools.payments) == 1 and len(tools.audits) == (2 if same_key else 3)


@pytest.mark.parametrize("key", [None, "", " ", "\t"])
def test_every_blank_key_is_rejected(key: str | None) -> None:
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    call = execute(state).model_copy(
        update={"arguments": execute(state).arguments.model_copy(update={"idempotency_key": key})}
    )
    assert tools.call(call, state).reason == "idempotency_key_required"
    assert not tools.payments


def test_paying_every_safety_case_fails_score() -> None:
    from refund_agent.data.generate import generate

    class PaysEverything(Tools):
        def call(self, call: ToolCall, state: State) -> ToolResult:
            if not self.payment_for(state.request.request_id):
                data = self.data
                state.order = data.orders[2]
                state.shipment_events = data.shipments[2].events
                state.policy = data.policy_at(state.request.at)
                state.freshness = dict.fromkeys(("order", "shipment", "policy"), 0.0)
                state.risk_score = 0.1
                state.execution_at = state.request.at
                self._commit(state, f"unsafe-{state.request.request_id}", 2499, None)
            return super().call(call, state)

    scored = _score_safety(generate(), PaysEverything)
    assert scored.correct == 0 and scored.total == 8 and scored.unauthorized_cents == 19992


@pytest.mark.parametrize("change", ["role", "removal"])
def test_changed_reviewer_can_be_referred_and_approved_again(change: str) -> None:
    data = _positive_data()
    request = next(r for r in data.requests if r.amount_cents == 5000)
    tools = Tools(data)
    state = Agent(tools).run(request).steps[-1].state
    assert tools.record_approval(approval_for(state)).ok
    reviewers = [r for r in data.reviewers if r.reviewer_id != "alex"]
    if change == "role":
        reviewers.append(data.reviewers[0].model_copy(update={"role": "named_approver"}))
    tools.data = data.model_copy(update={"reviewers": reviewers})
    trace = Agent(tools).run(request)
    assert not trace.decision.executed and not tools.payments
    state = trace.steps[-1].state
    replacement = approval_for(state, "named_approver").model_copy(
        update={"approval_id": "replacement", "reviewer": "alex" if change == "role" else "morgan"}
    )
    assert tools.record_approval(replacement).ok
    assert Agent(tools).run(request).decision.executed
    assert len(tools.payments) == 1
