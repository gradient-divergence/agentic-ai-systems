from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import evidence

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.tools import AuditWriteError, ReconciliationError, Tools


def error_facts(error: ReconciliationError) -> tuple[str, str, str, c.Receipt | None, bool]:
    return (
        error.store,
        repr(error.expected),
        repr(error.observed),
        c._detach(error.receipt),
        error.confirmed,
    )


def corrupt_error(
    error: ReconciliationError, receipt: c.Receipt | None = None
) -> tuple[str, str, str, c.Receipt | None, bool]:
    facts = error_facts(error)
    if isinstance(error.expected, dict):
        error.expected.clear()
    elif isinstance(error.expected, c.AuditRecord):
        error.expected.freshness.clear()
    if isinstance(error.observed, dict):
        error.observed.clear()
    if error.receipt and error.receipt.evidence:
        error.receipt.evidence.freshness.clear()
    error.receipt, error.confirmed = receipt, not error.confirmed
    error.expected, error.observed = {"request_id": "changed-expected"}, {"changed": "observed"}
    return facts


@pytest.mark.parametrize("after_payment", [False, True])
def test_mutated_reconciliation_preserves_facts_and_ledger(after_payment, monkeypatch):
    from test_store_matrix import FaultStore

    data = small_case()
    tools, state = evidence(data, data.requests[0])
    store = "audits" if after_payment else "proposals"
    fault = FaultStore(
        getattr(tools, store),
        "confirmation" if after_payment else "get",
        "alter" if after_payment else "raise_before",
    )
    monkeypatch.setattr(tools, f"_{store}", fault)
    with pytest.raises(ReconciliationError) as caught:
        tools.call(execute(state), state)
    foreign = (
        Agent(Tools(small_case(100)))
        .run(data.requests[0].model_copy(update={"amount_cents": 100}))
        .receipt
    )
    assert foreign is not None
    facts = corrupt_error(caught.value, foreign)
    for action in (
        lambda: Agent(tools).run(state.request),
        lambda: tools.call(execute(state, "retry"), state),
        lambda: tools.payment_for(state.request.request_id),
    ):
        with pytest.raises(ReconciliationError) as retry:
            action()
        assert error_facts(retry.value) == facts
        receipt = retry.value.receipt
        assert (receipt is not None) == after_payment
        if receipt:
            assert receipt == tools.payments[0] and receipt.amount_cents == 2499
        corrupt_error(retry.value, foreign)
    assert len(tools.payments) == int(after_payment)


@pytest.mark.parametrize("kind", ["raise_once", "drop_once", "alter"])
@pytest.mark.parametrize("outer_id", ["r", "outer"])
def test_nested_confirmation_failure_stops_with_ledger_receipt(kind, outer_id, monkeypatch):
    from test_store_matrix import FaultStore

    data = small_case()
    outer = data.requests[0].model_copy(update={"request_id": outer_id})
    if outer_id != "r":
        data.requests.append(outer)
        data.retrievals.append(data.retrievals[0].model_copy(update={"request_id": outer_id}))
    tools, prepared = evidence(data, data.requests[0])
    monkeypatch.setattr(tools, "_audits", FaultStore(tools.audits, "confirmation", kind))
    nested: list[AuditWriteError | ReconciliationError] = []

    class Model:
        def reply(self, state: c.State) -> c.ToolCall:
            try:
                tools.call(execute(prepared), prepared)
            except (AuditWriteError, ReconciliationError) as error:
                nested.append(error)
                raise
            return c.ToolCall(name="order.lookup")

    with pytest.raises(
        ReconciliationError if kind == "alter" else AuditWriteError, match="payment=refund-r"
    ) as caught:
        Agent(tools, Model()).run(outer)
    assert len(nested) == len(tools.payments) == 1
    for error in (*nested, caught.value):
        assert error.receipt == tools.payments[0]
        assert error.receipt is not None and error.receipt.amount_cents == 2499
        assert "payment=refund-r" in str(error)
    if kind != "alter":
        assert [record.event for record in tools.audits.values()] == ["intent"]
        if outer_id == "r":
            assert (
                caught.value.trace is not None and caught.value.trace.receipt == tools.payments[0]
            )
    else:
        assert isinstance(caught.value, ReconciliationError)
        assert caught.value.confirmed and caught.value.observed is not None


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_concurrent_distinct_requests_cannot_overspend(entry):
    data = small_case()
    second = data.requests[0].model_copy(update={"request_id": "second"})
    data.requests.append(second)
    data.retrievals.append(data.retrievals[0].model_copy(update={"request_id": "second"}))
    tools, first = evidence(data, data.requests[0])
    other = first.model_copy(update={"request": second, "proposal_id": None}, deep=True)
    assert tools.call(
        c.ToolCall(
            name="refund.propose", arguments=c.ToolArguments(amount_cents=2499, reason="damaged")
        ),
        other,
    ).ok
    barrier = Barrier(2)

    def pay(state: c.State) -> c.ToolResult | c.Trace:
        barrier.wait(timeout=5)
        return (
            Agent(tools).run(state.request)
            if entry == "agent"
            else tools.call(execute(state, state.request.request_id), state)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(pay, [first, other]))
    assert (
        sum(
            result.decision.executed if isinstance(result, c.Trace) else result.ok
            for result in results
        )
        == 1
    )
    assert len(tools.payments) == 1 and tools.payments[0].amount_cents == 2499
    assert tools.paid_cents("o", first.request.at) == 2499 < 3000
