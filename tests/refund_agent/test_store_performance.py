"""Storage verification work must not grow with the already verified corpus."""

import pytest
from test_review_regressions import small_case

from refund_agent import tools as module
from refund_agent.agent import Agent
from refund_agent.tools import ReconciliationError, Tools


def test_store_validation_work_scales_with_new_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    validate = module._validated
    checked = 0

    def count(model, value):
        nonlocal checked
        checked += 1
        return validate(model, value)

    monkeypatch.setattr(module, "_validated", count)
    data = small_case(100, 100000)

    def work(size: int) -> int:
        nonlocal checked
        requests = [
            data.requests[0].model_copy(update={"request_id": f"r-{i}"}) for i in range(size)
        ]
        retrievals = [
            data.retrievals[0].model_copy(update={"request_id": r.request_id}) for r in requests
        ]
        batch = data.model_copy(update={"requests": requests, "retrievals": retrievals})
        checked = 0
        tools = Tools(batch)
        for request in requests:
            assert Agent(tools).run(request).decision.executed
        assert len(tools.payments) == size
        assert tools.paid_cents(data.orders[0].order_id, requests[0].at) == size * 100
        return checked

    small, large = work(20), work(40)
    assert small > 0
    assert large <= small * 2 + 100, (small, large)


def test_replaced_store_cannot_restore_a_stale_owned_dictionary() -> None:
    data = small_case(100, 100000)
    second = data.requests[0].model_copy(update={"request_id": "second"})
    data = data.model_copy(
        update={
            "requests": [*data.requests, second],
            "retrievals": [
                *data.retrievals,
                data.retrievals[0].model_copy(update={"request_id": second.request_id}),
            ],
        }
    )
    tools = Tools(data)
    assert Agent(tools).run(data.requests[0]).decision.executed
    original = tools._ledger
    tools._ledger = tools.ledger
    assert Agent(tools).run(second).decision.executed
    tools._ledger = original
    with pytest.raises(ReconciliationError):
        tools.payment_for(second.request_id)


@pytest.mark.parametrize("keep", [False, True])
def test_deletion_updates_payment_identity_request_and_balance_indexes(keep: bool) -> None:
    data = small_case(100, 10000)
    tools = Tools(data)
    receipt = Agent(tools).run(data.requests[0]).receipt
    assert receipt is not None
    tools._write("ledger", receipt.idempotency_key, None, keep=keep)
    assert not tools.ledger and not tools.payments
    assert tools.payment_for(data.requests[0].request_id) is None
    assert tools.paid_cents(data.orders[0].order_id, data.requests[0].at) == 0
    # Reimporting that identity after deletion must count it once again.
    tools._write("ledger", receipt.idempotency_key, receipt)
    assert tools.payment_for(data.requests[0].request_id) == receipt
    assert tools.paid_cents(data.orders[0].order_id, data.requests[0].at) == 100
