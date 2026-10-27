"""Exercise 6: check the submitted Order validator in its isolated package copy."""

import json

import pytest
from pydantic import ValidationError

from refund_agent.baseline import baseline, score_decisions
from refund_agent.contracts import Order
from refund_agent.data.generate import generate
from refund_agent.tools import Tools


@pytest.fixture(scope="module")
def data():
    return generate(7)


def test_all_generated_orders_are_accepted_and_dollars_are_rejected(data):
    for order in data.orders:
        assert Order.model_validate_json(order.model_dump_json()) == order
    order = next(o for o in data.orders if o.order_id == "o-0007").model_dump(mode="json")
    sent = {
        **order,
        "items": [{"name": "physical desk lamp", "quantity": 1, "unit_cents": 4200}],
        "paid_cents": 4200,
        "prior_refunds": [],
    }
    assert Order.model_validate_json(json.dumps(sent)).paid_cents == 4200
    with pytest.raises(ValidationError):
        Order.model_validate_json(json.dumps({**sent, "paid_cents": 42}))
    renamed = dict(sent)
    renamed["amount_paid_cents"] = renamed.pop("paid_cents")
    for release in (
        renamed,
        {**sent, "currency": "USD"},
        {**sent, "observed_at": "2026-03-22T15:07:27"},
    ):
        with pytest.raises(ValidationError):
            Order.model_validate_json(json.dumps(release))


def test_total_sums_quantities_and_every_item(data):
    sent = data.orders[0].model_dump(mode="json")
    sent.update(
        items=[
            {"name": "a", "quantity": 2, "unit_cents": 100},
            {"name": "b", "quantity": 3, "unit_cents": 300},
        ],
        paid_cents=1100,
    )
    assert Order.model_validate_json(json.dumps(sent)).paid_cents == 1100
    for wrong in (400, 500, 1000):
        with pytest.raises(ValidationError):
            Order.model_validate_json(json.dumps({**sent, "paid_cents": wrong}))


def test_public_baseline_still_agrees_without_writes(data, monkeypatch):
    original = Tools.call

    def reads_only(tools, call, state):
        assert call.name in ("order.lookup", "shipment.status", "policy.retrieve")
        return original(tools, call, state)

    monkeypatch.setattr(Tools, "call", reads_only)
    public = set(data.public_scenarios)
    requests = [r for r in data.requests if r.request_id in public]
    labels = [d for d in data.correct if d.request_id in public]
    card = score_decisions(baseline(data, requests), labels)
    assert (card.correct, card.total, card.approve, card.deny, card.refer, card.executed) == (
        40,
        40,
        4,
        22,
        14,
        0,
    )
