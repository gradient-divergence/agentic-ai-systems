"""Exercise 6 acceptance; run with exercises/run_reference.py action-guardrails.

These checks are outside the companion's default tests, as in the latest chapters.
The cap is configured by the operator, and each run uses a fresh isolated patch.
"""

from datetime import datetime, timedelta, timezone

import pytest

from refund_agent import contracts as c
from refund_agent import rules
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools

T = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)


def case(prior=None):
    policy = c.Policy(
        version="2026-07-01",
        effective_at=T.replace(day=1, hour=0),
        return_days=30,
        cap_cents=100000,
        text="Fictional refunds in cents.",
    )
    data = Dataset(
        orders=[],
        requests=[],
        shipments=[],
        policies=[policy],
        retrievals=[],
        correct=[],
        reviewers=[c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer")],
    )
    times = [T + timedelta(minutes=i) for i in range(3)] + [T + timedelta(days=1)]
    for i, at in enumerate(times):
        order_id, request_id = f"o-{i}", f"r-{i}"
        data.orders.append(
            c.Order(
                order_id=order_id,
                reference=order_id,
                purchased_at=at - timedelta(days=2),
                observed_at=at,
                items=[c.Item(name="lamp", quantity=1, unit_cents=30000)],
                paid_cents=30000,
                payment_id=f"p-{i}",
                prior_refunds=[prior] if i == 0 and prior else [],
            )
        )
        data.requests.append(
            c.Request(
                request_id=request_id,
                requester_id="customer",
                order_reference=order_id,
                at=at,
                amount_cents=2499,
                reason="damaged",
            )
        )
        data.shipments.append(
            c.Shipment(
                order_id=order_id,
                observed_at=at,
                events=[c.ShipmentEvent(at=at - timedelta(days=1), status="delivered")],
            )
        )
        data.retrievals.append(
            c.PolicySnapshot(request_id=request_id, version=policy.version, observed_at=at)
        )
    return data


def prepared(tools, request, approved=False):
    state = c.State(
        request=request, trace_id=f"trace-{request.request_id}", execution_at=request.at
    )
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        assert tools.call(c.ToolCall(name=name), state).ok
    state.phase = c.Phase.PROPOSE
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
        ),
        state,
    ).ok
    approval_id = None
    if approved:
        assert tools.call(c.ToolCall(name="approval.request"), state).ok
        approval = c.Approval(
            approval_id=f"approval-{request.request_id}",
            request_id=request.request_id,
            order_id=state.order.order_id,
            proposal_id=state.proposal_id,
            amount_cents=request.amount_cents,
            reason=request.reason,
            policy_version=state.policy.version,
            reviewer="alex",
            approved_at=request.at,
        )
        assert tools.record_approval(approval).ok
        assert (
            rules.approval_problem(tools.data, state, tools.proposals[state.proposal_id], approval)
            is None
        )
        approval_id = approval.approval_id
        state.execution_at += timedelta(microseconds=1)
    state.phase = c.Phase.EXECUTE
    return state, c.ToolCall(
        name="refund.execute",
        arguments=c.ToolArguments(
            amount_cents=request.amount_cents,
            idempotency_key=f"execute-{request.request_id}",
            approval_id=approval_id,
        ),
    )


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(rules, "_TIER1_DAILY_CAP", 5000)


@pytest.mark.parametrize("approved", [False, True])
def test_daily_cap_refuses_the_third_payment_and_resets_next_day(capped, approved):
    data = case()
    tools = Tools(data)
    for request in data.requests[:2]:
        assert Agent(tools).run(request).decision.executed
    state, call = prepared(tools, data.requests[2], approved)
    result = tools.call(call, state)
    assert not result.ok and result.reason == "tier1_daily_cap"
    assert tools.payment_for(state.request.request_id) is None
    refused = list(tools.audits.values())[-1]
    assert refused.event == "decision" and not refused.money_moved
    assert refused.decision.reason == "tier1_daily_cap"
    assert sum(receipt.amount_cents for receipt in tools.ledger.values()) == 4998
    assert Agent(tools).run(data.requests[3]).decision.executed
    assert sum(receipt.amount_cents for receipt in tools.ledger.values()) == 7497


def test_same_day_prior_receipts_count_even_without_a_request(capped):
    prior = c.PriorRefund(payment_id="prior", amount_cents=2499, at=T - timedelta(hours=1))
    data = case(prior)
    tools = Tools(data)
    assert any(receipt.request is None for receipt in tools.ledger.values())
    assert Agent(tools).run(data.requests[0]).decision.executed
    trace = Agent(tools).run(data.requests[1])
    assert not trace.decision.executed and trace.decision.reason == "tier1_daily_cap"
    assert sum(receipt.amount_cents for receipt in tools.ledger.values()) == 4998
    assert not list(tools.audits.values())[-1].money_moved


def test_previous_day_prior_receipts_do_not_consume_todays_cap(capped):
    prior = c.PriorRefund(payment_id="prior", amount_cents=4999, at=T - timedelta(days=1))
    data = case(prior)
    tools = Tools(data)
    assert all(Agent(tools).run(request).decision.executed for request in data.requests[:2])


def test_reset_uses_execution_date_in_utc_instead_of_request_date(capped):
    data = case()
    request = data.requests[3].model_copy(update={"at": T + timedelta(minutes=3)})
    data.requests[3] = request
    tools = Tools(data)
    for earlier in data.requests[:2]:
        assert Agent(tools).run(earlier).decision.executed
    # 19:00 in UTC-05:00 on July 15 is midnight UTC on July 16.
    at = datetime(2026, 7, 15, 19, tzinfo=timezone(timedelta(hours=-5)))
    # Evidence is observed at noon UTC, so execute after that observation.
    at += timedelta(hours=12)
    trace = Agent(tools).run(request, at=at)
    assert trace.decision.executed
    assert trace.receipt.at.date().isoformat() == "2026-07-16"


def test_an_approved_tier1_payment_below_the_cap_is_paid(capped):
    data = case()
    tools = Tools(data)
    state, call = prepared(tools, data.requests[0], approved=True)
    assert tools.call(call, state).ok
    assert tools.payment_for(state.request.request_id).amount_cents == 2499


def test_exact_cap_passes_and_tier2_does_not_consume_the_tier1_cap(capped):
    assert rules.daily_cap_problem(2499, 2501) is None
    assert rules.daily_cap_problem(2499, 2502) == "tier1_daily_cap"
    assert rules.daily_cap_problem(5000, 5000) is None


def test_the_unconfigured_cap_preserves_the_original_behavior():
    assert rules._TIER1_DAILY_CAP is None
    data = case()
    tools = Tools(data)
    assert all(Agent(tools).run(request).decision.executed for request in data.requests[:3])
