"""Chapter 38's settlement experiment and window accounting entry point.

The complete stage-6 window requires chapter 37's demand/queue, chapter 36's
stage6 path and the Part VI admission records. No replacement queue lives here.
"""

from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset, _policies
from refund_agent.rollout.org_adoption_provider import Provider
from refund_agent.rollout.org_adoption_reconcile import reconcile
from refund_agent.rollout.org_adoption_records import (
    DecisionEntry,
    Intake,
    Window,
)
from refund_agent.tools import Tools


class BalanceFixture(c.Record):
    fixture_id: c.Identity
    paid_cents: c.Cents
    console_cents: c.Cents
    request_cents: c.Cents


TEXT_BALANCE = BalanceFixture(
    fixture_id="org-adoption-text",
    paid_cents=4000,
    console_cents=4000,
    request_cents=4000,
)
EXERCISE_BALANCE = BalanceFixture(
    fixture_id="org-adoption-exercise-04",
    paid_cents=9000,
    console_cents=6000,
    request_cents=4500,
)


def balance_data(fixture):
    fixture = c._validated(BalanceFixture, fixture)
    at = datetime(2026, 11, 29, 11, tzinfo=timezone.utc)
    order_id = fixture.fixture_id + "-order"
    order = c.Order(
        order_id=order_id,
        reference=order_id,
        purchased_at=at - timedelta(days=6),
        observed_at=at - timedelta(seconds=60),
        items=[
            c.Item(name="desk lamp", quantity=1, unit_cents=fixture.paid_cents)
        ],
        paid_cents=fixture.paid_cents,
        payment_id=fixture.fixture_id + "-purchase",
    )
    request = c.Request(
        request_id=fixture.fixture_id + "-request",
        requester_id=fixture.fixture_id + "-customer",
        order_reference=order_id,
        at=at,
        amount_cents=fixture.request_cents,
        reason="damaged",
    )
    shipment = c.Shipment(
        order_id=order_id,
        observed_at=at - timedelta(seconds=60),
        events=[
            c.ShipmentEvent(at=at - timedelta(days=1), status="delivered")
        ],
    )
    policy = _policies()[-1]
    return Dataset(
        orders=[order],
        requests=[request],
        shipments=[shipment],
        policies=[policy],
        retrievals=[
            c.PolicySnapshot(
                request_id=request.request_id,
                version=policy.version,
                observed_at=at - timedelta(hours=1),
            )
        ],
        correct=[],
        reviewers=[
            c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
            c.Reviewer(
                reviewer_id="morgan", name="Morgan", role="named_approver"
            ),
        ],
    )


def decision_entries(
    tools, *, actor_id="refund-agent", review_id=None, authorization_id=None
):
    return tuple(
        DecisionEntry(
            audit=row,
            path="governed",
            actor_id=actor_id,
            review_id=review_id,
            authorization_id=authorization_id,
        )
        for row in tools.audits.values()
        if row.event in ("decision", "confirmation")
    )


def balance_case(fixture, *, early):
    data = balance_data(fixture)
    request, order = data.requests[0], data.orders[0]
    tools, provider = Tools(data), Provider()
    console_at = request.at.replace(hour=10)
    settlement = (
        console_at.replace(minute=30) if early else console_at.replace(hour=18)
    )
    provider.refund_console(
        order.order_id,
        fixture.console_cents,
        by="alex",
        at=console_at,
        request_id=request.request_id,
        settles_at=settlement,
    )
    if early:
        provider.settle(tools, settlement)
    trace = Agent(tools).run(request)
    read = next(
        step.state for step in trace.steps if step.phase == c.Phase.ORDER
    )
    balance_read = order.paid_cents - read.refunded_cents
    provider.record_tools(tools)
    provider.settle(tools, console_at.replace(hour=18))
    count = reconcile(
        [
            Intake(
                request=request,
                order_id=order.order_id,
                received_at=request.at,
                decision_deadline=request.at + timedelta(hours=4),
            )
        ],
        decision_entries(tools),
        tuple(tools.ledger.values()),
        provider.payments,
        Window(
            window_id=fixture.fixture_id,
            start=console_at,
            end=console_at.replace(hour=18, minute=1),
            closed_at=console_at.replace(hour=18, minute=1),
        ),
    )
    total = sum(row.amount_cents for row in provider.payments)
    return {
        "fixture": fixture.fixture_id,
        "early": early,
        "order": order.order_id,
        "paid_cents": order.paid_cents,
        "console_cents": fixture.console_cents,
        "settles_at": settlement.isoformat(),
        "balance_read_cents": balance_read,
        "agent_paid_cents": sum(row.amount_cents for row in tools.payments),
        "total_refunded_cents": total,
        "excess_cents": max(0, total - order.paid_cents),
        "agent_payments": len(tools.payments),
        "agent_outcome": trace.decision.outcome.value,
        "reason": trace.decision.reason,
        "count": count,
        "imported_receipts": sum(
            row.request is None for row in tools.ledger.values()
        ),
    }
