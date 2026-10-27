"""Exercise 10 acceptance; run with exercises/run_reference.py perception-tools.

Like the latest chapters' acceptance files, these checks are outside tests/.
They require the exercise patch, rather than change the book's tool namespace.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent import rules
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset, generate, instructor_data
from refund_agent.model import ScriptedModel
from refund_agent.tools import Tools

T = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)


def case(age=300):
    event = c.ShipmentEvent(at=T - timedelta(days=1), status="delivered")
    request = c.Request(
        request_id="r",
        requester_id="customer",
        order_reference="o",
        at=T,
        amount_cents=2499,
        reason="damaged",
    )
    order = c.Order(
        order_id="o",
        reference="o",
        purchased_at=T - timedelta(days=2),
        observed_at=T,
        items=[c.Item(name="lamp", quantity=1, unit_cents=10000)],
        paid_cents=10000,
        payment_id="p",
    )
    policy = c.Policy(
        version="2026-07-01",
        effective_at=T.replace(day=1, hour=0),
        return_days=30,
        cap_cents=100000,
        text="Fictional refunds in cents.",
    )
    return Dataset(
        orders=[order],
        requests=[request],
        policies=[policy],
        correct=[],
        shipments=[c.Shipment(order_id="o", observed_at=T, events=[event])],
        retrievals=[c.PolicySnapshot(request_id="r", version=policy.version, observed_at=T)],
        carrier_events=[
            c.CarrierEvent(order_id="o", observed_at=T - timedelta(seconds=age), event=event)
        ],
    )


def prepared(tools, request):
    state = c.State(request=request, trace_id="trace-r", execution_at=T)
    for name in ("order.lookup", "shipment.status", "carrier.track", "policy.retrieve"):
        assert tools.call(c.ToolCall(name=name), state).ok
    state.phase = c.Phase.PROPOSE
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
        ),
        state,
    ).ok
    state.phase = c.Phase.EXECUTE
    return state


@pytest.mark.parametrize(
    "age, reason", [(1800, "eligible"), (1801, "stale_carrier_event"), (-1, "future_carrier_event")]
)
def test_carrier_time_sets_the_boundary_and_payment_outcome(age, reason):
    data = case(age)
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0], at=T)
    read = next(step for step in trace.steps if step.phase == c.Phase.CARRIER)
    assert read.result.carrier_event == data.carrier_events[0]
    assert read.result.age_seconds == age
    assert read.state.freshness["carrier_event"] == age
    assert trace.decision.reason == reason
    assert trace.decision.executed == (age == 1800)
    assert bool(tools.payments) == (age == 1800)
    assert list(tools.audits.values())[-1].money_moved == (age == 1800)


@pytest.mark.parametrize("observed", [None, T.replace(tzinfo=None)])
def test_missing_or_naive_carrier_time_fails_validation(observed):
    fields = {"order_id": "o", "event": case().carrier_events[0].event}
    if observed is not None:
        fields["observed_at"] = observed
    with pytest.raises(ValidationError):
        c.CarrierEvent.model_validate(fields)


def test_absent_carrier_record_is_refused_and_audited():
    data = case().model_copy(update={"carrier_events": []})
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0], at=T)
    assert trace.decision.reason == "carrier_event_not_found"
    assert not tools.payments
    assert list(tools.audits.values())[-1].decision.reason == "carrier_event_not_found"


def test_the_latest_carrier_observation_is_returned():
    data = case()
    latest = data.carrier_events[0]
    older = latest.model_copy(update={"observed_at": T - timedelta(seconds=1801)})
    data = data.model_copy(update={"carrier_events": [older, latest]})
    tools = Tools(data)
    trace = Agent(tools).run(data.requests[0], at=T)
    read = next(step for step in trace.steps if step.phase == c.Phase.CARRIER)
    assert read.result.carrier_event == latest
    assert read.result.age_seconds == 300
    assert trace.decision.executed


@pytest.mark.parametrize("argument", [{"age_seconds": 0}, {"tracking_number": "invented"}])
def test_the_caller_cannot_supply_age_or_an_undeclared_argument(argument):
    data = case()
    tools = Tools(data)
    state = c.State(request=data.requests[0], trace_id="trace-r")
    call = c.ToolCall.model_construct(name="carrier.track", arguments=argument)
    result = tools.call(call, state)
    assert not result.ok and result.reason == "invalid_tool_arguments"
    assert not tools.payments


def test_a_model_cannot_replace_the_carrier_age_with_zero():
    class ForgedAge(ScriptedModel):
        def reply(self, state):
            state.freshness["carrier_event"] = 0
            return super().reply(state)

    data = case(1801)
    tools = Tools(data)
    trace = Agent(tools, model=ForgedAge()).run(data.requests[0], at=T)
    assert trace.decision.reason == "stale_carrier_event"
    assert trace.steps[-1].state.freshness["carrier_event"] == 1801
    assert not tools.payments


def test_execution_rereads_the_carrier_before_paying():
    data = case(1800)
    tools = Tools(data)
    state = prepared(tools, data.requests[0])
    tools.data = case(1801)
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(amount_cents=2499, idempotency_key="execute-r"),
        ),
        state,
    )
    assert not result.ok and result.reason == "stale_carrier_event"
    assert state.freshness["carrier_event"] == 1801
    assert not tools.payments
    assert list(tools.audits.values())[-1].decision.reason == "stale_carrier_event"


def test_missing_carrier_freshness_or_read_cannot_authorize_payment():
    ages = {"order": 0.0, "shipment": 0.0, "policy": 0.0}
    assert rules.freshness_problem(ages) == "missing_carrier_event_freshness"
    data = case()
    tools = Tools(data)
    state = prepared(tools, data.requests[0])
    state.carrier_event = None
    assert rules.evidence_problem(data, state) == "missing_carrier_event"


def test_generated_and_instructor_datasets_keep_the_carrier_source():
    data = generate(7, orders=8, requests=8, instructor_seed=19)
    assert len(data.carrier_events) == len(data.orders)
    held = instructor_data(data)
    assert {row.order_id for row in held.carrier_events} == {row.order_id for row in held.orders}
