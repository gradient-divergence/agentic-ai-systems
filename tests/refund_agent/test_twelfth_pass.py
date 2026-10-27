from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from pathlib import Path

import pytest
from pydantic import ValidationError
from setuptools.command.build_py import build_py
from setuptools.config.pyprojecttoml import read_configuration
from setuptools.dist import Distribution
from test_review_regressions import T, small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence
from test_store_matrix import physical, run_entry

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.evaluation import _positive_data
from refund_agent.model import ScriptedModel
from refund_agent.tools import ReconciliationError, Tools


def lying_record(record):
    def compare(self, other):
        return True

    liar = type("Liar", (type(record),), {"__eq__": compare})

    field = "paid_amount_cents" if type(record) is c.AuditRecord else "amount_cents"
    if type(record) is c.ReviewItem:
        changes = {"required_role": "altered"}
    else:
        changes = {field: getattr(record, field) + 1000}
    return liar(**(record.model_dump() | changes))


class LyingStore(dict):
    def __setitem__(self, key, record):
        dict.__setitem__(self, key, lying_record(record))

    def setdefault(self, key, record=None):
        if key not in self:
            self[key] = record
        return dict.__getitem__(self, key)


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize("store", ["ledger", "audits", "proposals", "reviews", "approvals"])
def test_lying_record_write_requires_reconciliation(store, entry, monkeypatch):
    data = small_case(5000 if store in ("reviews", "approvals") else 2499, 100000)
    data.reviewers.append(c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"))
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state)
    monkeypatch.setattr(tools, f"_{store}", LyingStore(physical(tools, store)))
    actions = {
        "ledger": lambda: tools.call(execute(state), state),
        "audits": lambda: tools.call(execute(state), state),
        "proposals": lambda: tools.call(
            c.ToolCall(
                name="refund.propose",
                arguments=c.ToolArguments(amount_cents=2499, reason="damaged"),
            ),
            state,
        ),
        "reviews": lambda: tools.call(c.ToolCall(name="approval.request"), state),
        "approvals": lambda: tools.record_approval(approval),
    }
    # A proposal uses setdefault; remove the existing row to exercise a new write.
    if store == "proposals":
        tools = Tools(data)
        state = c.State(request=data.requests[0], trace_id="trace-r")
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            assert tools.call(c.ToolCall(name=name), state).ok
        monkeypatch.setattr(tools, "_proposals", LyingStore())
    with pytest.raises(ReconciliationError) as caught:
        run_entry(tools, state, actions[store], entry)
    error = caught.value
    assert error.store == store and error.observed is not None
    assert not error.confirmed and error.receipt is None
    ledger = physical(tools, "ledger")
    if store == "ledger":
        assert next(iter(ledger.values())).amount_cents == 3499
        assert "payment_outcome_unknown" in str(error)
        assert [a.event for a in tools.audits.values()] == ["intent"]
    else:
        assert not ledger
    for retry in (
        lambda: tools.call(execute(state, "retry"), state),
        lambda: Agent(tools).run(state.request),
    ):
        with pytest.raises(ReconciliationError):
            retry()
    assert physical(tools, "ledger").keys() == ledger.keys()


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize("view", ["items", "values", "get", "contents"])
def test_every_observed_view_rejects_lying_records(view, entry, monkeypatch):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    assert tools.call(execute(state), state).ok
    records = physical(tools, "ledger")
    payment = records["payment"]

    class FalseView(dict):
        def items(self):
            return (
                [(k, lying_record(v)) for k, v in dict.items(self)]
                if view == "items"
                else super().items()
            )

        def values(self):
            return (
                [lying_record(v) for v in dict.values(self)]
                if view == "values"
                else super().values()
            )

        def get(self, key, default=None):
            value = dict.get(self, key, default)
            return lying_record(value) if view == "get" and value is not None else value

    if view == "contents":

        class MaskedContents(FalseView):
            def items(self):
                return records.items()

            def values(self):
                return records.values()

            def get(self, key, default=None):
                return records.get(key, default)

        fault = MaskedContents({"payment": lying_record(payment)})
    else:
        fault = FalseView(records)
    monkeypatch.setattr(tools, "_ledger", fault)
    with pytest.raises(ReconciliationError) as caught:
        if entry == "agent":
            Agent(tools).run(state.request)
        else:
            tools.call(execute(state), state)
    assert caught.value.receipt is None and not caught.value.confirmed
    assert len(physical(tools, "ledger")) == 1


class ZeroAge(datetime):
    def astimezone(self, tz=None):
        return self

    def __rsub__(self, other):
        return timedelta(0)


class MutableAge(datetime):
    offset = timedelta(0)

    def astimezone(self, tz=None):
        return self

    def __deepcopy__(self, memo):
        return self

    def __rsub__(self, other):
        return other - datetime.fromtimestamp(self.timestamp(), timezone.utc) + self.offset


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize("kind", [ZeroAge, MutableAge])
def test_active_policy_timestamp_cannot_hide_31_day_age(kind, entry):
    data = small_case()
    at = T + timedelta(days=31)
    data.requests[0] = data.requests[0].model_copy(update={"at": at})
    data.orders[0] = data.orders[0].model_copy(update={"observed_at": at})
    data.shipments[0] = data.shipments[0].model_copy(update={"observed_at": at})
    tools = Tools(data)
    stamp = kind.fromtimestamp(T.timestamp(), timezone.utc)
    bad = data.model_copy(
        update={"retrievals": [data.retrievals[0].model_copy(update={"observed_at": stamp})]}
    )
    with pytest.raises(ValidationError):
        tools.data = bad
    stamp.offset = timedelta(days=-31)
    assert type(tools.data.retrievals[0].observed_at) is datetime
    if entry == "agent":
        trace = Agent(tools).run(data.requests[0])
        assert not trace.decision.executed and trace.decision.reason == "stale_policy"
    else:
        _, state = evidence(data, data.requests[0])
        result = tools.call(execute(state), state)
        assert not result.ok and result.reason == "stale_policy"
    assert not tools.payments
    assert list(tools.audits.values())[-1].decision.reason == "stale_policy"


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize(
    "stamp", [ZeroAge.fromtimestamp(T.timestamp(), timezone.utc), T.replace(tzinfo=None)]
)
def test_invalid_execution_timestamp_is_refused_and_audited(stamp, entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    if entry == "agent":
        trace = Agent(tools).run(state.request, at=stamp)
        assert not trace.decision.executed
    else:
        state.execution_at = stamp
        assert not tools.call(execute(state), state).ok
    assert not tools.payments
    assert list(tools.audits.values())[-1].failure_detail


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_request_timestamp_subclass_is_rejected(entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    bad = state.request.model_copy(
        update={"at": ZeroAge.fromtimestamp(T.timestamp(), timezone.utc)}
    )
    if entry == "agent":
        with pytest.raises(ValidationError):
            Agent(tools).run(bad)
    else:
        state.request = bad
        assert not tools.call(execute(state), state).ok
    assert not tools.payments


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_approval_timestamp_subclass_is_rejected_and_audited(entry):
    data = _positive_data()
    request = data.requests[3]
    tools, state = evidence(data, request)
    approval = approval_for(state).model_copy(
        update={"approved_at": ZeroAge.fromtimestamp(T.timestamp(), timezone.utc)}
    )
    result = []
    run_entry(tools, state, lambda: result.append(tools.record_approval(approval)), entry)
    assert not result[0].ok and result[0].reason == "invalid_approval"
    assert not tools.payments
    assert any(a.failure_detail == "invalid_approval" for a in tools.audits.values())


class DerivedInt(int):
    pass


class DerivedStr(str):
    pass


class ForeignOutcome(str, Enum):
    APPROVE = "approve"


class ForeignTier(IntEnum):
    AGENT = 1


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("amount_cents", DerivedInt(2499)),
        ("amount_cents", c.Tier.AGENT),
        ("idempotency_key", DerivedStr("payment")),
        ("idempotency_key", c.Outcome.APPROVE),
    ],
)
def test_tool_scalar_subclasses_are_refused_and_audited(field, value, entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    call = execute(state)
    bad = call.model_copy(update={"arguments": call.arguments.model_copy(update={field: value})})

    class Model(ScriptedModel):
        def reply(self, state):
            return bad if state.phase == c.Phase.EXECUTE else super().reply(state)

    if entry == "agent":
        trace = Agent(tools, Model()).run(state.request)
        assert not trace.decision.executed and trace.decision.reason == "invalid_tool_arguments"
    else:
        result = tools.call(bad, state)
        assert not result.ok and result.reason == "invalid_tool_arguments"
    assert not tools.payments
    assert list(tools.audits.values())[-1].failure_detail


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize(
    "field,value",
    [("amount_cents", DerivedInt(2499)), ("request_id", DerivedStr("r"))],
)
def test_request_scalar_subclasses_are_rejected(field, value, entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    call = execute(state)
    bad = state.request.model_copy(update={field: value})
    if entry == "agent":
        with pytest.raises(ValidationError):
            Agent(tools).run(bad)
    else:
        state.request = bad
        assert not tools.call(call, state).ok
    assert not tools.payments


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize("method", ["payment_for", "audit_for", "paid_cents", "policy_at"])
def test_public_lookup_intake_rejects_scalar_and_clock_subclasses(method, entry):
    data = small_case(5000, 10000)
    tools, state = evidence(data, data.requests[0])
    attempts = []
    actions = {
        "payment_for": lambda: tools.payment_for(DerivedStr("r")),
        "audit_for": lambda: tools.audit_for(DerivedStr("audit")),
        "paid_cents": lambda: tools.paid_cents(
            "o", ZeroAge.fromtimestamp(T.timestamp(), timezone.utc)
        ),
        "policy_at": lambda: tools.data.policy_at(
            ZeroAge.fromtimestamp(T.timestamp(), timezone.utc)
        ),
    }

    def attempt():
        with pytest.raises(ValueError):
            actions[method]()
        attempts.append("refused")

    run_entry(tools, state, attempt, entry)
    assert attempts == ["refused"] and not tools.payments


@pytest.mark.parametrize("entry", ["tools", "agent"])
@pytest.mark.parametrize(
    "field,value", [("outcome", ForeignOutcome.APPROVE), ("tier", ForeignTier.AGENT)]
)
def test_foreign_enum_values_are_refused_and_audited(field, value, entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    bad = tools.assess(state).model_copy(update={field: value})

    class Model(ScriptedModel):
        def reply(self, state):
            return bad if state.phase == c.Phase.ASSESS else super().reply(state)

    if entry == "agent":
        assert not Agent(tools, Model()).run(state.request).decision.executed
    else:
        state.decision = bad
        assert not tools.call(execute(state), state).ok
    assert not tools.payments and list(tools.audits.values())[-1].failure_detail


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_execution_before_arrival_cannot_create_backdated_proposal(entry):
    data = _positive_data()
    at = data.requests[0].at
    request = data.requests[0].model_copy(update={"at": at + timedelta(days=31)})
    tools = Tools(data)
    if entry == "agent":
        trace = Agent(tools).run(request, at=at)
        assert not trace.decision.executed
        assert trace.decision.reason == "execution_before_request"
    else:
        state = c.State(request=request, trace_id="backdated", execution_at=at)
        for name in (
            "order.lookup",
            "shipment.status",
            "policy.retrieve",
            "refund.propose",
            "refund.execute",
        ):
            result = tools.call(
                c.ToolCall(
                    name=name,
                    arguments=c.ToolArguments(
                        amount_cents=2499, reason="damaged", idempotency_key="backdated"
                    ),
                ),
                state,
            )
            assert not result.ok and result.reason == "execution_before_request"
    assert not tools.proposals and not tools.payments
    assert list(tools.audits.values())[-1].failure_detail == "execution_before_request"


@pytest.mark.parametrize("entry", ["tools", "agent"])
def test_execution_at_arrival_uses_plain_detached_utc_time(entry):
    data = small_case()
    tools, state = evidence(data, data.requests[0])
    at = T.astimezone(timezone(timedelta(hours=3)))
    if entry == "agent":
        receipt = Agent(tools).run(state.request, at=at).receipt
    else:
        state.execution_at = at
        assert tools.call(execute(state), state).ok
        receipt = state.receipt
    assert receipt is not None and receipt.at == T
    assert type(receipt.at) is datetime and receipt.at.tzinfo is timezone.utc
    assert receipt.at is not at


def test_positive_dataset_is_discovered_as_package_data():
    root = Path(__file__).resolve().parents[2]
    config = read_configuration(root / "pyproject.toml")
    package_data = config["tool"]["setuptools"].get("package-data", {})
    command = build_py(Distribution({"package_data": package_data}))
    command.finalize_options()
    command.analyze_manifest()
    path = root / "refund_agent" / "data"
    assert str(path / "positive.json") in command.find_data_files("refund_agent.data", str(path))
