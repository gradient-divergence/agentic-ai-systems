from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest
from test_review_regressions import small_case
from test_second_pass import execute
from test_stage1 import approval_for, evidence

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools


def distant_case(extra_microseconds: int) -> Dataset:
    data = small_case()
    at = datetime(3000, 1, 20, 12, 0, 0, 1, tzinfo=timezone.utc)
    request = data.requests[0].model_copy(update={"at": at})
    order = data.orders[0].model_copy(
        update={
            "purchased_at": at - timedelta(days=2),
            "observed_at": at - timedelta(seconds=1800, microseconds=extra_microseconds),
        }
    )
    policy = data.policies[0].model_copy(
        update={"version": "3000-01-20", "effective_at": at.replace(hour=0, microsecond=0)}
    )
    shipment = data.shipments[0].model_copy(
        update={
            "observed_at": at,
            "events": [c.ShipmentEvent(at=at - timedelta(days=1), status="delivered")],
        }
    )
    snapshot = data.retrievals[0].model_copy(update={"version": policy.version, "observed_at": at})
    return data.model_copy(
        update={
            "requests": [request],
            "orders": [order],
            "policies": [policy],
            "shipments": [shipment],
            "retrievals": [snapshot],
        }
    )


@pytest.mark.parametrize("extra_microseconds", [0, 1])
def test_distant_date_preserves_microsecond_freshness(extra_microseconds: int) -> None:
    data = distant_case(extra_microseconds)
    request = data.requests[0]
    tools = Tools(data)
    trace = Agent(tools).run(request, at=request.at)
    assert trace.decision.executed is (extra_microseconds == 0)
    assert trace.decision.reason == ("stale_order" if extra_microseconds else "eligible")
    assert trace.steps[-1].state.freshness["order"] == 1800 + extra_microseconds / 1_000_000
    assert sum(p.amount_cents for p in tools.ledger.values()) == (0 if extra_microseconds else 2499)
    assert type(tools.data.requests[0].at) is datetime
    assert tools.data.requests[0].at.tzinfo is timezone.utc
    assert tools.data.requests[0].at.microsecond == 1


def interleave(
    monkeypatch: pytest.MonkeyPatch,
    tools: Tools,
    hook: str,
    first: Callable[[], c.ToolResult],
    second: Callable[[], c.ToolResult],
) -> tuple[c.ToolResult, c.ToolResult]:
    """Pause an existing hook until the competing thread reaches the boundary."""
    paused, competing = Event(), Event()
    original_hook, original_boundary = getattr(tools, hook), tools._boundary

    def pause(*args, **kwargs):
        paused.set()
        assert competing.wait(timeout=5)
        return original_hook(*args, **kwargs)

    @contextmanager
    def boundary(state: c.State, operation: str) -> Iterator[None]:
        if paused.is_set():
            competing.set()
        with original_boundary(state, operation):
            yield

    monkeypatch.setattr(tools, hook, pause)
    monkeypatch.setattr(tools, "_boundary", boundary)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first)
        assert paused.wait(timeout=5)
        second_result = pool.submit(second)
        return first_result.result(timeout=10), second_result.result(timeout=10)


def test_concurrent_distinct_requests_share_one_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = small_case(paid=10000)
    other = data.requests[0].model_copy(update={"request_id": "other"})
    data = data.model_copy(
        update={
            "requests": [*data.requests, other],
            "retrievals": [
                *data.retrievals,
                data.retrievals[0].model_copy(update={"request_id": other.request_id}),
            ],
        }
    )
    tools, first = evidence(data, data.requests[0])
    second = first.model_copy(
        update={"request": other, "trace_id": "trace-other", "proposal_id": None}, deep=True
    )
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=other.amount_cents, reason=other.reason),
        ),
        second,
    ).ok
    paid, refused = interleave(
        monkeypatch,
        tools,
        "_commit",
        lambda: tools.call(execute(first, "shared"), first),
        lambda: tools.call(execute(second, "shared"), second),
    )
    assert paid.ok and not refused.ok and refused.reason == "idempotency_conflict"
    assert len(tools.ledger) == len(tools.payments) == 1
    assert tools.ledger["shared"].request == first.request
    assert tools.ledger["shared"].amount_cents == 2499
    assert tools.payment_for(other.request_id) is None


@pytest.mark.parametrize("order", ["execution_first", "approval_first", "closed", "executed"])
def test_concurrent_approval_intake_against_execution(
    monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    data = small_case(5000, 10000).model_copy(
        update={"reviewers": [c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer")]}
    )
    tools, state = evidence(data, data.requests[0])
    approval = approval_for(state)
    if order in ("closed", "executed"):
        assert tools.record_approval(approval).ok
    candidate = (
        approval.model_copy(update={"approval_id": "late"})
        if order in ("closed", "executed")
        else approval
    )
    approval_id = None if order == "execution_first" else approval.approval_id

    def pay() -> c.ToolResult:
        return tools.call(execute(state, approval_id=approval_id), state)

    def admit() -> c.ToolResult:
        return tools.record_approval(candidate)

    intake_first = order in ("approval_first", "closed")
    hook = {
        "approval_first": "_approve",
        "closed": "_admit_approval",
        "execution_first": "_authorize",
        "executed": "_commit",
    }[order]
    first, second = interleave(
        monkeypatch, tools, hook, admit if intake_first else pay, pay if intake_first else admit
    )
    intake, payment = (first, second) if intake_first else (second, first)
    if order == "execution_first":
        assert not payment.ok and payment.reason == "approval_required"
        assert intake.ok and not tools.ledger
        assert tools.call(execute(state, approval_id=approval.approval_id), state).ok
    else:
        assert payment.ok
        assert intake.ok is (order == "approval_first")
    if order in ("closed", "executed"):
        assert intake.reason == (
            "open_review_required" if order == "closed" else "request_already_executed"
        )
        assert tools.approval_for(state) is None
    receipt = tools.payments[0]
    assert len(tools.ledger) == 1 and receipt.amount_cents == 5000
    assert receipt.approvals == (approval,)
    approval_audits = [a for a in tools.audits.values() if a.event == "approval"]
    assert len(approval_audits) == 1 and approval_audits[0].approval_ids == [approval.approval_id]
    assert tools._approvals == {approval.approval_id: approval}
    assert all(not review.open for review in tools.reviews.values())
