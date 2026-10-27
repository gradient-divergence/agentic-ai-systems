"""The named approver's path above the agent's execution limit."""

import contextlib
import io
import runpy
from pathlib import Path
from typing import Any

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent

DEMO = Path(__file__).resolve().parents[1] / "demos" / "refund_gate_demo.py"


@pytest.fixture(scope="module")
def demo() -> tuple[dict[str, Any], str]:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        namespace = runpy.run_path(str(DEMO), run_name="refund_gate_demo")
    return namespace, printed.getvalue()


def test_tier3_packet_approval_agent_limit_direct_payment_and_denial(
    demo: tuple[dict[str, Any], str],
) -> None:
    namespace, printed = demo
    assert "request r-tier3: 50001 cents, tier 3" in printed
    first = namespace["tier3_first"]
    assert first.decision.outcome == c.Outcome.REFER
    assert first.decision.reason == "approval_required" and not first.decision.executed
    proposal, review = namespace["tier3_proposal"], namespace["tier3_review"]
    assert proposal.amount_cents == 50001 and proposal.reason == "damaged"
    assert proposal.evidence.policy.cap_cents >= 50001
    assert proposal.evidence.order.paid_cents >= 50001
    assert proposal.evidence.freshness == {"order": 60, "shipment": 120, "policy": 3600}
    assert review.open and review.required_role == "named_approver"
    approval = namespace["tier3_approval"]
    assert approval.reviewer == "morgan" and approval.role == "named_approver"
    assert approval.proposal_id == proposal.proposal_id
    assert namespace["tier3_approval_result"].ok
    second = namespace["tier3_second"]
    assert second.decision.reason == "approval_required" and not second.decision.executed
    assert second.steps[-1].state.approvals == [approval]
    assert namespace["tier3_result"].ok and namespace["tier3_retry"].ok
    tools = namespace["tier3_tools"]
    assert len(tools.payments) == 1 and tools.payments[0].amount_cents == 50001
    assert tools.payments[0].approvals == (approval,)
    denied, denied_tools = namespace["tier3_denied"], namespace["tier3_denied_tools"]
    assert denied.decision.reason == "already_delivered"
    assert denied.decision.outcome == c.Outcome.REFER and not denied.decision.executed
    denied_review = next(iter(denied_tools.reviews.values()))
    assert denied_review.open and denied_review.required_role == "named_approver"
    assert not denied_tools.payments
    assert "agent after approval: refer approval_required, not paid" in printed
    assert "morgan direct call: ok=True, paid_cents=50001, payments=1" in printed
    assert "same key again: ok=True, payments=1" in printed


def test_agent_independently_refuses_an_approved_tier3_execute(
    demo: tuple[dict[str, Any], str],
) -> None:
    namespace, _ = demo
    # A malicious model can ask for execution even though the normal loop never does.
    state = namespace["tier3_second"].steps[-1].state.model_copy(deep=True)
    state.phase, state.receipt = c.Phase.EXECUTE, None
    result = Agent(namespace["tier3_tools"])._process_reply(state, namespace["tier3_call"])
    assert not result.ok and result.reason == "approval_required"
    assert not state.decision.executed
    assert len(namespace["tier3_tools"].payments) == 1
