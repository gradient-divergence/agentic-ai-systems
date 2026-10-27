"""Reprint ``refund_agent.baseline.baseline`` at the book's page width.

A test holds this function equal to the package's function.
"""

from collections.abc import Sequence

from refund_agent.contracts import Decision, Outcome, Request, State, Tier, ToolCall
from refund_agent.data.generate import Dataset
from refund_agent.rules import assess, decision
from refund_agent.tools import Tools


# region book:fnd.what-is-agentic-ai.baseline-01
def baseline(
    data: Dataset, requests: Sequence[Request] | None = None
) -> list[Decision]:
    tools = Tools(data)
    decisions = []
    for request in data.requests if requests is None else requests:
        state = State(
            request=request, trace_id=f"baseline-{request.request_id}"
        )
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            result = tools.call(ToolCall(name=name), state)
            if not result.ok:
                state.decision = decision(state, Outcome.REFER, result.reason)
                break
        chosen = state.decision if state.decision is not None else assess(
            data, state
        )
        if chosen.outcome == Outcome.APPROVE and chosen.tier != Tier.AGENT:
            chosen = decision(state, Outcome.REFER, "approval_required")
        decisions.append(chosen)
    return decisions
# endregion book:fnd.what-is-agentic-ai.baseline-01
