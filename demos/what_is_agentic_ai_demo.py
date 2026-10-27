"""Stage-0 reads, drafts and audit records for Chapter 1's two requests."""

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refund_agent.baseline import baseline
from refund_agent.contracts import Decision, Outcome, Request, State, ToolCall, ToolResult
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools

# Reuse Chapter 2's construction without printing its other examples.
with contextlib.redirect_stdout(io.StringIO()):
    from demos.modern_ai_stack_demo import REQUEST, load, sent

READS = ("order.lookup", "shipment.status", "policy.retrieve")


@dataclass
class DraftRun:
    decision: Decision
    reads: list[tuple[str, ToolResult]]
    state: State
    tools: Tools


def run_baseline(data: Dataset, request: Request) -> DraftRun:
    """Observe the real calls and retain the baseline's own state and tools."""
    reads = []
    observed = []
    call = Tools.call

    def observe(tools: Tools, tool_call: ToolCall, state: State) -> ToolResult:
        result = call(tools, tool_call, state)
        reads.append((tool_call.name, result))
        observed.append((tools, state))
        return result

    with patch.object(Tools, "call", observe):
        chosen = baseline(data, [request])[0]
    tools, state = observed[-1]
    return DraftRun(chosen, reads, state, tools)


data = load(sent)
request = next(r for r in data.requests if r.request_id == REQUEST)
assert request.reason == "changed_mind"
public = set(data.public_scenarios)
ambiguous_request = next(
    r for r in data.requests if r.request_id in public and r.failure == "ambiguous_order"
)
approved = run_baseline(data, request)
ambiguous = run_baseline(data, ambiguous_request)
matches = sum(
    ambiguous_request.order_reference in (order.order_id, order.reference) for order in data.orders
)


# region book:fnd.what-is-agentic-ai.run-01 output
print(f"{request.request_id} asks {request.amount_cents} cents for a change of mind")
name, result = approved.reads[0]
print(
    f"{name}: {result.order.order_id} paid {result.order.paid_cents} cents; "
    f"refunded {approved.state.refunded_cents} cents"
)
name, result = approved.reads[1]
delivered = max(e.at for e in result.shipment.events if e.status == "delivered")
days = (request.at - delivered).total_seconds() / 86400
print(f"{name}: delivered {days:g} days before the request")
name, result = approved.reads[2]
print(
    f"{name}: {result.policy.policy_id} version {result.policy.version}; "
    f"window {result.policy.return_days} days"
)
chosen = approved.decision
print(f"draft: {chosen.outcome.value}, tier {chosen.tier.value}, {chosen.reason}")
print(f"cites {chosen.policy_citation}")
print(f"audit records: {len(approved.tools.audits)}")
name, result = ambiguous.reads[0]
print(f"{ambiguous_request.request_id}: {name} failed: {result.reason} ({matches} matching orders)")
chosen = ambiguous.decision
citation = f"cites {chosen.policy_citation}" if chosen.policy_citation else "no citation"
print(f"draft: {chosen.outcome.value}, tier {chosen.tier.value}, {chosen.reason}; {citation}")
records = list(ambiguous.tools.audits.values())
details = "; ".join(f"{r.event} {r.decision.outcome.value}, {r.decision.reason}" for r in records)
print(f"audit records: {len(records)}; {details}")
write_calls = sum(name not in READS for run in (approved, ambiguous) for name, reply in run.reads)
executions = sum(
    run.tools.payment_for(run.state.request.request_id) is not None for run in (approved, ambiguous)
)
choices = ", ".join(outcome.value for outcome in Outcome)
print(
    f"draft kinds: {len(Outcome)} ({choices}); write tool calls: {write_calls}; executions: {executions}"
)
# endregion book:fnd.what-is-agentic-ai.run-01
