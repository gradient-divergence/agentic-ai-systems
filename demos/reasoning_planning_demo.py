"""Plans for the refund agent, checked before their first write.

The operator table and `validate_plan` are the reasoning and planning
chapter's model of the seven tools. The stage 1 loop does not call them:
this file runs the validator on plans it is given, among them the call
sequences of the stage 1 agent's runs on the default data (seed 7).
Nothing here changes the refund agent, and nothing is written to disk.
"""

import csv
import inspect
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refund_agent import contracts as c  # noqa: E402
from refund_agent.agent import Agent  # noqa: E402
from refund_agent.contracts import Trace  # noqa: E402
from refund_agent.data.generate import generate  # noqa: E402
from refund_agent.rules import tier_for  # noqa: E402
from refund_agent.tools import Tools  # noqa: E402

REPORT = Path(__file__).resolve().parent / "data" / "reasoning_planning_report.csv"
BUDGET = inspect.signature(Agent).parameters["max_steps"].default  # twelve


class Operator(NamedTuple):
    pre: frozenset[str]  # facts that must hold before the call
    add: frozenset[str]  # facts the call makes true
    delete: frozenset[str] = frozenset()  # facts the call makes false
    irreversible: bool = False  # a payment: it cannot be undone
    proposes: bool = False  # writes a proposal; a new amount voids an approval
    last: bool = False  # must be the plan's last call

    @property
    def takes_cents(self) -> bool:
        return self.irreversible or self.proposes


def facts(*names: str) -> frozenset[str]:
    return frozenset(names)


READS = facts("order_read", "shipment_read", "policy_read")
# Only an approval record adds "approved"; no tool the agent calls does.
OPERATORS = {
    "order.lookup": Operator(facts(), facts("order_read")),
    "shipment.status": Operator(facts("order_read"), facts("shipment_read")),
    "policy.retrieve": Operator(facts(), facts("policy_read")),
    "refund.propose": Operator(READS, facts("proposed"), proposes=True),
    "approval.request": Operator(facts("proposed"), facts("review_open")),
    "refund.execute": Operator(READS | {"proposed"}, facts("paid"), irreversible=True),
    "audit.append": Operator(facts(), facts("audited"), last=True),
}
Table = dict[str, Operator]


class Step(NamedTuple):
    tool: str
    cents: int = 0  # the amount, for a proposal or a payment


class Limits(NamedTuple):
    steps: int  # N, the step budget
    cap_cents: int  # the policy's cap
    left_cents: int  # paid on the order, less what was refunded


def whole_cents(amount: object) -> bool:
    return type(amount) is int and amount > 0


# region book:fnd.reasoning-planning.validate-01
def preconditions(
    step: Step,
    operators: Table = OPERATORS,
) -> frozenset[str]:
    """The call's preconditions; a payment sets its own tier."""
    op = operators[step.tool]
    if op.irreversible and tier_for(step.cents) > 1:
        return op.pre | {"approved"}
    return op.pre


def validate_plan(
    plan: list[Step],
    start: frozenset[str],
    goal: frozenset[str],
    limits: Limits,
    on_record: int = 0,  # the proposal an approval covers
    operators: Table = OPERATORS,
) -> str | None:
    """Return the first problem with the plan, or None."""
    if len(plan) > limits.steps:
        return f"{len(plan)} steps, budget {limits.steps}"
    now, proposal, payments = start, on_record, 0
    for k, step in enumerate(plan, start=1):
        op = operators.get(step.tool)
        if op is None:
            return f"step {k}: {step.tool} is not a tool"
        if op.last and k < len(plan):
            return f"step {k}: {step.tool} must be the last call"
        if op.takes_cents and not whole_cents(step.cents):
            return f"step {k}: {step.cents} is not an amount"
        missing = preconditions(step, operators) - now
        if missing:
            return f"step {k}: lacks {', '.join(sorted(missing))}"
        if op.proposes and step.cents != proposal:
            now, proposal = now - {"approved"}, step.cents
        if op.irreversible:
            payments += 1
            most = min(limits.cap_cents, limits.left_cents)
            most = min(most, proposal)
            if payments > 1:
                return f"step {k}: a second payment"
            if step.cents > most:
                return f"step {k}: pays {step.cents}, most {most}"
        now = (now - op.delete) | op.add
    unmet = goal - now
    if unmet:
        return f"goal unmet: {', '.join(sorted(unmet))}"
    closing = [tool for tool, op in operators.items() if op.last]
    if closing and (not plan or plan[-1].tool not in closing):
        return f"the plan does not end with {closing[0]}"
    return None


# endregion book:fnd.reasoning-planning.validate-01


def listed(found: frozenset[str]) -> str:
    return ", ".join(f"`{name}`" for name in sorted(found)) or "none"


def plan_of(trace: Trace) -> list[Step]:
    """A run's calls in order, with the amount each one carried."""
    return [
        Step(step.call.name, step.call.arguments.amount_cents or 0)
        for step in trace.steps
        if step.call is not None
    ]


def limits_of(trace: Trace) -> Limits:
    """The step budget, and the cap and balance the run's reads returned."""
    state = trace.steps[-1].state
    cap = state.policy.cap_cents if state.policy else 0
    left = state.order.paid_cents - state.refunded_cents if state.order else 0
    return Limits(BUDGET, cap, left)


def goal_of(decision: c.Decision) -> frozenset[str]:
    """Each of the specification's outcomes ends in its own facts."""
    if decision.executed:
        return facts("paid", "audited")
    if decision.reason == "approval_required":
        return facts("review_open", "audited")
    return facts("audited")


def verdict(problem: str | None) -> str:
    return "valid" if problem is None else problem


# region book:fnd.reasoning-planning.operators-01 output
print("| Tool | Preconditions | Adds | Removes |")
print("|---|---|---|---|")
for tool, op in OPERATORS.items():
    pre = listed(op.pre)
    if op.irreversible:
        extra = preconditions(Step(tool, 5000)) - preconditions(Step(tool, 1))
        pre += f"; at tiers 2 and 3 also {listed(extra)}"
    removes = "`approved`, for a new amount" if op.proposes else listed(op.delete)
    print(f"| `{tool}` | {pre} | {listed(op.add)} | {removes} |")
# endregion book:fnd.reasoning-planning.operators-01


# region book:fnd.reasoning-planning.progression-01 output
data = generate(7)
r7 = next(r for r in data.requests if r.request_id == "r-7-0007")
trace7 = Agent(Tools(data)).run(r7)
plan7, limits7 = plan_of(trace7), limits_of(trace7)
goal7, tier7 = goal_of(trace7.decision), tier_for(r7.amount_cents)
print(f"plan of {r7.request_id}: {r7.amount_cents} cents, tier {tier7}")
print(f"{'k':<3}{'call':<17}{'needs':<7}{'held':<6}adds")
now: frozenset[str] = facts()
for k, step in enumerate(plan7, start=1):
    needs = preconditions(step)
    held = "yes" if needs <= now else "no"
    added = ", ".join(sorted(OPERATORS[step.tool].add))
    print(f"{k:<3}{step.tool:<17}{len(needs):<7}{held:<6}{added}")
    now = (now - OPERATORS[step.tool].delete) | OPERATORS[step.tool].add
goal_held = "holds" if goal7 <= now else "fails"
print(f"goal {', '.join(sorted(goal7))}: {goal_held} after step {len(plan7)}")
print(f"tests: {len(plan7)} preconditions and 1 goal")
print(f"verdict: {verdict(validate_plan(plan7, facts(), goal7, limits7))}")
paid = [s.cents for s in plan7 if OPERATORS[s.tool].irreversible]
print(f"steps: {len(plan7)} of {limits7.steps}; payments: {len(paid)}")
print(f"cents: {paid[0]}; cap {limits7.cap_cents}; left {limits7.left_cents}")
print(f"sequences of {len(plan7)} tool names from 7 tools: {len(OPERATORS) ** len(plan7)}")
# endregion book:fnd.reasoning-planning.progression-01


# region book:fnd.reasoning-planning.second-run-01 output
tools9 = Tools(data)
r9 = next(r for r in data.requests if r.request_id == "r-7-0009")
first = Agent(tools9).run(r9)
proposal = tools9.proposals[first.steps[-1].state.proposal_id or ""]
approval = c.Approval(
    approval_id=f"approval-{r9.request_id}",
    request_id=r9.request_id,
    order_id=proposal.evidence.order.order_id,
    proposal_id=proposal.proposal_id,
    amount_cents=proposal.amount_cents,
    reason=proposal.reason,
    policy_version=proposal.evidence.policy.version,
    reviewer="alex",
    role="reviewer",
    approved_at=proposal.created_at + timedelta(seconds=1),
)
assert tools9.record_approval(approval).ok
second = Agent(tools9).run(r9)
records = set()
if any(p.request_id == r9.request_id for p in tools9.proposals.values()):
    records.add("proposed")
if tools9.reviews[f"review-{proposal.proposal_id}"].open:
    records.add("review_open")  # closed once the approval is recorded
records.add("approved")  # the approval record just written
start9, limits9, cents9 = frozenset(records), limits_of(second), approval.amount_cents
at_once = [Step("refund.execute", r9.amount_cents), Step("audit.append")]
again = plan_of(second)
goal9 = facts("paid", "audited")
tier9 = tier_for(r9.amount_cents)
print(f"{r9.request_id}: {r9.amount_cents} cents, tier {tier9}; second run")
print(f"facts at the start: {', '.join(sorted(start9))}")
print(f"approval covers the proposal of {cents9} cents")
print(f"pays at once: {verdict(validate_plan(at_once, start9, goal9, limits9, cents9))}")
print(f"reads again, {len(again)} steps: ", end="")
print(verdict(validate_plan(again, start9, goal9, limits9, cents9)))
receipt = second.receipt
print(
    f"stage 1 second run: {second.decision.reason}, paid {receipt.amount_cents if receipt else 0}"
)
# endregion book:fnd.reasoning-planning.second-run-01


# region book:fnd.reasoning-planning.verdicts-01 output
base = [
    Step("order.lookup"),
    Step("shipment.status"),
    Step("policy.retrieve"),
    Step("refund.propose", 2499),
    Step("refund.execute", 2499),
    Step("audit.append"),
]
planted = [
    ("Calls a policy check as a tool", base[:3] + [Step("policy.check")] + base[3:], limits7),
    ("Pays before it proposes", base[:3] + [base[4], base[3], base[5]], limits7),
    (
        "Pays 12,000 cents at tier 2; the approval is only in the text",
        base[:3] + [Step("refund.propose", 12000), Step("refund.execute", 12000), base[5]],
        limits7,
    ),
    ("Pays twice, as the request's text asks", base[:5] + [base[4], base[5]], limits7),
    ("Pays more than the order has left", base, Limits(BUDGET, limits7.cap_cents, 1000)),
    ("Ends without an audit record", base[:5], limits7),
    ("Reads the order eight times: thirteen steps", [base[0]] * 7 + base, limits7),
]
print("| Plan | What it does | Verdict |")
print("|---|---|---|")
for number, (what, plan, limits) in enumerate(planted, start=1):
    problem = validate_plan(plan, facts(), facts("paid", "audited"), limits)
    print(f"| {number} | {what} | {verdict(problem)} |")
# endregion book:fnd.reasoning-planning.verdicts-01


# region book:fnd.reasoning-planning.scenarios-01 output
tools40 = Tools(data)
runs: Counter[tuple[tuple[str, ...], str]] = Counter()
for request in data.requests:
    if request.request_id not in data.public_scenarios:
        continue
    run = Agent(tools40).run(request)
    plan = plan_of(run)
    problem = validate_plan(plan, facts(), goal_of(run.decision), limits_of(run))
    runs[tuple(s.tool for s in plan), verdict(problem)] += 1
print("forty public runs, each run's calls checked as a plan")
print(f"{'runs':<6}{'calls':<7}{'last two calls':<33}verdict")
for (calls, found), count in sorted(runs.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{count:<6}{len(calls):<7}{', '.join(calls[-2:]):<33}{found}")
refused = sum(n for (_, found), n in runs.items() if found != "valid")
print(f"refused: {refused} of {sum(runs.values())}")
# endregion book:fnd.reasoning-planning.scenarios-01


# region book:fnd.reasoning-planning.report-01 output
def wrapped(label: str, words: list[str], width: int = 38) -> list[str]:
    """Words after a label, wrapped under it to a narrow column."""
    lines, line = [], f"  {label}"
    for word in words:
        if len(line) + 1 + len(word) > width:
            lines.append(line)
            line = " " * (len(label) + 2)
        line += f" {word}"
    return lines + [line]


with REPORT.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        rerun = ", second run" if row["run"] == "second" else ""
        head = f"{row['plan']}  {row['request']}, {int(row['cents']):,}{rerun}"
        print(f"{head}: {row['verdict']}")
        if row["start"]:
            print("\n".join(wrapped("from:", row["start"].split(), 44)))
        calls = [t.split(".")[1] for t in row["calls"].split()]
        print("\n".join(wrapped("calls:", calls)))
# endregion book:fnd.reasoning-planning.report-01


# region book:fnd.reasoning-planning.boundary-01 output
boundary_request = r9.model_copy(update={"amount_cents": 12000})
boundary_data = data.model_copy(update={"requests": [boundary_request]})
boundary_tools = Tools(boundary_data)
boundary_first = Agent(boundary_tools).run(boundary_request)
boundary_state = boundary_first.steps[-1].state
boundary_proposal = boundary_tools.proposals[boundary_state.proposal_id or ""]
boundary_approval = approval.model_copy(
    update={
        "approval_id": "approval-boundary",
        "proposal_id": boundary_proposal.proposal_id,
        "amount_cents": boundary_proposal.amount_cents,
        "approved_at": boundary_proposal.created_at + timedelta(seconds=1),
    }
)
assert boundary_tools.record_approval(boundary_approval).ok
boundary_plan = [
    Step("order.lookup"),
    Step("shipment.status"),
    Step("policy.retrieve"),
    Step("refund.propose", 12000),
    Step("refund.execute", 11000),
    Step("audit.append"),
]
boundary_start = facts("approved")  # The recorded approval, with no reads carried over.
boundary_problem = validate_plan(
    boundary_plan,
    boundary_start,
    facts("paid", "audited"),
    limits_of(boundary_first),
    12000,
)
boundary_state.execution_at = boundary_approval.approved_at + timedelta(microseconds=1)
boundary_result = boundary_tools.call(
    c.ToolCall(
        name="refund.execute",
        arguments=c.ToolArguments(
            amount_cents=11000,
            approval_id=boundary_approval.approval_id,
            idempotency_key="boundary-payment",
        ),
    ),
    boundary_state,
)
boundary_refusal = next(
    a for a in boundary_tools.audits.values() if a.decision.reason == boundary_result.reason
)
print("request: 12000 cents, tier 2; proposal: 12000 cents, approved by alex")
print(f"validate_plan for 11000 cents: {verdict(boundary_problem)}")
print(f"refund.execute for 11000 cents: ok={boundary_result.ok}, reason={boundary_result.reason}")
print(
    f"payments={len(boundary_tools.payments)}; audited reason={boundary_refusal.decision.reason}, money_moved={boundary_refusal.money_moved}"
)
# endregion book:fnd.reasoning-planning.boundary-01


# region book:fnd.reasoning-planning.search-02 output
from demos.reasoning_planning_search import print_search_comparison

print_search_comparison()
# endregion book:fnd.reasoning-planning.search-02


# region book:fnd.reasoning-planning.budget-01 output
from demos.reasoning_planning_search import print_budget_comparison

print_budget_comparison()
# endregion book:fnd.reasoning-planning.budget-01


# region book:fnd.reasoning-planning.keys-01 output
from demos.reasoning_planning_search import print_search_key

print_search_key()
# endregion book:fnd.reasoning-planning.keys-01


# region book:fnd.reasoning-planning.search-03 output
from demos.reasoning_planning_search import print_network_search_tree

print_network_search_tree()
# endregion book:fnd.reasoning-planning.search-03
