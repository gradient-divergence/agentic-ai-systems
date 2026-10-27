"""Output for the agentic loop chapter, from the refund agent's default run.

The first region runs the default command, `python -m refund_agent.run --seed 7`, into
`refund-output` under the working directory; the later regions print what the chapter
quotes from the files that run writes. Nothing here changes the refund agent.
"""

import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refund_agent import contracts as c  # noqa: E402
from refund_agent.agent import Agent  # noqa: E402
from refund_agent.baseline import baseline, score_decisions  # noqa: E402
from refund_agent.data.generate import generate  # noqa: E402
from refund_agent.run import main  # noqa: E402
from refund_agent.tools import Tools  # noqa: E402

out = Path("refund-output")  # The default run's own default folder, under the working directory.


# region book:fnd.agentic-loop.scorecard-01 output
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    main(["--seed", "7", "--out", str(out)])
for line in printed.getvalue().splitlines():
    print(line.replace("; ", ";\n    "))  # Each scorecard line, broken after its semicolon.
# endregion book:fnd.agentic-loop.scorecard-01


# region book:fnd.agentic-loop.trace-01 output
trace = json.loads((out / "traces" / "r-7-0007.json").read_text(encoding="utf-8"))
print(f"{'step':<6}{'phase':<10}{'call':<17}{'age_seconds':<13}record or reason")
for step in trace["steps"]:
    call = step["call"]["name"] if step["call"] else "(draft)"
    result = step["result"]
    age = "" if result["age_seconds"] is None else result["age_seconds"]
    detail = result["record_id"] or result["reason"]
    if step["reply"]:
        reply = step["reply"]
        detail = f"{reply['outcome']}, tier {reply['tier']}: {detail}"
    print(f"{step['index']:<6}{step['phase']:<10}{call:<17}{age!s:<13}{detail}".rstrip())
results = {step["phase"]: step["result"] for step in trace["steps"]}
request = trace["steps"][0]["state"]["request"]
delivered = results["shipment"]["shipment"]["events"][-1]
policy = results["policy"]["policy"]
print(
    f"request {request['request_id']}: {request['amount_cents']} cents, "
    f"{request['reason']}, at {request['at']}"
)
print(f"order {results['order']['order']['order_id']}; {delivered['status']} at {delivered['at']}")
print(
    f"policy {policy['policy_id']} version {policy['version']}, return_days {policy['return_days']}"
)
# endregion book:fnd.agentic-loop.trace-01


# region book:fnd.agentic-loop.audits-01 output
records = json.loads((out / "audits.json").read_text(encoding="utf-8"))["records"]
paid = [record for record in records if record["trace_id"] == "trace-r-7-0007"]
rows = [("field", [record["event"] for record in paid])]
for field in ("money_moved", "paid_amount_cents", "receipt_id", "trace_id", "at"):
    rows.append((field, [json.dumps(record[field]) for record in paid]))
rows.append(("decision.executed", [json.dumps(record["decision"]["executed"]) for record in paid]))
for name, values in rows:
    print((f"{name:<19}" + "".join(f"{value:<24}" for value in values)).rstrip())
# endregion book:fnd.agentic-loop.audits-01


# region book:fnd.agentic-loop.endings-01 output
refused = json.loads((out / "traces" / "r-7-0002.json").read_text(encoding="utf-8"))
print(f"{'step':<6}{'phase':<10}{'call':<17}{'ok':<6}reason or age_seconds")
for step in refused["steps"]:
    call = step["call"]["name"] if step["call"] else "(draft)"
    result = step["result"]
    detail = result["reason"] or ("" if result["age_seconds"] is None else result["age_seconds"])
    ok = json.dumps(result["ok"])
    print(f"{step['index']:<6}{step['phase']:<10}{call:<17}{ok:<6}{detail}".rstrip())
print()
print(f"{'request':<10}{'cents':<7}{'tier':<6}{'decision':<26}{'steps':<7}executed")
for request_id in ("r-7-0002", "r-7-0009"):
    run = json.loads((out / "traces" / f"{request_id}.json").read_text(encoding="utf-8"))
    chosen = run["decision"]
    ending = f"{chosen['outcome']} {chosen['reason']}"
    print(
        f"{request_id:<10}{chosen['amount_cents']:<7}{chosen['tier']:<6}{ending:<26}"
        f"{len(run['steps']):<7}{json.dumps(chosen['executed'])}"
    )
# endregion book:fnd.agentic-loop.endings-01


# region book:fnd.agentic-loop.share-01
def share_all_right(p: float, n: int) -> float:
    """Share of n-step runs with every step right, each step right with probability p."""
    return p**n


# endregion book:fnd.agentic-loop.share-01


# region book:fnd.agentic-loop.compound-01 output
print(f"seven steps at 0.98: {share_all_right(0.98, 7):.3f}")
print(f"eight steps at 0.98: {share_all_right(0.98, 8):.3f}")
print(f"seven steps, one at 0.90: {share_all_right(0.98, 6) * 0.90:.3f}")
# endregion book:fnd.agentic-loop.compound-01


# region book:fnd.agentic-loop.disagreement-01 output
disagreement_data = generate(7)
public = [
    r for r in disagreement_data.requests if r.request_id in disagreement_data.public_scenarios
]
labels = {d.request_id: d for d in disagreement_data.correct}
disagreements = []
for candidate, draft in zip(public, baseline(disagreement_data, public)):
    stored = json.loads(
        (out / "traces" / f"{candidate.request_id}.json").read_text(encoding="utf-8")
    )
    recorded = c.Decision.model_validate(stored["decision"])
    if not score_decisions([recorded], [draft]).correct:
        disagreements.append((candidate, labels[candidate.request_id], draft, recorded))
chosen, label, draft, recorded = disagreements[0]
disagreement_tools = Tools(disagreement_data)
replayed = Agent(disagreement_tools).run(chosen)
assert replayed.decision == recorded
review = next(iter(disagreement_tools.reviews.values()))
print(f"request {chosen.request_id}: {chosen.amount_cents} cents, tier {recorded.tier.value}")
print(f"label: {label.outcome.value} {label.reason}")
print(f"baseline: {draft.outcome.value} {draft.reason}")
print(f"agent: {recorded.outcome.value} {recorded.reason}, executed={recorded.executed}")
print(
    f"review: open={review.open}, required_role={review.required_role}; payments={len(disagreement_tools.payments)}"
)
counts = Counter((d.tier.value, d.outcome.value) for _, d, _, _ in disagreements)
print(
    "disagreements: "
    + "; ".join(
        f"tier {tier}, label {outcome}={count}" for (tier, outcome), count in sorted(counts.items())
    )
)
# endregion book:fnd.agentic-loop.disagreement-01


# region book:fnd.agentic-loop.outcomes-01 output
total_runs = 1000
right_runs = total_runs * share_all_right(0.98, 7)
wrong_runs = total_runs - right_runs
without_checker = (right_runs, 0.0, wrong_runs)
with_checker = (right_runs * 0.98, wrong_runs * 0.90 + right_runs * 0.02, wrong_runs * 0.10)
print("expected runs out of 1000; seven conditional step probabilities of 0.98")
print("checker: catches 0.90 of wrong runs; refers 0.02 of right runs")
for arm, outcomes in (("without checker", without_checker), ("with checker", with_checker)):
    correct, referred, wrong = outcomes
    print(f"{arm}: correct={correct:.3f}, referred={referred:.3f}, wrong={wrong:.3f}")
print(f"with checker, every step right: {with_checker[0] / total_runs:.3f}")
# endregion book:fnd.agentic-loop.outcomes-01
