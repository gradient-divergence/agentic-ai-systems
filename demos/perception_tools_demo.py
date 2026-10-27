"""Output for the perception and tools chapter.

The setup runs the refund agent's default command, `python -m refund_agent.run --seed 7`,
into `refund-output` under the working directory. The regions print what the chapter
quotes from that run, the package's own source for three listings (with whitespace
reflowed), calls made directly to the package's tools, and the convolution arithmetic
of the chapter's vision sections. Nothing here changes the refund agent.
"""

# ruff: noqa: E402

import ast
import inspect
import json
import math
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from demos.output_wrap import print_wrapped
from refund_agent import contracts, rules
from refund_agent.agent import Agent
from refund_agent.contracts import Phase, State, ToolArguments, ToolCall
from refund_agent.data.generate import generate
from refund_agent.model import ScriptedModel
from refund_agent.run import main
from refund_agent.tools import Tools

out = Path("refund-output")  # The default run's own default folder, under the working directory.
main(["--seed", "7", "--out", str(out)])
data = generate(7)
records = json.loads((out / "audits.json").read_text(encoding="utf-8"))["records"]
LIMITS = dict(rules._FRESHNESS_LIMITS)
READS = (("order", "order.lookup"), ("shipment", "shipment.status"), ("policy", "policy.retrieve"))


def trace(request_id: str) -> dict:
    return json.loads((out / "traces" / f"{request_id}.json").read_text(encoding="utf-8"))


def request_for(request_id: str) -> contracts.Request:
    return next(r for r in data.requests if r.request_id == request_id)


def print_source(module: object, *names: str) -> None:
    """Print top-level definitions of a module, preserving their tokens."""
    text = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
    lines = text.splitlines()
    chunks = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found = [t.id for t in targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            found = [node.name]
        else:
            continue
        if set(found) & set(names):
            chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    print_wrapped("\n\n".join(chunks))


def prepared(request: contracts.Request) -> tuple[Tools, State]:
    """A fresh set of tools, with the three reads and the proposal done, at the execute phase."""
    tools = Tools(data)
    state = State(request=request, trace_id=f"trace-{request.request_id}")
    for _, name in READS:
        assert tools.call(ToolCall(name=name), state).ok
    state.phase = Phase.PROPOSE
    propose = ToolArguments(amount_cents=request.amount_cents, reason=request.reason)
    assert tools.call(ToolCall(name="refund.propose", arguments=propose), state).ok
    state.phase = Phase.EXECUTE
    return tools, state


def execute(state: State, key: str | None, amount: int | None = None) -> ToolCall:
    amount = state.request.amount_cents if amount is None else amount
    return ToolCall(
        name="refund.execute",
        arguments=ToolArguments(amount_cents=amount, idempotency_key=key),
    )


# region book:fnd.perception-tools.stale-01 output
run = trace("r-7-0002")
request = run["steps"][0]["state"]["request"]
ages = {step["phase"]: step["result"]["age_seconds"] for step in run["steps"]}
kept = [r for r in records if r["trace_id"] == run["trace_id"]]
decision = run["decision"]
print(f"request {request['request_id']}: {request['amount_cents']} cents, {request['reason']}")
print(f"shipment record age at use: {ages['shipment']:,.0f} s; limit {LIMITS['shipment']:,} s")
print(f"decision: {decision['outcome']}, {decision['reason']}, tier {decision['tier']}")
print(f"audit records: {len(kept)}; money moved: {str(kept[0]['money_moved']).lower()}")
# endregion book:fnd.perception-tools.stale-01


# region book:fnd.perception-tools.freshness-01 output
print_source(rules, "_FRESHNESS_LIMITS", "freshness_problem")
# endregion book:fnd.perception-tools.freshness-01


# region book:fnd.perception-tools.reads-01 output
run = trace("r-7-0007")
ages = {step["phase"]: step["result"]["age_seconds"] for step in run["steps"]}
paid = run["steps"][0]["state"]["request"]["at"] == run["receipt"]["at"]
print(f"{'source':<10}{'age at use':>12}{'limit':>13}{'margin':>13}")
margins = {}
for source, _ in READS:
    margins[source] = LIMITS[source] - ages[source]
    print(f"{source:<10}{ages[source]:>10,.0f} s{LIMITS[source]:>11,} s{margins[source]:>11,.0f} s")
first = min(margins, key=margins.get)
print(f"paid at the request's own time: {str(paid).lower()}")
print(
    f"expiry: {margins[first]:,.0f} s ({margins[first] / 60:.0f} min) after the request, by {first}"
)
# endregion book:fnd.perception-tools.reads-01


# region book:fnd.perception-tools.boundary-01 output
cases = (("1,800 s", 1800.0), ("1,801 s", 1801.0), ("-1 s", -1.0), ("missing", None))
cases += (("not a number", math.nan),)
for label, age in cases:
    ages = {"order": 60.0, "policy": 3600.0}
    if age is not None:
        ages["shipment"] = age
    print(f"shipment age {label:<13} -> {rules.freshness_problem(ages) or 'fresh'}")
# endregion book:fnd.perception-tools.boundary-01


# region book:fnd.perception-tools.versions-01 output
print_wrapped(f"{'request':<10}{'injected':<17}{'read':<12}{'in force':<12}{'age':>11}  reason")
for request_id in ("r-7-0001", "r-7-0003"):
    run = trace(request_id)
    request = request_for(request_id)
    policy = next(s["result"]["policy"] for s in run["steps"] if s["phase"] == "policy")
    age = next(s["result"]["age_seconds"] for s in run["steps"] if s["phase"] == "policy")
    in_force = data.policy_at(request.at).version
    print_wrapped(
        f"{request_id:<10}{request.failure:<17}{policy['version']:<12}{in_force:<12}"
        f"{age:>9,.0f} s  {run['decision']['reason']}"
    )
# endregion book:fnd.perception-tools.versions-01


# region book:fnd.perception-tools.contracts-01 output
print_source(contracts, "Cents")
print_wrapped()
record = inspect.getsource(contracts.Record).splitlines()
print_wrapped("\n".join(record[:2] + ["    ..."]))
print_wrapped()
print_source(contracts, "ToolArguments", "ToolCall", "ToolResult")
# endregion book:fnd.perception-tools.contracts-01


# region book:fnd.perception-tools.calls-01 output
class LaxArguments(BaseModel):  # A schema with the library's defaults, for comparison.
    amount_cents: int


tools, state = prepared(request_for("r-7-0007"))
calls = (
    ("amount '2499', a string", {"amount_cents": "2499", "idempotency_key": "k-1"}),
    ("field priority added", {"amount_cents": 2499, "idempotency_key": "k-1", "priority": 1}),
)
for label, arguments in calls:
    call = ToolCall.model_construct(name="refund.execute", arguments=arguments)
    print(f"{label:<24}-> {tools.call(call, state).reason}")
print(f"{'tool policy.check':<24}-> {tools.call(ToolCall(name='policy.check'), state).reason}")
print(f"payments in the ledger: {len(tools.payments)}")
lax = LaxArguments.model_validate({"amount_cents": "2499", "priority": "high"})
print(f"lax schema, same two inputs: {lax.model_dump()}")
# endregion book:fnd.perception-tools.calls-01


# region book:fnd.perception-tools.execute-01 output
source = textwrap.dedent(inspect.getsource(Tools._execute)).rstrip()
# Group the conditional assignment so its continuation remains valid Python.
source = source.replace(
    "approved = previous.approvals[0].approval_id if previous.approvals else None",
    "approved = (\n"
    "            previous.approvals[0].approval_id\n"
    "            if previous.approvals else None\n"
    "        )",
)
print_wrapped(source)
# endregion book:fnd.perception-tools.execute-01


# region book:fnd.perception-tools.keys-01 output
tools, state = prepared(request_for("r-7-0007"))
key = "execute-r-7-0007"
steps = (
    ("first call", execute(state, key)),
    ("same key, same arguments", execute(state, key)),
    ("same key, amount + 1", execute(state, key, state.request.amount_cents + 1)),
    ("new key, same request", execute(state, "another-key")),
    ("blank key", execute(state, " ")),
)
for label, call in steps:
    result = tools.call(call, state)
    print(f"{label:<26}-> {result.record_id if result.ok else result.reason}")
print(f"payments in the ledger: {len(tools.payments)}")
tools, state = prepared(request_for("r-7-0007"))
barrier = Barrier(2)


def pay(key: str) -> str:
    barrier.wait(timeout=5)
    result = tools.call(execute(state, key), state.model_copy(deep=True))
    return result.record_id if result.ok else result.reason


with ThreadPoolExecutor(max_workers=2) as pool:
    results = sorted(pool.map(pay, ["key-a", "key-b"]))
print(f"two calls at once: {' and '.join(results)}")
print(f"payments in the ledger: {len(tools.payments)}")
# endregion book:fnd.perception-tools.keys-01


# region book:fnd.perception-tools.failures-01 output
print("| Injected failure | Request | Last step | Reason | Audit records | Money moved |")
print("|---|---|---|---|---|---|")
for request in data.requests:
    if request.failure is None or request.request_id not in data.public_scenarios:
        continue
    run = trace(request.request_id)
    last = run["steps"][-2]
    step = last["call"]["name"] if last["call"] else last["phase"]
    kept = [r for r in records if r["trace_id"] == run["trace_id"]]
    moved = "yes" if any(r["money_moved"] for r in kept) else "no"
    print(
        f"| `{request.failure}` | `{request.request_id}` | `{step}` "
        f"| `{run['decision']['reason']}` | {len(kept)} | {moved} |"
    )


class MalformedModel(ScriptedModel):
    """The scripted model, except that its payment call carries the amount as a string."""

    def reply(self, state: State) -> ToolCall | contracts.Decision:
        if state.phase != Phase.EXECUTE:
            return super().reply(state)
        arguments = {"amount_cents": "2499", "idempotency_key": "k-1"}
        return ToolCall.model_construct(name="refund.execute", arguments=arguments)


tools = Tools(data)
run = Agent(tools, MalformedModel()).run(request_for("r-7-0007"))
kept = [r for r in tools.audits.values() if r.trace_id == run.trace_id]
moved = "yes" if tools.payments else "no"
print(
    f"| malformed call | `r-7-0007` | `refund.execute` | `{run.decision.reason}` "
    f"| {len(kept)} | {moved} |"
)
# endregion book:fnd.perception-tools.failures-01


# region book:fnd.perception-tools.conv-01
import numpy as np


def cross_correlate(x, w, b=0.0, stride=1, pad=0, dil=1):
    """Lay the kernel on, multiply, add; step by the stride."""
    x = np.pad(np.asarray(x, dtype=float), pad)
    w = np.asarray(w, dtype=float)
    span = dil * (len(w) - 1) + 1  # the width the kernel covers
    h_out = (x.shape[0] - span) // stride + 1
    w_out = (x.shape[1] - span) // stride + 1
    z = np.empty((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            r, c = i * stride, j * stride
            window = x[r : r + span : dil, c : c + span : dil]
            z[i, j] = b + np.sum(w * window)
    return z


def output_size(n, k, stride=1, pad=0, dil=1):
    return (n + 2 * pad - dil * (k - 1) - 1) // stride + 1


def parameters(k, c_in, c_out):
    return k * k * c_in * c_out + c_out  # weights, then biases


# endregion book:fnd.perception-tools.conv-01


# region book:fnd.perception-tools.conv-02 output
def show(z):
    return " ".join("[" + " ".join(f"{v:+.0f}" for v in row) + "]" for row in z)


X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
W = [[1, 0], [0, -1]]
print(f"cross-correlation of X with W:  {show(cross_correlate(X, W))}")
print(f"convolution (W flipped):        {show(cross_correlate(X, np.flip(W)))}")
line = np.zeros((5, 5))
line[1:4, 2] = 1
vertical = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
Z = cross_correlate(line, vertical, pad=1)
others = np.delete(Z, 2, axis=1)
print(f"vertical line, 'same' size: column 3 reads {Z[:, 2].astype(int).tolist()}")
print(f"vertical line, every other output: {int(np.abs(others).max())}")
print(f"224 pixels, k 3, stride 2, pad 1:        {output_size(224, 3, stride=2, pad=1)}")
print(f"224 pixels, k 3, stride 1, pad 1, dil 2: {output_size(224, 3, pad=1, dil=2)}")
print(f"3-by-3 kernels, 3 -> 32 channels:        {parameters(3, 3, 32):,} parameters")
# endregion book:fnd.perception-tools.conv-02


# region book:fnd.perception-tools.vision-01 output
def receptive_field(layers):
    """Layers as (k, stride, dil), first layer first; returns the field's side."""
    r, s = 1, 1
    for k, stride, dil in layers:
        r += (k - 1) * dil * s
        s *= stride
    return r


for label, layers in (
    ("three 3x3 layers, stride 1", [(3, 1, 1)] * 3),
    ("strides 2, 1, 1", [(3, 2, 1), (3, 1, 1), (3, 1, 1)]),
    ("dilations 1, 2, 1", [(3, 1, 1), (3, 1, 2), (3, 1, 1)]),
    ("dilations 1, 2, 4", [(3, 1, 1), (3, 1, 2), (3, 1, 4)]),
):
    print(f"receptive field, {label}: {receptive_field(layers)}")


def flagged_defective(prior, sensitivity, false_positive):
    hit = sensitivity * prior
    return hit / (hit + false_positive * (1 - prior))


for fp in (0.1, 0.3):
    share = flagged_defective(0.02, 0.9, fp)
    defects = round(10_000 * 0.02)
    flags = round(10_000 * (0.9 * 0.02 + fp * 0.98))
    caught = round(0.9 * defects)
    print(
        f"false-positive rate {fp}: Pr(defect | flag) = {share:.6f}; per 10,000 parts "
        f"{flags:,} flags, {caught} of them defects"
    )
# endregion book:fnd.perception-tools.vision-01


# region book:fnd.perception-tools.exercises-01 output
observed = {"order": -60, "shipment": -120, "policy": -3600}
paid_at = 1710 + 1
ages = {source: float(paid_at - t) for source, t in observed.items()}
print(f"exercise 5: ages at payment {ages}; {rules.freshness_problem(ages)}")
expiry = min(t + LIMITS[source] for source, t in observed.items())
print(f"exercise 5: evidence expired {expiry} s after the request")
X5, W5 = [[2, 0, 1], [1, 3, 0], [0, 1, 2]], [[1, -1], [0, 2]]
print(f"exercise 6: cross-correlation {show(cross_correlate(X5, W5))}")
print(f"exercise 6: convolution {show(cross_correlate(X5, np.flip(W5)))}")
print(f"exercise 6: output size {output_size(32, 5, pad=2)}; parameters {parameters(5, 3, 16):,}")
print(f"exercise 6: receptive field {receptive_field([(3, 2, 1), (3, 1, 1), (3, 1, 1)])}")
for fp in (0.1, 0.25):
    flags = round(10_000 * (0.9 * 0.02 + fp * 0.98))
    print(f"exercise 7(a): fp {fp}: {flagged_defective(0.02, 0.9, fp):.3f}; {flags:,} flags")

# Added exercise answers; the payment rows above retain their original bytes.
from demos.perception_tools_enrichment import (
    action_probabilities,
    intersection_over_union,
    parts_per_slice,
    wilson_interval,
)

low, high = wilson_interval(36, 40)
print(f"exercise 7(b): 36/40 = {36 / 40:.2f}; 95% Wilson [{low:.6f}, {high:.6f}]")
print(f"exercise 7(b): at 0.02 defects, expected parts {parts_per_slice(40, 0.02):,.0f}")
for offset in (10, 4):
    prediction = (10 + offset, 10 + offset, 50 + offset, 50 + offset)
    iou = intersection_over_union(prediction, (10, 10, 50, 50))
    print(f"exercise 8: offset {offset}; IoU {iou:.6f}; {'accept' if iou >= 0.5 else 'refuse'}")
for outcome, probability in action_probabilities(20, 0.95, 0.9).items():
    print(f"exercise 8: {outcome} {probability:.6f}")
tools, state = prepared(request_for("r-7-0007"))
missing = tools.call(execute(state, None), state)
first = tools.call(execute(state, "exercise-10"), state)
repeat = tools.call(execute(state, "exercise-10"), state)
conflict = tools.call(execute(state, "exercise-10", state.request.amount_cents + 1), state)
print(f"exercise 4: missing key -> {missing.reason}; repeat receipt {repeat == first}")
print(f"exercise 4: changed binding -> {conflict.reason}; payments {len(tools.payments)}")
stale_tools, stale_state = prepared(request_for("r-7-0002"))
stale_result = stale_tools.call(execute(stale_state, "exercise-10-stale"), stale_state)
print(f"exercise 4: gate -> {stale_result.reason}; payments {len(stale_tools.payments)}")
# endregion book:fnd.perception-tools.exercises-01


# region book:fnd.perception-tools.vision-02 output
from demos.perception_tools_enrichment import parts_per_slice, wilson_interval

print("dim-light slice; given defect rate 0.02; 95% Wilson (z=1.96)")
print("flagged/defects  sensitivity  Wilson interval         expected parts")
for flagged, defects in ((45, 50), (450, 500)):
    low, high = wilson_interval(flagged, defects)
    print(
        f"{flagged:>3}/{defects:<3}         {flagged / defects:.2f}"
        f"         [{low:.6f}, {high:.6f}]"
        f"     {parts_per_slice(defects, 0.02):>6,.0f}"
    )
# endregion book:fnd.perception-tools.vision-02


# region book:fnd.perception-tools.screen-01 output
from datetime import datetime

from demos.perception_tools_enrichment import (
    action_probabilities,
    box_overlap_and_union,
    intersection_over_union,
    locate_element,
    page_age_seconds,
    simulate_actions,
)

page = {
    "role": "page", "box": (0, 0, 640, 480),
    "captured_at": "2026-10-15T10:30:00+00:00",
    "children": [
        {"id": "updated", "box": (80, 10, 400, 30), "text": "2026-10-15T10:00:00+00:00"},
        {"id": "status", "box": (80, 40, 300, 60), "text": "In transit"},
        {"id": "refresh", "box": (10, 10, 50, 50), "text": "Refresh"},
    ],
}
used_at = datetime.fromisoformat("2026-10-15T10:30:01+00:00")
print(f"constructed carrier; tracking page only; {page['role']} box={page['box']}")
for node in page["children"]:
    print(f"  {node['id']}: box={node['box']} text={node['text']}")
print(f"capture {page['captured_at']}; use {used_at.isoformat()}")
print(f"page age from displayed update: {page_age_seconds(page, used_at):,.0f} s")
for offset in (10, 4):
    prediction = (10 + offset, 10 + offset, 50 + offset, 50 + offset)
    reference = page["children"][2]["box"]
    overlap, union = box_overlap_and_union(prediction, reference)
    iou = intersection_over_union(prediction, reference)
    try:
        locate_element(page, "refresh", prediction)
        verdict = "accept"
    except ValueError as error:
        if str(error) != "localization_refused":
            raise
        verdict = "refuse"
    print(
        f"offset {offset:>2}: overlap/union {overlap:g}/{union:g}; "
        f"IoU {iou:.6f} ({iou:.2f}); {verdict} at 0.5"
    )
actions, p_act, p_catch, runs, seed = 20, 0.95, 0.9, 10000, 7
print(f"{actions} independent actions; p_act={p_act}; p_catch={p_catch}")
print("one retry, checked too; detected wrong retry stops; no false alarms")
print("retry replaces a reversible effect; 'all right' describes final effects")
per_action = action_probabilities(1, p_act, p_catch)
print(
    f"one action: correct {per_action['checked_all_right']:.6f}; "
    f"silent {per_action['checked_silent_completed']:.6f}; "
    f"stop {per_action['checked_stopped']:.6f}"
)
exact = action_probabilities(actions, p_act, p_catch)
sampled = simulate_actions(actions, p_act, p_catch, runs=runs, seed=seed)
print(f"outcome                     exact     {runs:,} runs (seed {seed})")
for outcome, probability in exact.items():
    print(f"{outcome:<28}{probability:.6f}  {sampled[outcome]:>5} ({sampled[outcome] / runs:.4f})")
initial_miss = 1 - (1 - (1 - p_act) * (1 - p_catch)) ** actions
print(f"20 initial attempts only (no retries/stops): any silent error {initial_miss:.6f}")
# endregion book:fnd.perception-tools.screen-01


# region book:fnd.perception-tools.listing-01 output
import json
import textwrap
from pathlib import Path

from refund_agent.contracts import ToolArguments as ListingArguments

schema = ListingArguments.model_json_schema()
# Wrap only between encoder chunks, never inside a JSON string token.
schema_lines, current = [], ""
for chunk in json.JSONEncoder(separators=(",", ":")).iterencode(schema):
    if current and len(current) + len(chunk) > 79:
        schema_lines.append(current)
        current = ""
    current += chunk
schema_lines.append(current)
schema_text = "\n".join(schema_lines)
chapter = Path(__file__).resolve().parents[2] / "mainmatter/perception-tools.qmd"
contract_row = next(
    row for row in chapter.read_text(encoding="utf-8").splitlines()
    if row.startswith("| `refund.execute` |")
)
print("refund.execute: generated shared ToolArguments JSON Schema")
print(schema_text)
print("Contract row (effect/checks; refusals):")
print(textwrap.fill("; ".join(part.strip() for part in contract_row.strip("|").split("|")[1:]), 79))
# endregion book:fnd.perception-tools.listing-01
