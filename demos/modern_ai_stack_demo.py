"""Output for the architecture chapter: the order record's contract, four releases of the
order system checked against it, and what the rule-only baseline drafts from each.

The data are the default seed's (7). One order and one request take the values the chapter
gives: order o-0007 paid 4,200 cents for one item, with no prior refund, and request
r-7-0007 asks for 3,800 cents back. Nothing here changes the refund agent: every check is
the package's own `Order` record and `baseline` function.
"""

import sys
from pathlib import Path
from textwrap import fill

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refund_agent import contracts
from refund_agent.baseline import baseline, score_decisions
from refund_agent.contracts import State, ToolCall
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools

REQUEST = "r-7-0007"
ASKED = 3800


def excerpt(lines: list[str], first: str) -> list[str]:
    """Lines of contracts.py from the one starting with `first` to the end of its statement."""
    start = next(i for i, line in enumerate(lines) if line.startswith(first))
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " #)"):
        end += 1
    while not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


# region book:fnd.modern-ai-stack.contract-01 output
from demos.output_wrap import print_wrapped

source = Path(contracts.__file__).read_text(encoding="utf-8").splitlines()
record = excerpt(source, "class Record(")
# Keep the displayed assignment valid Python while fitting the book measure.
record[1] = record[1].replace("{", "{\n        ").replace(
    ", ", ",\n        "
).replace("}", ",\n    }")
validator = record.index('    @field_validator("*")')
cents = source.index(next(line for line in source if line.startswith("Cents =")))
blocks = [
    source[cents - 1 : cents + 1],
    record[:2] + ["    ..."] + record[validator:],
    excerpt(source, "class Item("),
    excerpt(source, "class Order("),
]
print_wrapped("# refund_agent/contracts.py, excerpt")
print_wrapped("\n\n".join("\n".join(block) for block in blocks))
# endregion book:fnd.modern-ai-stack.contract-01


# region book:fnd.modern-ai-stack.invariant-01 output
import json

from refund_agent.contracts import Order
from refund_agent.data.generate import generate


def items_total(order: dict) -> int:
    return sum(
        item["quantity"] * item["unit_cents"]
        for item in order["items"]
    )


ORDER = "o-0007"
PAID = 4200
data = generate(seed=7)
generated = next(o for o in data.orders if o.order_id == ORDER)
# The order system's record, with the chapter's given values.
sent = {
    **generated.model_dump(mode="json"),
    "items": [
        {
            "name": "physical desk lamp",
            "quantity": 1,
            "unit_cents": PAID,
        }
    ],
    "paid_cents": PAID,
    "prior_refunds": [],
}
orders = [o.model_dump(mode="json") for o in data.orders]
holds = sum(o["paid_cents"] == items_total(o) for o in orders)
print(
    "orders whose paid_cents equals the items' total: "
    f"{holds} of {len(orders)}"
)
dollars = {**sent, "paid_cents": sent["paid_cents"] // 100}
print(
    f"the record in dollars: paid_cents {dollars['paid_cents']}, "
    f"items {items_total(dollars)}"
)
all_dollars = {
    **dollars,
    "items": [
        {**item, "unit_cents": item["unit_cents"] // 100}
        for item in dollars["items"]
    ],
}
checked = Order.model_validate_json(json.dumps(all_dollars))
total = items_total(checked.model_dump(mode="json"))
verdict = (
    "accepted" if checked.paid_cents == total else "rejected"
)
print(
    f"all amounts in dollars: paid_cents {checked.paid_cents}, "
    f"items {total}, {verdict}"
)
# endregion book:fnd.modern-ai-stack.invariant-01


def problems(error: ValidationError) -> None:
    for problem in error.errors():
        field = ".".join(map(str, problem["loc"])) or "record"
        print(fill(problem["msg"], width=66, initial_indent=f"  {field}: ",
                   subsequent_indent="    "))


# region book:fnd.modern-ai-stack.compat-01 output
from refund_agent.contracts import Order

renamed = dict(sent)
renamed["amount_paid_cents"] = renamed.pop("paid_cents")
releases = {
    "renamed field": renamed,
    "added field": {**sent, "currency": "USD"},
    "naive time": {**sent, "observed_at": "2026-03-22T15:07:27"},
    "in dollars": {**sent, "paid_cents": 42},
}
for change, record in releases.items():
    try:
        order = Order.model_validate_json(json.dumps(record))
    except ValidationError as error:
        print(f"{change}: rejected")
        problems(error)
    else:
        print(f"{change}: accepted as {order.paid_cents} cents")
# endregion book:fnd.modern-ai-stack.compat-01


def load(order: dict, asked: int = ASKED) -> Dataset:
    """The default data with o-0007 replaced as sent and r-7-0007 asking `asked` cents."""
    raw = data.model_dump(mode="json")
    raw["orders"] = [order if o["order_id"] == ORDER else o for o in raw["orders"]]
    raw["requests"] = [
        {**r, "amount_cents": asked} if r["request_id"] == REQUEST else r for r in raw["requests"]
    ]
    return Dataset.model_validate_json(json.dumps(raw))


def drafts(changes: dict[str, dict], asked: int = ASKED) -> None:
    """What the baseline drafts for r-7-0007 when the order system sends each record."""
    for change, order in changes.items():
        try:
            changed = load(order, asked)
        except ValidationError as error:
            print(f"{change}: data load stopped, 0 of {len(data.requests)} drafted")
            problems(error)
            continue
        request = next(r for r in changed.requests if r.request_id == REQUEST)
        state = State(request=request, trace_id="balance")
        Tools(changed).call(ToolCall(name="order.lookup"), state)
        assert state.order is not None
        balance = state.order.paid_cents - state.refunded_cents
        decision = baseline(changed, [request])[0]
        print(
            f"{change}: balance {balance} cents; {decision.outcome.value}, "
            f"tier {decision.tier.value}, {decision.reason}"
        )
        print(f"  cites {decision.policy_citation}")


# region book:fnd.modern-ai-stack.drafts-01 output
request = next(r for r in data.requests if r.request_id == REQUEST)
print(f"request {REQUEST}: {ASKED} cents, {request.reason}, order {ORDER}")
drafts({"as sent": sent, "renamed field": renamed, "in dollars": releases["in dollars"]})
# endregion book:fnd.modern-ai-stack.drafts-01


# region book:fnd.modern-ai-stack.scorecard-01 output
public = set(data.public_scenarios)
requests = [r for r in data.requests if r.request_id in public]
labels = [d for d in data.correct if d.request_id in public]
# The release sends whole dollars, with fractional dollars discarded.
in_dollars = data.model_copy(
    update={
        "orders": [o.model_copy(update={"paid_cents": o.paid_cents // 100}) for o in data.orders]
    }
)
# A Markdown table, so that the book can set it as a table in the exercises.
print("| Forty public scenarios | Agree | Approve | Deny | Refer | Paid |")
print("|---|---:|---:|---:|---:|---:|")
for name, version in (("Before the release", data), ("After the release", in_dollars)):
    try:
        card = score_decisions(baseline(version, requests), labels)
    except ValidationError:  # an order record that refuses the release stops the run
        print(f"| {name} | data load stopped | | | | |")
        continue
    cells = (f"{card.correct}/{card.total}", card.approve, card.deny, card.refer, card.executed)
    print(f"| {name} | " + " | ".join(map(str, cells)) + " |")
# endregion book:fnd.modern-ai-stack.scorecard-01


# region book:fnd.modern-ai-stack.exercise-03 output
exercise = {
    "paid as text": {**sent, "paid_cents": "4200"},
    "payment pending": {**sent, "paid_cents": 0},
    "local time": {**sent, "observed_at": "2026-03-22T11:07:27-04:00"},
    "hundredths": {**sent, "paid_cents": 420000},
}
accepted = {}
for change, record in exercise.items():
    try:
        Order.model_validate_json(json.dumps(record))
    except ValidationError as error:
        print(f"{change}: rejected")
        problems(error)
    else:
        accepted[change] = record
drafts(accepted)
print("the same request at 5000 cents:")
drafts({"as sent": sent, "hundredths": exercise["hundredths"]}, asked=5000)
# endregion book:fnd.modern-ai-stack.exercise-03
