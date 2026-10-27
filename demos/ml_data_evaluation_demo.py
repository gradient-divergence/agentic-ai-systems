"""Output for the chapter on ML data and evaluation.

`counts-01` computes the measures from counts, precision at another base rate, the Wald
and Wilson intervals and the cases a stated width needs; `counts-02` prints the figures
the chapter quotes beside them. `label-01` relabels the generated requests under the
latest policy version and counts the decisions that change. `windows-01` cuts the
time-safe split (`refund_agent/split.py`, whose `time_split` is region `split-01`) and
counts each window. The golden window is then frozen, before any check or model.
`leak-01` works on the development requests only, the training and validation windows:
it builds the inputs a score may know at decision time, plants a copy of the label and
a field from the agent's audit records, runs a value check ported from the course
showcases and a time check, and scores a small classifier in four settings, the
time-ordered ones on July. `golden-01` prints the frozen set's manifest, reads its rows
through one logged consultation, scores the baseline and the classifier on them, and
prints the whole access log; `golden-02` prints the per-tier table. `exercises-01`
prints the keys to exercises 3 to 6; the chapter does not include it.

Between the freeze and that consultation no check or model uses a golden request or
label: the script's `windows` and `labels` hold the development cases only. The label
count and the window counts before the freeze describe the labels the set is built
from; they score nothing.

The agent and the baseline run on each request with a dataset that holds only that
request's order, shipment and policy retrieval, and the policy versions
(`one_request_data`). They decide as on the full dataset, which takes minutes. Importing
this module runs nothing; the work and the printing happen when the file runs as a
script. Nothing here changes the refund agent.
"""

# ruff: noqa: E402

# region book:fnd.ml-data-and-evaluation.counts-01 output
import math

Z = 1.96  # the two-sided 95% cutoff of the standard normal


def measures(tp, fp, fn, tn):
    return {
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "F1": 2 * tp / (2 * tp + fp + fn),
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "false-positive rate": fp / (fp + tn),
    }


def precision_at(base_rate, recall, fp_rate):
    """Bayes' rule: the share of flags that are positive."""
    hits = recall * base_rate
    return hits / (hits + fp_rate * (1 - base_rate))


def wald_interval(k, n, z=Z):
    rate = k / n
    half = z * math.sqrt(rate * (1 - rate) / n)
    return max(0.0, rate - half), min(1.0, rate + half)


def wilson_interval(k, n, z=Z):
    rate, t = k / n, z * z / n
    center = (rate + t / 2) / (1 + t)
    spread = math.sqrt(rate * (1 - rate) / n + t / (4 * n))
    half = z * spread / (1 + t)
    return max(0.0, center - half), min(1.0, center + half)


def cases_needed(rate, half_width, z=Z):
    return math.ceil(z * z * rate * (1 - rate) / half_width**2)


if __name__ == "__main__":
    bank = measures(tp=18, fp=6, fn=2, tn=74)
    for name, value in bank.items():
        print(f"{name}: {value:.4f}")
    recall, fpr = bank["recall"], bank["false-positive rate"]
    for base in (0.005, 0.01, 0.2, 0.5):
        p = precision_at(base, recall, fpr)
        print(f"precision at a base rate of {base}: {p:.4f}")
    for k, n in ((10, 100), (0, 15), (2, 20)):
        wald = "[{:.4f}, {:.4f}]".format(*wald_interval(k, n))
        wilson = "[{:.4f}, {:.4f}]".format(*wilson_interval(k, n))
        print(f"{k} in {n}: Wald {wald}, Wilson {wilson}")
    need = cases_needed(0.1, 0.05)
    print(f"cases for a half-width of 0.05 at 0.1: {need}")
    mean = need / 0.01
    print(f"payments at a 1% base rate, on average: {mean:,.0f}")
# endregion book:fnd.ml-data-and-evaluation.counts-01


# region book:fnd.ml-data-and-evaluation.counts-02 output
# Figures the chapter quotes beside counts-01; not included.
def coverage(n, rate, interval=wilson_interval):
    """The actual coverage of an interval at n cases and a true rate: the chance, summed
    over the binomial counts, that the interval computed from the sample holds the rate.
    A nominal 95% interval aims at 0.95 through the normal approximation."""
    held = 0.0
    for k in range(n + 1):
        low, high = interval(k, n)
        if low <= rate <= high:
            held += math.comb(n, k) * rate**k * (1 - rate) ** (n - k)
    return held


if __name__ == "__main__":
    tp, fp, fn, tn = 0, 0, 1, 99  # one positive in 100, never flagged
    print(f"never flags: accuracy {(tp + tn) / 100:.2f}, recall {tp / (tp + fn):.1f}, ", end="")
    print(f"precision undefined ({tp + fp} flags)")
    print(f"one-sided 95% bound, 0 errors in 15: {1 - 0.05 ** (1 / 15):.4f}")
    print(f"Wilson upper end at 0 in 15, z^2/(n + z^2): {Z * Z / (15 + Z * Z):.4f}")
    t = Z * Z / 100  # the worked Wilson interval for 10 errors in 100, step by step
    center = (0.1 + t / 2) / (1 + t)
    half = Z * math.sqrt(0.1 * 0.9 / 100 + t / 400) / (1 + t)
    print(f"10 in 100: z^2/n {t:.4f}, center {center:.4f}, half-width {half:.4f}")
    print(
        f"standard deviation of the estimate at 0.1 and n = 100: {math.sqrt(0.1 * 0.9 / 100):.2f}"
    )
    low, high = wilson_interval(2, 20)  # the bank's 2 misses among its 20 fraud cases
    print(
        f"recall 18 of 20, Wilson on the miss rate turned around: [{1 - high:.4f}, {1 - low:.4f}]"
    )
    shown = ", ".join(f"{coverage(15, rate):.4f} at a rate of {rate}" for rate in (0.01, 0.2))
    print(f"Wilson's actual coverage at n = 15: {shown}")
# endregion book:fnd.ml-data-and-evaluation.counts-02


import random
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from refund_agent.agent import Agent
from refund_agent.baseline import baseline, score_decisions
from refund_agent.contracts import Decision, Order, Outcome, Request, Shipment
from refund_agent.data.generate import Dataset, generate
from refund_agent.labels import LABEL_POLICY, cases, relabel
from refund_agent.split import CUTS, WINDOWS, freeze, time_split
from refund_agent.tools import Tools

INPUTS = (
    "amount_cents",
    "damaged",
    "changed_mind",
    "not_delivered",
    "balance_cents",
    "delivered",
    "days_since_delivery",
    "return_days",
    "cap_cents",
)
# Where each input is read from, and whether the decision itself writes that record.
SOURCES = {
    "amount_cents": ("request", False),
    "damaged": ("request", False),
    "changed_mind": ("request", False),
    "not_delivered": ("request", False),
    "balance_cents": ("order", False),
    "delivered": ("shipment", False),
    "days_since_delivery": ("shipment", False),
    "return_days": ("policy", False),
    "cap_cents": ("policy", False),
    "paid_amount_cents": ("audit record", True),
    "label_copy": ("label", True),
}


def one_request_data(data: Dataset, request: Request) -> Dataset:
    """Only what one request's run reads; an ambiguous reference keeps every order it names."""
    orders = [o for o in data.orders if request.order_reference in (o.order_id, o.reference)]
    ids = {o.order_id for o in orders}
    return Dataset(
        orders=orders,
        requests=[request],
        shipments=[s for s in data.shipments if s.order_id in ids],
        policies=data.policies,
        retrievals=[s for s in data.retrievals if s.request_id == request.request_id],
        correct=[d for d in data.correct if d.request_id == request.request_id],
        reviewers=data.reviewers,
    )


def audit_fields(data: Dataset, requests: list[Request]) -> dict[str, tuple[int, float]]:
    """Run the agent on each request; return the cents its audit records say were paid,
    and how many seconds after the request the latest of those records is stamped."""
    found = {}
    for request in requests:
        tools = Tools(one_request_data(data, request))
        Agent(tools).run(request)
        records = list(tools.audits.values())
        paid = max(r.paid_amount_cents for r in records)
        after = max((r.at - request.at).total_seconds() for r in records)
        found[request.request_id] = (paid, after)
    return found


def baseline_decisions(data: Dataset, requests: list[Request]) -> list[Decision]:
    return [baseline(one_request_data(data, r))[0] for r in requests]


def payment_forbidden(label: Decision) -> int:
    """1 if the policy forbids paying the request: the label denies it, or refers it
    for a reason other than the approval its tier requires."""
    refers_for_cause = label.outcome == Outcome.REFER and label.reason != "approval_required"
    return int(label.outcome == Outcome.DENY or refers_for_cause)


def decision_time_inputs(data: Dataset, requests: list[Request] | None = None) -> list[dict]:
    """What a score may know at each request's time: the request, the order and shipment
    as the agent's reads return them, and the policy version then in force. One row for
    each of `requests`, or for every request of the dataset."""
    by_reference = defaultdict(list)
    for order in data.orders:
        by_reference[order.order_id].append(order)
        if order.reference != order.order_id:
            by_reference[order.reference].append(order)
    shipments = {s.order_id: s for s in data.shipments}
    rows = []
    for r in data.requests if requests is None else requests:
        found = by_reference[r.order_reference]
        order = found[0] if len(found) == 1 else None  # an ambiguous reference reads nothing
        events = shipments[order.order_id].events if order else []
        delivered = [e.at for e in events if e.status == "delivered" and e.at <= r.at]
        refunded = sum(p.amount_cents for p in order.prior_refunds if p.at <= r.at) if order else 0
        policy = data.policy_at(r.at)
        rows.append(
            {
                "amount_cents": r.amount_cents,
                "damaged": int(r.reason == "damaged"),
                "changed_mind": int(r.reason == "changed_mind"),
                "not_delivered": int(r.reason == "not_delivered"),
                "balance_cents": order.paid_cents - refunded if order else math.nan,
                "delivered": int(bool(delivered)),
                "days_since_delivery": (r.at - max(delivered)).total_seconds() / 86400
                if delivered
                else math.nan,
                "return_days": policy.return_days,
                "cap_cents": policy.cap_cents,
            }
        )
    return rows


def record_seconds_after(
    data: Dataset, name: str, case: tuple[Request, Order, Shipment], audit_after: float
) -> float:
    """Seconds between a request's time and the time stamped on the record behind `name`."""
    request, order, shipment = case
    record = SOURCES[name][0]
    if record == "order":
        stamped = order.observed_at
    elif record == "shipment":
        stamped = shipment.observed_at
    elif record == "policy":
        stamped = data.policy_at(request.at).effective_at
    elif record == "audit record":
        return audit_after
    else:  # the request, and a label taken as of the request's time
        stamped = request.at
    return (stamped - request.at).total_seconds()


def value_check(values: list[float], labels: list[int]) -> str:
    """Ported from the course showcases (`ml_core.leakage._exact_target_leakage`): a column
    equal to the label in 98% of rows, or correlated with it at 0.995 or more, is flagged.
    It compares values with the label and never reads a time."""
    same = sum(str(v) == str(y) for v, y in zip(values, labels)) / len(labels)
    if same >= 0.98:
        return "flags"
    pairs = np.array([(v, y) for v, y in zip(values, labels) if not math.isnan(v)], dtype=float)
    if len(pairs) > 2 and pairs[:, 0].std() > 0:
        if abs(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) >= 0.995:
            return "flags"
    return "passes"


def time_check(name: str, latest_after: float) -> str:
    """Flag an input whose record the decision writes, or that is stamped after it."""
    if SOURCES[name][1]:
        return "flags: decision writes it"
    return "flags: stamped after" if latest_after > 0 else "passes"


def fit_and_predict(train: list[dict], train_y: list[int], test: list[dict], columns):
    """A decision tree on the columns; missing values take the median of the training
    rows, fitted on those rows only."""
    medians = {name: float(np.nanmedian([row[name] for row in train])) for name in columns}

    def matrix(rows):
        return np.array(
            [[medians[n] if math.isnan(row[n]) else row[n] for n in columns] for row in rows],
            dtype=float,
        )

    model = DecisionTreeClassifier(random_state=0).fit(matrix(train), train_y)
    return model.predict(matrix(test)).tolist()


def agreeing(predicted: list[int], truth: list[int]) -> str:
    """How many of the flags equal their labels, as "k of n"."""
    return f"{sum(p == t for p, t in zip(predicted, truth))} of {len(truth)}"


def interval_text(k: int, n: int, complement: bool = False) -> str:
    """A rate's Wilson interval; for recall and precision, the complement of the rate."""
    low, high = wilson_interval(k, n)
    if complement:
        return f"{1 - k / n:.2f} ({1 - high:.2f} to {1 - low:.2f})"
    return f"{low:.3f} to {high:.3f}"


def tier_counts(read, drafts: list[Decision], flagged: dict[str, int]) -> list[tuple]:
    """For each tier and for all rows of a consultation: the requests, the baseline's
    errors, and the classifier's true positives, false positives and false negatives on
    requests whose payment the policy forbids. An error is counted by the stage 0 scorer
    (`refund_agent.baseline.score_decisions`): a draft is right when its outcome, amount
    and tier are the label's and, for a denial, its policy citation too."""
    counted = []
    for tier in (1, 2, 3, None):
        chosen = [
            (draft, r, label)
            for draft, (r, label) in zip(drafts, read)
            if tier is None or int(label.tier) == tier
        ]
        card = score_decisions([d for d, _, _ in chosen], [label for _, _, label in chosen])
        y = [payment_forbidden(label) for _, _, label in chosen]
        p = [flagged[r.request_id] for _, r, _ in chosen]
        tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
        fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
        fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
        counted.append((tier or "All", len(chosen), card.total - card.correct, tp, fp, fn))
    return counted


if __name__ == "__main__":
    data = generate(7)
    labels = {d.request_id: d for d in data.correct}
    gap = timedelta(days=LABEL_POLICY.window_days)
    windows = time_split(data.requests, CUTS, gap)


# region book:fnd.ml-data-and-evaluation.label-01 output
if __name__ == "__main__":
    latest = data.policies[-1].version
    relabeled = {d.request_id: d for d in relabel(data, latest)}
    changed = [
        r for r in data.requests if relabeled[r.request_id].outcome != labels[r.request_id].outcome
    ]
    policy = LABEL_POLICY
    print(f"label policy {policy.version}, effective {policy.effective_at.date()}, ", end="")
    print("final when decided" if policy.window_days == 0 else f"{policy.window_days} days")
    print(f"under version {latest}: {len(changed)} of {len(data.requests)} decisions change")
    tiers = Counter(int(labels[r.request_id].tier) for r in changed)
    print("by tier: " + ", ".join(f"{t}: {tiers[t]}" for t in sorted(tiers)))
    versions = Counter(data.policy_at(r.at).version for r in changed)
    shown = [f"{p.version} {versions[p.version]}" for p in data.policies]
    print("by version in force: " + ", ".join(shown[:2]) + ",")
    print("  " + ", ".join(shown[2:]))
    moves = Counter(
        f"{labels[r.request_id].outcome.value} to {relabeled[r.request_id].outcome.value}"
        for r in changed
    )
    print("changes: " + ", ".join(f"{move} {n}" for move, n in sorted(moves.items())))
    ambiguous = next(r for r in data.requests if r.failure == "ambiguous_order")
    outcome = labels[ambiguous.request_id].outcome.value
    print(f"{ambiguous.request_id}, {ambiguous.failure}: {outcome} under both")
    first = next(
        r
        for r in sorted(changed, key=lambda r: r.at)
        if relabeled[r.request_id].outcome == Outcome.APPROVE
    )
    shipment = next(s for q, _, s in cases(data) if q is first)
    delivered = max(e.at for e in shipment.events if e.status == "delivered")
    print(
        f"example: {first.request_id}, decided {first.at.date()}, "
        f"{(first.at - delivered).days} days after delivery"
    )
    print(
        f"  window then {data.policy_at(first.at).return_days} days, "
        f"now {data.policies[-1].return_days} days: "
        f"{labels[first.request_id].outcome.value} becomes "
        f"{relabeled[first.request_id].outcome.value}"
    )
# endregion book:fnd.ml-data-and-evaluation.label-01


# region book:fnd.ml-data-and-evaluation.windows-01 output
if __name__ == "__main__":
    print(f"gap: {gap.days} days, since a label is final when decided")
    print(f"{'window':<11}{'requests':>9}  {'dates':<26}tiers 1, 2, 3")
    for name, members in windows.items():
        tiers = Counter(int(labels[r.request_id].tier) for r in members)
        dates = f"{members[0].at.date()} to {members[-1].at.date()}"
        counts = ", ".join(str(tiers[t]) for t in (1, 2, 3))
        print(f"{name:<11}{len(members):>9}  {dates:<26}{counts}")
    print(f"{'policy version':<16}" + "".join(f"{name:>12}" for name in WINDOWS))
    for policy in data.policies:
        counts = [
            sum(data.policy_at(r.at).version == policy.version for r in windows[name])
            for name in WINDOWS
        ]
        print(f"{policy.version:<16}" + "".join(f"{n:>12}" for n in counts))
    no_delivery = sum(not any(e.status == "delivered" for e in s.events) for s in data.shipments)
    print(f"shipments with no delivered event: {no_delivery} of {len(data.shipments)}")
    unseen = sum(
        not any(e.status == "delivered" and e.at <= r.at for e in s.events)
        for r, o, s in cases(data)
    )
    print(f"requests with none when they arrived: {unseen} of {len(data.requests)}")
# endregion book:fnd.ml-data-and-evaluation.windows-01


if __name__ == "__main__":
    # Freeze the golden window now, before any check or model. From this line on,
    # `windows` and `labels` hold the development cases only, the training and
    # validation windows, and a golden request or label is reached only through
    # `golden.consult`, which logs the read.
    golden = freeze(data, windows.pop("golden"), LABEL_POLICY)
    development = windows["training"] + windows["validation"]
    labels = {r.request_id: labels[r.request_id] for r in development}


# region book:fnd.ml-data-and-evaluation.leak-01 output
if __name__ == "__main__":
    rows = decision_time_inputs(data, development)
    audits = audit_fields(data, development)
    truth = [payment_forbidden(labels[r.request_id]) for r in development]
    for row, r, y in zip(rows, development, truth):
        row["paid_amount_cents"] = audits[r.request_id][0]
        row["label_copy"] = y
    print(f"checks on the {len(development)} requests before {CUTS[1].date()}")
    print(f"{'input':<20}{'record':<13}{'value':<8}time")
    triples = {case[0].request_id: case for case in cases(data) if case[0].request_id in labels}
    for name in (*INPUTS, "paid_amount_cents", "label_copy"):
        values = [row[name] for row in rows]
        after = max(
            record_seconds_after(data, name, triples[r.request_id], audits[r.request_id][1])
            for r in development
        )
        print(
            f"{name:<20}{SOURCES[name][0]:<13}{value_check(values, truth):<8}"
            f"{time_check(name, after)}"
        )
    stamped = max(after for _, after in audits.values())
    print(f"audit records, latest seconds after their request: {stamped:g}")

    # A random split of the development requests that holds out as many as July has,
    # then the time split, tested on July, with and without the audit field; "at
    # decision" blanks the field, since no audit record exists before the decision.
    order = list(range(len(rows)))
    random.Random(7).shuffle(order)
    train_i = list(range(len(windows["training"])))
    july_i = list(range(len(windows["training"]), len(rows)))
    held, kept = order[: len(july_i)], order[len(july_i) :]
    leaky = (*INPUTS, "paid_amount_cents")
    tier_of = [int(labels[r.request_id].tier) for r in development]

    def scored(train, test, columns, blank=False):
        tests = [{**rows[i], "paid_amount_cents": 0} if blank else rows[i] for i in test]
        predicted = fit_and_predict(
            [rows[i] for i in train], [truth[i] for i in train], tests, columns
        )
        tier_1 = [j for j, i in enumerate(test) if tier_of[i] == 1]
        return (
            agreeing(predicted, [truth[i] for i in test]),
            agreeing([predicted[j] for j in tier_1], [truth[test[j]] for j in tier_1]),
        )

    settings = [
        ("shuffled split, random rows, field as written", scored(kept, held, leaky)),
        ("time split, July rows, field as written", scored(train_i, july_i, leaky)),
        ("same model, July rows, field at decision", scored(train_i, july_i, leaky, True)),
        ("time split, July rows, no audit field", scored(train_i, july_i, INPUTS)),
    ]
    print("held-out requests where flag and forbidden-payment label agree")
    print(f"{'':<46}{'all':>10}{'tier 1':>10}")
    for name, (everyone, first_tier) in settings:
        print(f"{name:<46}{everyone:>10}{first_tier:>10}")
# endregion book:fnd.ml-data-and-evaluation.leak-01


# region book:fnd.ml-data-and-evaluation.golden-01 output
if __name__ == "__main__":
    for key, value in golden.manifest.items():
        print(f"{key}: {value[:16] if key == 'sha256' else value}")
    read = golden.consult("stage 1 report", "baseline and classifier")
    print("access log, every entry:")
    for entry in golden.log:
        print(f"  {entry['entry']}: {entry['set']}, {entry['reader']}, {entry['purpose']}")
    asked = [request for request, _ in read]
    drafts = baseline_decisions(data, asked)
    card = score_decisions(drafts, [label for _, label in read])
    print("baseline, scored on outcome, amount, tier and a denial's citation:")
    print(f"  errors: {card.total - card.correct} of {card.total}")
    # The time-split classifier of leak-01, without the audit field, flags the rows
    # this consultation returned.
    flags = fit_and_predict(
        [rows[i] for i in train_i],
        [truth[i] for i in train_i],
        decision_time_inputs(data, asked),
        INPUTS,
    )
    flagged = dict(zip((r.request_id for r in asked), flags))
    missed = [(r, label) for r, label in read if flagged[r.request_id] != payment_forbidden(label)]
    for kind, flag in (("misses", 0), ("false-positive flags", 1)):
        group = [(r, label) for r, label in missed if flagged[r.request_id] == flag]
        print(f"classifier {kind}: {len(group)}")
        for reason in sorted({label.reason for _, label in group}):
            same = [(r, label) for r, label in group if label.reason == reason]
            tiers = sorted({str(int(label.tier)) for _, label in same})
            versions = Counter(data.policy_at(r.at).version for r, _ in same)
            dated = ", ".join(f"{v} {n}" for v, n in sorted(versions.items()))
            where = ("tiers " if len(tiers) > 1 else "tier ") + " and ".join(tiers)
            print(f"  label {reason}, {where}: {len(same)}")
            print(f"    by version in force: {dated}")
    tagged = [(r.request_id, r.failure) for r, _ in missed if r.failure]
    print(f"flags that disagree on an injected failure: {tagged or 'none'}")
# endregion book:fnd.ml-data-and-evaluation.golden-01


# region book:fnd.ml-data-and-evaluation.golden-02 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped(
        "| Tier | N | Errors | Error, 95% | Forbidden "
        "| Recall, 95% | Prec., 95% |"
    )
    print_wrapped("|---|---|---|---|---|---|---|")
    for tier, n, errors, tp, fp, fn in tier_counts(read, drafts, flagged):
        print_wrapped(
            f"| {tier} | {n} | {errors} | {interval_text(errors, n)} | {tp + fn} "
            f"| {interval_text(fn, tp + fn, complement=True)} "
            f"| {interval_text(fp, tp + fp, complement=True)} |"
        )
# endregion book:fnd.ml-data-and-evaluation.golden-02


# region book:fnd.ml-data-and-evaluation.trace-01 output
if __name__ == "__main__":
    # First policy-payable tier 3 referral by time and ID, from the already logged read.
    trace_request, trace_label = min(
        ((request, label) for request, label in read
         if request.amount_cents > 50000 and label.reason == "approval_required"),
        key=lambda pair: (pair[0].at, pair[0].request_id),
    )
    trace_approver = data.policy_at(trace_request.at).named_approver
    print("tier 3 contrast: 600-case policy dataset, independent of the paid history")
    print(
        f"golden request {trace_request.request_id}: {trace_request.at:%Y-%m-%d}; "
        f"amount ${trace_request.amount_cents / 100:.2f}; "
        f"decision label {trace_label.outcome.value}, tier {int(trace_label.tier)}, "
        f"reason {trace_label.reason}; named approver {trace_approver}"
    )
    print("agent prepares the packet only; named approver decides, denials included")
# endregion book:fnd.ml-data-and-evaluation.trace-01


# region book:fnd.ml-data-and-evaluation.exercises-01 output
# Keys to exercises 3 to 6; not included.
if __name__ == "__main__":
    given = measures(tp=30, fp=10, fn=15, tn=945)
    print("exercise 3: " + ", ".join(f"{name} {value:.4f}" for name, value in given.items()))
    base = precision_at(0.005, given["recall"], given["false-positive rate"])
    print(f"exercise 3, precision at a base rate of 0.005: {base:.4f}")
    for k in (3, 0):
        low, high = wilson_interval(k, 50)
        print(f"exercise 4, Wilson for {k} in 50: [{low:.4f}, {high:.4f}]")
    need = cases_needed(0.2, 0.05)
    print(f"exercise 4, positives for recall near 0.8 within 0.05: {need}")
    print(f"exercise 4, requests at a base rate of 0.02, on average: {need / 0.02:,.0f}")

    # Exercise 5. The manifest's median is the one over every request, golden ones
    # included: that is its planted error. It is reproduced here from the requests'
    # inputs alone; no label is read and nothing is scored.
    every = float(np.nanmedian([row["days_since_delivery"] for row in decision_time_inputs(data)]))
    days = [row["days_since_delivery"] for row in rows]
    training_only = float(np.nanmedian([days[i] for i in train_i]))
    print(f"exercise 5, median days since delivery: all 600 {every}, training {training_only}")

    # The same tree, with missing days filled by each median in turn, compared on
    # July. The golden rows stay unread: this key spends no consultation on them.
    def predictions(median, test):
        fill = {n: float(np.nanmedian([rows[i][n] for i in train_i])) for n in INPUTS}
        fill["days_since_delivery"] = median

        def matrix(index):
            return np.array(
                [
                    [fill[n] if math.isnan(rows[i][n]) else rows[i][n] for n in INPUTS]
                    for i in index
                ],
                dtype=float,
            )

        model = DecisionTreeClassifier(random_state=0)
        model.fit(matrix(train_i), [truth[i] for i in train_i])
        return model.predict(matrix(test)).tolist()

    changed = sum(
        a != b for a, b in zip(predictions(every, july_i), predictions(training_only, july_i))
    )
    print(
        f"exercise 5, July predictions that differ between the medians: {changed} of {len(july_i)}"
    )

    # Exercise 6: the September slice of the golden set. It names its parent and
    # shares the parent's log, so this read is the set's second and no fresh start.
    september = [r for r in data.requests if r.at >= data.policies[-1].effective_at]
    piece = freeze(data, september, LABEL_POLICY, name="golden-1-september", parent=golden)
    shown = {k: (v[:16] if "sha256" in k else v) for k, v in piece.manifest.items()}
    print(f"exercise 6, manifest: {shown}")
    rows_2 = piece.consult("exercise 6", "baseline and classifier, September slice")
    early = sum(r.at < CUTS[0] + gap for r, _ in rows_2)
    print(
        f"exercise 6, requests before the cutoff plus the gap: {early}, "
        f"fingerprints recompute: slice {piece.intact()}, parent {golden.intact()}"
    )
    for entry in golden.log:
        print(
            f"exercise 6, access log, entry {entry['entry']} of {len(golden.log)}: "
            f"{entry['set']}, {entry['reader']}, {entry['purpose']}"
        )
    drafts_2 = baseline_decisions(data, [r for r, _ in rows_2])
    for tier, n, errors, tp, fp, fn in tier_counts(rows_2, drafts_2, flagged):
        print(
            f"exercise 6, tier {str(tier).lower()}: {n} requests, baseline errors {errors} "
            f"({interval_text(errors, n)}), forbidden {tp + fn}, "
            f"recall {interval_text(fn, tp + fn, complement=True)}, "
            f"precision {interval_text(fp, tp + fp, complement=True)}"
        )
# endregion book:fnd.ml-data-and-evaluation.exercises-01
