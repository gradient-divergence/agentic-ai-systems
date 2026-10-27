"""Listings and output for the chapter on supervised learning and calibrated risk.

The chapter trains a risk score on the generated refund history
(`refund_agent/data/history.py`), whose label is an outcome: a paid claim found false
within 30 days. Every window is cut by the day a label became final, so nothing fitted
here uses a label that was still open on 1 August 2026, the day the score starts work.

`split-01` cuts the windows. Straight after it, before any model is fit, the two held-out
months are frozen with `refund_agent.split.freeze`, the machinery of the chapter on ML
data and evaluation: August and September each become a set with a manifest (its dates,
its count, the outcome label's policy version and a fingerprint of its requests and
outcomes) and an access log. From that line on a held-out request or outcome is reached
only through `consulted`, which logs the read.

`fit-01` fits a logistic regression on the training window and prints its log loss
beside the base rate's. `boost-01` builds three one-split trees by hand, each fit to what
the trees before it left. `fit-02` prints the boosted trees' losses beside the logistic
regression's, and `biasvar-01` the estimated bias and variance of three models at two
fixed requests, over training sets whose outcomes are drawn again from the generator's
planted risk. `decide-01` computes the expected cost of each action for three incident
tickets and for a refund, and the edges of the band where a review is cheapest.
`calibrate-01` bins a score, measures its calibration error on the validation window,
and fits a two-parameter map there.

`august-01` prints August's manifest and makes its one consultation, after every fit.
`record-01`, `record-02`, `gate-01` and `sweep-01` print what that read found: the four
scores at the tier threshold, the proposed score's calibration by score bin and by
amount, three drafts against the placeholder rule, and the cost of every cut-off.
`shift-01` corrects a probability for a changed base rate and for a class weight, and
`shift-02` makes the one consultation of September, the month after the policy change.
`numbers-01` prints the arithmetic the chapter quotes, `keys-01` the keys to its
exercises, and `figures-01` the data its three figures are drawn from; the chapter
includes none of those three.

The six regions the chapter prints as listings (`split-01`, `fit-01`, `boost-01`,
`decide-01`, `calibrate-01`, `shift-01`) run in that order on their own: the one name
they take from outside themselves, `boosted`, is imported where it is first used. They
read no held-out request.

"The proposed score" is the boosted trees followed by their recalibration map. It is
demonstrated here offline, on requests the policy paid. Nothing here changes the refund
agent, whose `risk_score` is still the text rule in `refund_agent/rules.py`.

`numbers-01` also holds two diagnostics the chapter describes in words and does not
quote. The first is the class-weighted map's fitted slope and intercept beside the 1 and
minus log 26.6 that the weighted-loss result predicts at the loss's minimum. The second
is why September's odds correction cannot be exact: doubling a risk multiplies its odds
by 2.04 at 0.02 and by 2.43 at 0.15, where the correction applies one multiplier, 1.50.

Importing this module defines the functions and runs nothing: the history is generated,
the models are fitted and every line is printed only when the file runs as a script.
Run with `--write-fixture` to rewrite the report that exercise 7 hands the reader.
"""

# ruff: noqa: E402

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import os

    for pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[pool] = "1"  # one thread, so every run prints the same digits


# region book:fnd.supervised-learning.split-01 output
from datetime import datetime, timezone

from refund_agent.data.history import generate_history

CUTS = [datetime(2026, month, 1, tzinfo=timezone.utc) for month in (7, 8, 9)]
FIT_BY, GO_LIVE, SHIFT = CUTS  # 1 July, 1 August and 1 September
WINDOWS = ("training", "validation", "open", "test", "shifted")


def windows(paid):
    """Place each paid request by the day its 30-day label became final."""
    window = {name: [] for name in WINDOWS}
    for r in paid:
        if r.at >= SHIFT:
            window["shifted"].append(
                r
            )  # decided under September's policy version
        elif r.at >= GO_LIVE:
            window["test"].append(r)  # held out
        elif r.label_final_at <= FIT_BY:
            window["training"].append(r)
        elif r.label_final_at <= GO_LIVE:
            window["validation"].append(r)
        else:
            window["open"].append(r)  # no label yet when the score starts work
    return window


if __name__ == "__main__":
    history = generate_history(seed=7)
    window = windows([r for r in history if r.paid])
    print(f"{len(history):,} requests, {sum(r.paid for r in history):,} paid")
    for name, rows in window.items():
        span = f"{rows[0].at:%Y-%m-%d} to {rows[-1].at:%Y-%m-%d}"
        print(f"{name:10} {len(rows):6,}  decided {span}")
# endregion book:fnd.supervised-learning.split-01


from types import SimpleNamespace
from typing import Callable, Sequence, cast

from refund_agent import contracts
from refund_agent.data.generate import Dataset
from refund_agent.data.history import FEATURE_COLUMNS, START_AT, HistoryRow
from refund_agent.labels import LabelPolicy
from refund_agent.rules import tier_for
from refund_agent.split import freeze

# The outcome label, in the fields of the label policy of the chapter on ML data and
# evaluation. It is a second policy beside that chapter's decision label, not a new
# version of it.
OUTCOME_POLICY = LabelPolicy(
    version="outcome-1",
    effective_at=START_AT,
    records="whether a paid refund's claim was found false within 30 days: 1 if so, else 0",
    as_of="30 days after the payment",
    window_days=30,
    exclusions=("requests that were not paid",),
    owner="the merchant's owner of the refund decision",
)


class Outcome(contracts.Record):
    """One paid request's outcome label, as a record `freeze` can fingerprint."""

    request_id: str
    tier: int
    found_false_30d: int
    label_final_at: datetime


class Ledger:
    """What `freeze` asks of a data set, read off paid history rows: a label for each
    request, and the policy version in force at a request's time. It stands in for a
    `Dataset`, of which `freeze` reads `correct` and `policy_at` and nothing else."""

    def __init__(self, rows):
        self.correct = [
            Outcome(
                request_id=r.request_id,
                tier=int(tier_for(r.amount_cents)),
                found_false_30d=r.found_false_30d,
                label_final_at=r.label_final_at,
            )
            for r in rows
        ]
        self._version = {r.at: r.policy_version for r in rows}

    def policy_at(self, at):
        return SimpleNamespace(version=self._version[at])


def frozen(rows, name):
    """Freeze paid history rows as that chapter froze its golden set: a manifest with
    the rows' fingerprint and the label policy's version, and an empty access log."""
    return freeze(cast(Dataset, Ledger(rows)), rows, OUTCOME_POLICY, name=name)


if __name__ == "__main__":
    # Freeze the two held-out months now, before any model is fit. From this line on
    # `window` holds only requests a fit may read, and an August or September request
    # or outcome is reached only through `consulted`, which logs the read.
    august = frozen(window.pop("test"), "history-august")
    september = frozen(window.pop("shifted"), "history-september")
    del history  # the full list still held both months


# region book:fnd.supervised-learning.fit-01 output
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from refund_agent.model.history_baseline import feature_matrix


def arrays(rows):
    """Decision-time inputs, the outcome label, and the amount in dollars."""
    X = np.nan_to_num(feature_matrix(rows))  # no delivery scan counts as age 0
    y = np.array([r.found_false_30d for r in rows])
    return X, y, np.array([r.amount_cents for r in rows]) / 100


def log_loss(y, p):
    """The mean of -[y log p + (1 - y) log(1 - p)], in nats."""
    p = np.asarray(p)
    observed = np.where(np.asarray(y) == 1, p, 1 - p)
    with np.errstate(divide="ignore"):
        return float(-np.log(observed).mean())


def risk(model, X):
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    Xt, yt, _ = arrays(window["training"])
    Xv, yv, _ = arrays(window["validation"])
    print(f"wrong payments: training {yt.sum()} ({yt.mean():.4f}),", end=" ")
    print(f"validation {yv.sum()} ({yv.mean():.4f})")
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    )
    models = {
        "base rate": DummyClassifier(strategy="prior").fit(Xt, yt),
        "logistic regression": logistic.fit(Xt, yt),
    }
    print(f"{'log loss':20} training  validation")
    for name, model in models.items():
        losses = log_loss(yt, risk(model, Xt)), log_loss(yv, risk(model, Xv))
        print(f"{name:20}   {losses[0]:.4f}      {losses[1]:.4f}")
# endregion book:fnd.supervised-learning.fit-01


# region book:fnd.supervised-learning.boost-01 output
def stump(xs, residual):
    """The one-split tree with the least squared error on the residual."""
    best = None
    for cut in sorted(set(xs))[:-1]:
        left = [r for x, r in zip(xs, residual) if x <= cut]
        right = [r for x, r in zip(xs, residual) if x > cut]
        a, b = sum(left) / len(left), sum(right) / len(right)
        error = (
            sum((r - a) ** 2 for r in left)
            + sum((r - b) ** 2 for r in right)
        )
        if best is None or error < best[0]:
            best = (error, cut, a, b)
    _, cut, a, b = best
    return cut, lambda x: a if x <= cut else b


if __name__ == "__main__":
    xs, ys = [1, 2, 3, 4, 5, 6], [0, 0, 4, 4, 10, 10]
    F = [0.0] * len(xs)  # the ensemble starts at zero
    for m in (1, 2, 3):
        residual = [y - f for y, f in zip(ys, F)]  # (1)
        cut, tree = stump(xs, residual)  # (2)
        F = [f + 1.0 * tree(x) for x, f in zip(xs, F)]  # learning rate 1  # (3)
        error = sum((y - f) ** 2 for y, f in zip(ys, F))
        print(f"tree {m}: residuals {residual}, split after {cut}")
        print(f"  predictions {F}, squared error {error:g}")
# endregion book:fnd.supervised-learning.boost-01


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def boosted(class_weight=None):
    """A hundred trees of at most seven leaves, each fit to what the earlier ones left."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        class_weight=class_weight,
        random_state=7,
    )


# region book:fnd.supervised-learning.fit-02 output
# The boosted trees beside the logistic regression; output only.
if __name__ == "__main__":
    models["boosted trees"] = boosted().fit(Xt, yt)

    def losses_of(name):
        """Training and validation log loss, as printed: to four decimals."""
        model = models[name]
        return [round(log_loss(y, risk(model, X)), 4) for X, y in ((Xt, yt), (Xv, yv))]

    floor = losses_of("base rate")
    print(f"{'log loss':20} training  validation  below the base rate's")
    for name in ("logistic regression", "boosted trees"):
        losses = losses_of(name)
        print(f"{name:20}   {losses[0]:.4f}      {losses[1]:.4f}", end="  ")
        print(f"{floor[0] - losses[0]:.4f} and {floor[1] - losses[1]:.4f}")
# endregion book:fnd.supervised-learning.fit-02


import json
import math

from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier

from refund_agent.data.history import _false_probability as planted_risk

REVIEW, WRONG_DENIAL = 4.00, 20.00  # illustrative dollars, the owner's to replace
TIER_THRESHOLD = 0.080  # the review cost over the largest tier 1 amount, to three places


def consulted(golden, purpose):
    """Read a frozen month: log the consultation, then hand over the rows' arrays."""
    read = golden.consult("stage 1 report", purpose)
    return arrays([row for row, _ in read])


def print_record(golden):
    """A frozen set's manifest, with a short fingerprint, and every entry of its log."""
    for key, value in golden.manifest.items():
        print(f"{key}: {value[:16] if key == 'sha256' else value}")
    print("access log, every entry:")
    for entry in golden.log:
        print(f"  {entry['entry']}: {entry['set']}, {entry['reader']}, {entry['purpose']}")


def wilson_interval(k, n, z=1.96):
    """The Wilson interval of the chapter on ML data and evaluation."""
    rate, t = k / n, z * z / n
    center = (rate + t / 2) / (1 + t)
    half = z * math.sqrt(rate * (1 - rate) / n + t / (4 * n)) / (1 + t)
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_means(values, draws=2000, seed=7) -> np.ndarray:
    """Draw whole requests with replacement; both drafts travel in their difference."""
    values = np.asarray(values, dtype=float)
    index = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    return values[index].mean(axis=1)


def bootstrap(values, draws=2000, seed=7):
    """The mean of per-request values, with a 95% percentile interval."""
    values = np.asarray(values, dtype=float)
    low, high = np.quantile(bootstrap_means(values, draws, seed), [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def realized_cost(y, amount, pay):
    """A paid request costs its amount if it was a wrong payment; a referral, a review."""
    return np.where(pay, y * amount, REVIEW)


def example_request(rows, reason, low=3900, high=4100):
    """The first request with this reason, a prior refund, about $40, 5 to 8 days old."""
    for r in rows:
        if r.reason == reason and r.prior_refund_count > 0 and low <= r.amount_cents <= high:
            if r.delivery_age_days is not None and 5 <= r.delivery_age_days <= 8:
                return r
    raise ValueError(f"no example request for {reason}")


def bias_variance(makers, pool, points, sets=200, size=5000, seed=7):
    """Estimates of the squared bias and the variance of each model's prediction at
    fixed requests.

    Each training set draws `size` requests from the pool with replacement and draws
    their outcomes anew from the generator's planted risk, which no model sees. The
    mean over `sets` predictions has variance Var / sets of its own, so that much is
    taken off its squared distance from the true risk, and a result below zero is set
    to zero. A squared bias of zero therefore says the gap is no larger than `sets`
    training sets can resolve, not that the model has no bias.
    """
    X_pool = arrays(pool)[0]
    mu_pool = np.array([planted_risk(r) for r in pool])
    X_points = arrays(points)[0]
    mu = np.array([planted_risk(r) for r in points])
    rng = np.random.default_rng(seed)
    predictions = {name: [] for name in makers}
    for _ in range(sets):
        index = rng.integers(0, len(pool), size)
        outcomes = (rng.random(size) < mu_pool[index]).astype(int)
        for name, make in makers.items():
            predictions[name].append(risk(make().fit(X_pool[index], outcomes), X_points))
    rows = []
    for name, drawn in predictions.items():
        drawn = np.array(drawn)
        mean, variance = drawn.mean(axis=0), drawn.var(axis=0, ddof=1)
        bias2 = np.maximum((mean - mu) ** 2 - variance / sets, 0.0)
        rows.append((name, mu, mean, bias2, variance, drawn))
    return rows


# region book:fnd.supervised-learning.biasvar-01 output
from demos.output_wrap import print_wrapped

# Estimated bias and variance of three models at two fixed requests; output only.
if __name__ == "__main__":
    points = [example_request(window["training"], "damaged"),
              example_request(window["training"], "changed_mind")]  # fmt: skip
    makers = {
        "Logistic regression": lambda: make_pipeline(StandardScaler(),
                                                     LogisticRegression(max_iter=1000)),
        "One deep tree": lambda: DecisionTreeClassifier(random_state=7),
        "Boosted trees": boosted,
    }  # fmt: skip
    bv_rows = bias_variance(makers, window["training"], points)
    labels = ("Damage claim", "Change of mind")  # both customers have a prior refund
    head = "| Request | True risk | Model | Mean "
    print_wrapped(head + "| Sq. bias | Variance |")
    print_wrapped("|---|---|---|---|---|---|")
    for j, label in enumerate(labels):
        for i, (name, mu, mean, bias2, variance, _) in enumerate(bv_rows):
            lead = f"| {label} | {mu[j]:.3f} |" if i == 0 else "| | |"
            print_wrapped(f"{lead} {name} | {mean[j]:.3f} | {bias2[j]:.4f} | {variance[j]:.4f} |")
# endregion book:fnd.supervised-learning.biasvar-01


# region book:fnd.supervised-learning.decide-01 output
def expected_costs(probabilities, costs):
    """Each action's cost: the probability-weighted sum of its column."""
    actions = range(len(costs[0]))
    return [
        sum(p * row[a] for p, row in zip(probabilities, costs))
        for a in actions
    ]


def band(review, amount, wrong_denial):
    """The risks between which a review costs less than paying or denying."""
    return review / amount, 1 - review / wrong_denial


QUEUES = ("standard", "priority", "escalation")
ROUTER = [
    [0, 2, 5],
    [4, 0, 1],
    [10, 3, 0],
]  # row: the true queue; column: the route

if __name__ == "__main__":
    print(
        f"{'ticket':18}{'standard':>11}{'priority':>11}{'escalation':>11}  "
        f"route"
    )
    for ticket in ([0.90, 0.06, 0.04], [0.75, 0.05, 0.20], [0.21, 0.00, 0.79]):
        costs = expected_costs(ticket, ROUTER)
        cells = "".join(f"{cost:11.2f}" for cost in costs)
        print(f"{str(ticket):18}{cells}  {QUEUES[costs.index(min(costs))]}")
    review, wrong_denial = 4.00, 20.00  # illustrative dollars
    print("refund at risk 0.12:  pay  refer   deny  review cheapest between")
    for amount in (49.99, 40.00, 25.00, 10.00, 5.00):
        refund = [
            [0, review, wrong_denial],
            [amount, review, 0],
        ]  # pay, refer, deny
        pay, refer, deny = expected_costs([0.88, 0.12], refund)
        low, high = band(review, amount, wrong_denial)
        print(
            f"${amount:<19.2f} {pay:5.2f}  {refer:5.2f}  {deny:5.2f}", end="  "
        )
        print(f"{low:.3f} and {high:.3f}")
# endregion book:fnd.supervised-learning.decide-01


# region book:fnd.supervised-learning.calibrate-01 output
from sklearn.metrics import roc_auc_score

from demos.supervised_learning_demo import boosted  # the hundred small trees


def logit(p):
    return np.log(p / (1 - p))


def reliability(y, p, bins=5):
    """Equal-count score bins: requests, mean prediction, wrong payments."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    index = np.searchsorted(edges[1:-1], p, side="right")
    groups = [index == k for k in range(bins) if (index == k).any()]
    return [
        (int(g.sum()), float(p[g].mean()), int(y[g].sum())) for g in groups
    ]


def calibration_error(table):
    """The mean gap between a bin's observed rate and its mean prediction."""
    total = sum(n for n, _, _ in table)
    return sum(abs(wrong - n * mean) for n, mean, wrong in table) / total


def platt(p, y):
    """Fit sigmoid(w1 * logit(p) + w0) by log loss; return the map."""
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(
        logit(p).reshape(-1, 1), y
    )
    w1, w0 = float(fit.coef_[0, 0]), float(fit.intercept_[0])
    return (lambda q: 1 / (1 + np.exp(-(w1 * logit(q) + w0)))), w1, w0


if __name__ == "__main__":
    example = [(600, 0.05, 24), (300, 0.20, 72), (100, 0.60, 50)]
    print(f"worked example, three bins: {calibration_error(example):.3f}")
    models = {
        "boosted trees": boosted().fit(Xt, yt),
        "class-weighted": boosted(class_weight="balanced").fit(Xt, yt),
    }
    rate = f"{yv.sum()} wrong payments ({yv.mean():.4f})"
    print(f"validation: {len(yv)} requests, {rate}")
    print(f"{'score':15}   mean   error    AUC  map: slope, intercept")
    maps = {}
    for name, model in models.items():
        p = risk(model, Xv)
        maps[name], w1, w0 = platt(p, yv)
        error = calibration_error(reliability(yv, p))
        row = (
            f"{name:15} {p.mean():.4f}  {error:.4f}  "
            f"{roc_auc_score(yv, p):.3f}"
        )
        print(f"{row}  {w1:.3f}, {w0:.3f}")
# endregion book:fnd.supervised-learning.calibrate-01


def gate(score, amount=None, threshold=TIER_THRESHOLD):
    """True where the draft pays: below the tier threshold, or below a review's cost."""
    return score < threshold if amount is None else score * amount < REVIEW


def request_trace(
    rows: Sequence[HistoryRow],
    logistic: Pipeline,
    trees: HistGradientBoostingClassifier,
    mapping: Callable,
) -> tuple[HistoryRow, dict]:
    """Choose by decision-time fields and scores, without reading an outcome.

    In the development open pool, prefer a mapped tier referral, then a raw
    risk at most 0.1 (self-training's label 0), then an amount of at least $25.
    Relax amount first, then confidence, then referral if necessary. Within
    each preference take the first by decision time, breaking ties by identifier.
    """
    ordered = sorted(rows, key=lambda row: (row.at, row.request_id))
    if not ordered:
        raise ValueError("the development open pool is empty")
    X = np.nan_to_num(feature_matrix(ordered))
    raw = risk(trees, X)
    mapped = mapping(raw)
    referred, confident = ~gate(mapped), raw <= 0.1
    amount = np.array([row.amount_cents for row in ordered]) / 100
    for eligible in (
        referred & confident & (amount >= 25),
        referred & confident,
        referred,
        np.ones(len(ordered), dtype=bool),
    ):
        if eligible.any():
            index = int(np.flatnonzero(eligible)[0])
            break
    row = ordered[index]
    transformed = logistic[0].transform(X[index : index + 1])[0]
    coefficients = logistic[1].coef_[0]
    names = [
        column
        for name in FEATURE_COLUMNS
        for column in (("damaged", "changed_mind") if name == "reason" else (name,))
    ]
    inputs = [
        {
            "feature": name,
            "input": float(value),
            "coefficient": float(coefficient),
            "contribution": float(value * coefficient),
        }
        for name, value, coefficient in zip(names, transformed, coefficients)
    ]
    intercept = float(logistic[1].intercept_[0])
    return row, {
        "request_id": row.request_id,
        "decision_date": row.at.isoformat(),
        "policy_version": row.policy_version,
        "policy_decision": row.policy_decision,
        "raw_fields": {name: getattr(row, name) for name in FEATURE_COLUMNS},
        "logistic": {
            "inputs": inputs,
            "intercept": intercept,
            "log_odds": float(transformed @ coefficients + intercept),
            "probability": float(risk(logistic, X[index : index + 1])[0]),
        },
        "boosted": {"as_trained": float(raw[index]), "mapped": float(mapped[index])},
        "costs": {"pay": float(mapped[index] * amount[index]), "review": REVIEW},
        "drafts": {
            "tier_threshold": "pay" if gate(mapped[index]) else "refer",
            "review_cost": "pay" if gate(mapped[index], amount[index]) else "refer",
        },
        "permitted_action": "drafts only; a person executes",
    }


# region book:fnd.supervised-learning.trace-01 output
# One development request through the five chapters; the snapshot excludes its outcome.
if __name__ == "__main__":
    trace_logs_before = {
        name: list(golden.log) for name, golden in (("august", august), ("september", september))
    }
    trace_request, decision_snapshot = request_trace(
        window["open"], logistic, models["boosted trees"], maps["boosted trees"]
    )
    r, snapshot = trace_request, decision_snapshot
    trace_output = io.StringIO()
    with redirect_stdout(trace_output):
        print("Offline replay with fits fixed by 1 August; every input is from the decision time.")
        print("Selection: first by decision time in the July open pool; prefer mapped risk >= 0.080,")
        print("  raw risk <= 0.1, amount >= $25; relax amount first, then confidence, then referral.")
        print("Decision snapshot begins (decision-time fields only)")
        print(
            f"request {r.request_id}; decided {r.at.isoformat()}; tier {int(tier_for(r.amount_cents))}"
        )
        print(f"policy refund_cap_v4@{r.policy_version}; policy decision {r.policy_decision} (paid)")
        print(
            f"amount {r.amount_cents} cents (${r.amount_cents / 100:.2f}); reason {r.reason}; "
            f"days since delivery {r.delivery_age_days}; earlier refunds {r.prior_refund_count}"
        )
        print(
            f"original payment {r.paid_cents} cents; order evidence age {r.order_evidence_age_seconds}s; "
            f"shipment evidence age {r.shipment_evidence_age_seconds}s"
        )
        print("feature                            standardized input  coefficient  contribution")
        for item in snapshot["logistic"]["inputs"]:
            print(
                f"{item['feature']:34} {item['input']:18.6f} {item['coefficient']:12.6f} "
                f"{item['contribution']:13.6f}"
            )
        fitted = snapshot["logistic"]
        print(
            f"logistic intercept {fitted['intercept']:.6f}; log-odds {fitted['log_odds']:.6f}; "
            f"probability {fitted['probability']:.6f}"
        )
        print(
            f"boosted trees as trained {snapshot['boosted']['as_trained']:.6f}; "
            f"after validation Platt map {snapshot['boosted']['mapped']:.6f}"
        )
        print(f"expected cost: pay ${snapshot['costs']['pay']:.6f}; review ${REVIEW:.2f}")
        print(
            f"mapped drafts: tier threshold 0.080 -> {snapshot['drafts']['tier_threshold']}; "
            f"cost rule -> {snapshot['drafts']['review_cost']}; denied on a score: 0"
        )
        print(f"specification permits: {snapshot['permitted_action']}")
        print("Decision snapshot ends")
        assert r.label_final_at is not None and r.found_false_30d is not None
        print(
            f"Later observed outcome: final {r.label_final_at.isoformat()}; "
            f"found_false_30d {r.found_false_30d}; paid cost ${r.found_false_30d * r.amount_cents / 100:.2f}"
        )
        probabilities = models["boosted trees"].predict_proba(np.nan_to_num(feature_matrix([r])))[0]
        accepted = bool(probabilities.max() >= 0.9)
        trace_self_training = {
            "accepted_round": 1 if accepted else None,
            "pseudo_label": int(models["boosted trees"].classes_[probabilities.argmax()])
            if accepted
            else None,
        }
        print(
            f"Self-training on the open pool {len(window['open'])}, confidence 0.9: "
            f"would accept in round {trace_self_training['accepted_round']}; "
            f"pseudo-label {trace_self_training['pseudo_label']}"
        )
        print("Only the paid outcome is observed; the referral outcome and cost were never recorded.")
        trace_selection = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "book_outputs"
                / "_fnd.automl-and-selection.record-full-01.txt"
            ).read_text()
        )
        if trace_selection["selected"] != "boosted trees with map":
            raise ValueError("the stored selection no longer retains the traced boosted baseline")
        print(
            f"Selection record keeps {trace_selection['selected']}; "
            f"score {snapshot['boosted']['mapped']:.6f}; offline tier draft {snapshot['drafts']['tier_threshold']}"
        )
        print(f"Promotion granted: {trace_selection['grants']}")
        print(f"Signer: {trace_selection['signer']}")
        trace_logs_after = {
            name: list(golden.log) for name, golden in (("august", august), ("september", september))
        }
        assert trace_logs_before == trace_logs_after
        print(
            f"Frozen access logs unchanged by trace: August {len(august.log)}, September {len(september.log)}"
        )
    trace_lines = trace_output.getvalue().splitlines(keepends=True)
    print("".join(trace_lines), end="")
# endregion book:fnd.supervised-learning.trace-01


# region book:fnd.supervised-learning.trace-02 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    for line in trace_lines[3:18]:
        if line.startswith("feature "):
            print_wrapped(
                f"{'feature':29} {'standardized input':>18} {'coefficient':>12} "
                f"{'contribution':>13}"
            )
        elif line[:1] != " " and len(line.split()) == 4:
            name, value, coefficient, contribution = line.split()
            print_wrapped(f"{name:29} {value:>18} {coefficient:>12} {contribution:>13}")
        else:
            print_wrapped(line, end="")
# endregion book:fnd.supervised-learning.trace-02


# region book:fnd.supervised-learning.trace-03 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped("".join(trace_lines[18:23]), end="")
# endregion book:fnd.supervised-learning.trace-03


def cut_off_sweep(y, amount, score, cut_offs):
    """Mean cost per request when expected loss at or above a cut-off is referred."""
    return [float(realized_cost(y, amount, score * amount < c).mean()) for c in cut_offs]


def describe_sweep(costs, cut_offs, slack=0.02):
    """The cheapest cut-off and its cost, the cut-offs within `slack`, the cost at a review's."""
    costs = np.array(costs)
    best = int(costs.argmin())
    near = np.array(cut_offs)[costs <= costs[best] + slack]
    at_review = float(costs[list(cut_offs).index(REVIEW)])
    return (
        float(cut_offs[best]),
        float(costs[best]),
        (float(near.min()), float(near.max())),
        at_review,
    )


# region book:fnd.supervised-learning.august-01 output
# August's frozen requests: the manifest, then the one consultation, made after every
# fit above; output only.
if __name__ == "__main__":
    X_test, y_test, a_test = consulted(august, "the proposed score's August record")
    print_record(august)
# endregion book:fnd.supervised-learning.august-01


# region book:fnd.supervised-learning.record-01 output
# Four scores on the requests that consultation returned; output only.
if __name__ == "__main__":
    scores = {}
    for name in ("boosted trees", "class-weighted"):
        scores[f"{name}, as trained"] = risk(models[name], X_test)
        scores[f"{name}, with its map"] = maps[name](scores[f"{name}, as trained"])
    low, high = wilson_interval(int(y_test.sum()), len(y_test))
    print(f"August: {len(y_test)} requests, {y_test.sum()} wrong payments", end=" ")
    print(f"({y_test.mean():.4f}, {low:.3f} to {high:.3f})")
    print(f"{'score':29}   mean   error  referred  cost")
    for name, p in scores.items():
        pay = gate(p)
        cost = realized_cost(y_test, a_test, pay).mean()
        error = calibration_error(reliability(y_test, p))
        print(f"{name:29} {p.mean():.4f}  {error:.4f}  {int((~pay).sum()):8d}  {cost:.2f}")
    ranks = [roc_auc_score(y_test, scores[f"{name}, as trained"]) for name in maps]
    print(f"ROC AUC: boosted trees {ranks[0]:.3f}, class-weighted {ranks[1]:.3f}")
# endregion book:fnd.supervised-learning.record-01


def by_amount(y, p, amount, cut=25.0):
    """The same requests cut at an amount: requests, mean prediction, wrong payments."""
    groups = ((f"Under \\${cut:.0f}", amount < cut), (f"\\${cut:.0f} and over", amount >= cut))
    return [(name, int(g.sum()), float(p[g].mean()), int(y[g].sum())) for name, g in groups]


def record_table(y, p, amount):
    """The calibration record as Markdown rows: five bins of equal count by score, then
    the same requests cut at an amount, each row with its Wilson interval."""
    lines = ["| Requests | Count | Mean score | Wrong | Rate, 95% interval |",
             "|---|---|---|---|---|"]  # fmt: skip
    rows = [(f"Score bin {k}", *row) for k, row in enumerate(reliability(y, p), start=1)]
    for name, n, mean, wrong in rows + by_amount(y, p, amount):
        low, high = wilson_interval(wrong, n)
        rate = f"{wrong / n:.3f} ({low:.3f} to {high:.3f})"
        lines.append(f"| {name} | {n} | {mean:.3f} | {wrong} | {rate} |")
    return lines


# region book:fnd.supervised-learning.record-02 output
from demos.output_wrap import print_wrapped

# The proposed score in August, by score bin and by amount; output only.
if __name__ == "__main__":
    proposed = scores["boosted trees, with its map"]
    print_wrapped("\n".join(record_table(y_test, proposed, a_test)))
# endregion book:fnd.supervised-learning.record-02


MINUS = "\u2212"  # a table sets a negative number with a minus sign, not a hyphen


def draft_row(name, y, amount, pay, placeholder_cost=None):
    """One draft's line: counts, the wrong payments it paid, its cost and its saving."""
    paid, wrong = int(pay.sum()), int(y[pay].sum())
    low, high = wilson_interval(wrong, paid)
    cost = realized_cost(y, amount, pay)
    mean, c_low, c_high = bootstrap(cost)
    row = f"| {name} | {paid} | {len(y) - paid} | {wrong} ({low:.3f} to {high:.3f}) "
    row += f"| {mean:.2f} ({c_low:.2f} to {c_high:.2f}) "
    if placeholder_cost is None:
        return row + "| |"
    saving, s_low, s_high = bootstrap(placeholder_cost - cost)
    cell = f"{saving:.2f} ({s_low:.2f} to {s_high:.2f})"
    return row + f"| {cell.replace('-', MINUS)} |"


# region book:fnd.supervised-learning.gate-01 output
from demos.output_wrap import print_wrapped

# August under three drafts, with the boosted score as trained; output only.
if __name__ == "__main__":
    trained = scores["boosted trees, as trained"]
    everyone = np.ones(len(y_test), dtype=bool)  # the placeholder's 0.1 is below its 0.2
    placeholder_cost = realized_cost(y_test, a_test, everyone)
    print_wrapped("| Draft | Paid | Referred | Wrong payments paid | Cost per request | Saving |")
    print_wrapped("|---|---|---|---|---|---|")
    print_wrapped(draft_row("Placeholder rule", y_test, a_test, everyone))
    by_tier, by_request = gate(trained), gate(trained, a_test)
    print_wrapped(draft_row("Tier threshold, 0.080", y_test, a_test, by_tier, placeholder_cost))
    print_wrapped(draft_row("Review-cost rule, \\$4", y_test, a_test, by_request, placeholder_cost))
# endregion book:fnd.supervised-learning.gate-01


# region book:fnd.supervised-learning.bootstrap-01 output
# Figure V4's actual paired distribution, from the arrays already consulted for gate-01.
if __name__ == "__main__":
    tier_cost = realized_cost(y_test, a_test, by_tier)
    paired_savings = placeholder_cost - tier_cost
    resampled_savings = bootstrap_means(paired_savings, draws=2000, seed=7)
    low, high = np.quantile(resampled_savings, [0.025, 0.975])
    print(
        "August tier threshold 0.080 versus placeholder; score as trained; paired requests; seed 7"
    )
    print(
        f"requests {len(paired_savings)}; resamples {len(resampled_savings)}; "
        f"mean saving ${paired_savings.mean():.6f}; 95% percentiles ${low:.6f} to ${high:.6f}"
    )
    print("resample mean_saving_usd")
    for index, saving in enumerate(resampled_savings, start=1):
        print(f"{index:04d} {saving:.9f}")
# endregion book:fnd.supervised-learning.bootstrap-01


CUT_OFFS = [c / 2 for c in range(1, 121)]  # 50 cents to $60


# region book:fnd.supervised-learning.sweep-01 output
# Every cut-off on expected loss, in August, for two scores; output only.
if __name__ == "__main__":
    sweeps = {name: cut_off_sweep(y_test, a_test, scores[name], CUT_OFFS)
              for name in ("boosted trees, with its map", "class-weighted, as trained",
                           "boosted trees, as trained", "class-weighted, with its map")}  # fmt: skip
    print("refer when score times amount reaches a cut-off: mean cost per request")
    print(f"{'score':29} cheapest  cost  within 0.02 of it  at 4.00")
    for name, costs in sweeps.items():
        best, cost, (low, high), at_review = describe_sweep(costs, CUT_OFFS)
        print(f"{name:29}   {best:5.2f}  {cost:.2f}  {low:5.2f} to {high:5.2f}", end="     ")
        print(f"{at_review:.2f}")
    # The tier's one threshold against the review-cost rule, for each form of the score.
    for name, p in (("proposed score", proposed), ("score as trained", trained)):
        tier, rule = gate(p), gate(p, a_test)
        print(f"{name}: the tier threshold refers {int((~tier).sum())},", end=" ")
        print(f"the review-cost rule {int((~rule).sum())}")
        print(f"  the threshold's extra reviews: {int(rule.sum() - tier.sum())},", end=" ")
        print(f"{int((rule & ~tier).sum())} of them of refunds the rule pays")
        costs = [realized_cost(y_test, a_test, pay) for pay in (tier, rule)]
        extra, e_low, e_high = bootstrap(costs[0] - costs[1])
        print(f"  cost per request {costs[0].mean():.2f} against {costs[1].mean():.2f}:", end=" ")
        print(f"the threshold costs {extra:.2f} more ({e_low:.2f} to {e_high:.2f})")
# endregion book:fnd.supervised-learning.sweep-01


# region book:fnd.supervised-learning.shift-01 output
def odds(p):
    return p / (1 - p)


def shift_base_rate(p, rate_then, rate_now):
    """Move a probability from one base rate to another, on the odds scale."""
    moved = odds(p) * odds(rate_now) / odds(rate_then)
    return moved / (1 + moved)


if __name__ == "__main__":
    low, high = (
        shift_base_rate(0.5, 0.20, 0.01),
        shift_base_rate(0.9, 0.20, 0.01),
    )
    print(f"bank, 20% to 1%: 0.5 becomes {low:.3f}, 0.9 becomes {high:.3f}")
    omega = (
        1 - yt.mean()
    ) / yt.mean()  # the balanced weight on a wrong payment
    for mu in (0.01, yt.mean()):
        learned = omega * odds(mu) / (1 + omega * odds(mu))
        back = shift_base_rate(learned, 0.5, yt.mean())
        line = (
            f"weight {omega:.1f}: a true {mu:.4f} is learned as {learned:.3f}"
        )
        print(f"{line}; undone, {back:.4f}")
# endregion book:fnd.supervised-learning.shift-01


# region book:fnd.supervised-learning.shift-02 output
# September's frozen requests, read once; output only.
if __name__ == "__main__":
    X_shift, y_shift, a_shift = consulted(september, "the proposed score after the policy change")
    rate = f"{y_shift.sum()} wrong payments ({y_shift.mean():.4f})"
    print(f"September: {len(y_shift)} requests, {rate}")
    proposed_s = maps["boosted trees"](risk(models["boosted trees"], X_shift))
    # from the rate where the map was fit to the rate now
    moved = shift_base_rate(proposed_s, yv.mean(), y_shift.mean())
    print(f"{'score':22}   mean   error  top bin, predicted and observed")
    for name, p in (("proposed score", proposed_s), ("after the correction", moved)):
        table = reliability(y_shift, p)
        n, top, wrong = table[-1]
        print(f"{name:22} {p.mean():.4f}  {calibration_error(table):.4f}", end="  ")
        print(f"{top:.3f} and {wrong / n:.3f}")
# endregion book:fnd.supervised-learning.shift-02


def isotonic(p, y):
    """A map that assumes only that a higher score means no lower a risk."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y).predict


def recalibration_report(fit_on, judged_on, p_fit, y_fit, p_judged, y_judged):
    """What an agent would file after recalibrating a score with an isotonic map."""
    mapped = isotonic(p_fit, y_fit)(p_judged)
    table = reliability(y_judged, mapped)
    return {
        "score": "boosted trees, class-weighted",
        "map": "isotonic regression",
        "map_fit_on": fit_on,
        "judged_on": judged_on,
        "bins": [
            {
                "requests": n,
                "mean_prediction": round(mean, 4),
                "wrong_payments": wrong,
                "observed_rate": round(wrong / n, 4),
            }
            for n, mean, wrong in table
        ],  # fmt: skip
        "calibration_error": round(calibration_error(table), 4),
        "summary": "Calibrated after isotonic recalibration; ready for the tier 1 threshold.",
    }


FIXTURE = Path(__file__).resolve().parent / "data" / "supervised_learning_recalibration_07.json"
VALIDATION_WINDOW = {"decided": "2026-06-01 to 2026-07-01", "requests": 988}
AUGUST_WINDOW = {"decided": "2026-08-01 to 2026-08-31", "requests": 999}


# region book:fnd.supervised-learning.numbers-01 output
# Arithmetic the chapter quotes beside its listings; not included.
if __name__ == "__main__":
    print(f"opening: 0.15 x $49.99 = ${0.15 * 49.99:.2f}; 0.30 x $5 = ${0.30 * 5:.2f}")
    print(f"sigmoid(-1.99) = {1 / (1 + math.exp(1.99)):.4f}; exp(0.7) = {math.exp(0.7):.4f}")
    print(f"log loss of 0.9: right {-math.log(0.9):.4f}, wrong {-math.log(0.1):.4f}")
    print(f"a wrong payment scored 0.12: p - y = {0.12 - 1:.2f}")
    x, weights, bias = np.array([1, 0.5]), np.array([0.7, 0.62]), -3.0
    log_odds = float(weights @ x + bias)
    probability = 1 / (1 + math.exp(-log_odds))
    error = probability - 1
    gradient = error * x
    print(
        f"constructed request: log-odds {log_odds:.4f}; probability {probability:.4f}; "
        f"wrong-payment loss {-math.log(probability):.2f}; "
        f"sound-claim loss {-math.log1p(-probability):.4f}"
    )
    print(
        f"constructed wrong payment: error {error:.2f}; "
        f"gradient ({gradient[0]:.2f}, {gradient[1]:.2f}); bias gradient {error:.2f}"
    )
    updated_weights, updated_bias = weights - 0.1 * gradient, bias - 0.1 * error
    updated_probability = 1 / (1 + math.exp(-float(updated_weights @ x + updated_bias)))
    print(
        f"step 0.1: weights ({updated_weights[0]:.3f}, {updated_weights[1]:.3f}); "
        f"bias {updated_bias:.3f}; probability {updated_probability:.3f}; "
        f"loss {-math.log(updated_probability):.2f}"
    )
    savings = np.array([0, 0, 0, -4, 36])
    resample = np.array([1, 4, 4, 2, 5])  # request numbers in the text are one-based
    resampled = savings[resample - 1]
    print(f"constructed savings (0, 0, 0, -4, 36): mean ${savings.mean():.2f}")
    print(
        "resample requests (1, 4, 4, 2, 5): "
        f"savings {tuple(int(value) for value in resampled)}; mean ${resampled.mean():.2f}"
    )
    missed = ((len(savings) - 1) / len(savings)) ** len(savings)
    print(f"chance of missing request 5: (4/5)^5 = {missed:.4f} ({missed:.2f})")
    mu_check, drawn = 0.2, np.array([0.1, 0.3, 0.2, 0.4])
    noise, bias2, variance = mu_check * (1 - mu_check), (drawn.mean() - mu_check) ** 2, drawn.var()
    direct = np.mean(mu_check * (1 - drawn) ** 2 + (1 - mu_check) * drawn**2)
    print(f"bias-variance check: noise {noise:.4f} + squared bias {bias2:.4f}", end=" ")
    print(f"+ variance {variance:.4f} = {noise + bias2 + variance:.4f}; direct {direct:.4f}")
    for j, label in enumerate(labels):
        mu_j = float(bv_rows[0][1][j])
        print(f"{label}: true risk {mu_j:.4f}, noise {mu_j * (1 - mu_j):.4f},", end=" ")
        print(f"amount ${points[j].amount_cents / 100:.2f}, {points[j].delivery_age_days} days")
    deep = bv_rows[1][5]  # the deep tree's predictions, one row for each training set
    whole = [int(np.isin(deep[:, j], (0.0, 1.0)).sum()) for j in range(len(labels))]
    print(f"one deep tree: a prediction of exactly 0 or 1 in {whole[0]} and {whole[1]}", end=" ")
    print(
        f"of {len(deep)} training sets; at the change of mind, {int((deep[:, 1] == 0).sum())} are 0"
    )
    print("estimated squared bias: the squared gap, less variance / 200, floored at zero")
    for name, truth, mean_p, printed, spread, sets_drawn in bv_rows:
        for j, label in enumerate(labels):
            gap, less = float(mean_p[j] - truth[j]), float(spread[j] / len(sets_drawn))
            print(f"  {label}, {name}: gap {gap:+.4f}, squared {gap * gap:.6f},", end=" ")
            print(f"less {less:.6f}, printed {printed[j]:.4f}")
    print(f"tier 1: $4 / $49.99 = {REVIEW / 49.99:.4f}; $4 / $50 = {REVIEW / 50:.4f}")
    zero_one = expected_costs(
        [0.75, 0.05, 0.20], [[int(actual != route) for route in QUEUES] for actual in QUEUES]
    )
    cells = ", ".join(f"{queue} {cost:.2f}" for queue, cost in zip(QUEUES, zero_one))
    print(f"ticket 2 under 0-1 costs: {cells}, route {QUEUES[zero_one.index(min(zero_one))]}")
    # Calibrated over the pool and wrong at each amount: what the cost rule cannot use.
    dear, cheap, pooled = 0.12, 0.0, (0.12 + 0.0) / 2
    print(f"pooled calibration: equal numbers of $49.99 requests at a true {dear:.2f} and", end=" ")
    print(f"$5 requests at {cheap:.2f}, all scored {pooled:.2f}, their pooled rate")
    print(
        f"  the rule sees {pooled:.2f} x $49.99 = ${pooled * 49.99:.2f}, under ${REVIEW:.0f},",
        end=" ",
    )
    print(f"and pays; the true expected loss is ${dear * 49.99:.2f}")
    print(f"balanced weight on a wrong payment: {omega:.2f}; minus its log: {-math.log(omega):.3f}")
    for name in ("boosted trees", "class-weighted"):
        raw, mapped = scores[f"{name}, as trained"], scores[f"{name}, with its map"]
        print(f"{name}: August log loss {log_loss(y_test, raw):.4f},", end=" ")
        print(f"with its map {log_loss(y_test, mapped):.4f}, AUC {roc_auc_score(y_test, raw):.3f}")
    print(
        f"base rate's log loss in August: {log_loss(y_test, np.full(len(y_test), yt.mean())):.4f}"
    )
    # ROC AUC counted pair by pair, on the validation window: a tie counts as half.
    p_v = risk(models["boosted trees"], Xv)
    print(f"validation, boosted trees, {int(yv.sum()) * int((1 - yv).sum()):,} pairs", end=" ")
    print("of a wrong and a sound payment:")
    for name, s in (("the score", p_v), ("the score to two decimals", np.round(p_v, 2))):
        gaps = s[yv == 1][:, None] - s[yv == 0][None, :]
        won, tied = float((gaps > 0).mean()), float((gaps == 0).mean())
        print(f"  {name}: the wrong one is higher in {won:.4f}, level in {tied:.4f};", end=" ")
        print(f"{won:.4f} + {tied:.4f} / 2 = {won + tied / 2:.4f}, AUC {roc_auc_score(yv, s):.4f}")
    low, high = wilson_interval(int(yv.sum()), len(yv))
    print(f"validation rate {yv.mean():.4f}, 95% interval {low:.3f} to {high:.3f}")
    for month, y_m, p_m in (("August", y_test, proposed), ("September", y_shift, proposed_s)):
        pay = gate(p_m)
        paid, wrong = int(pay.sum()), int(y_m[pay].sum())
        low, high = wilson_interval(wrong, paid)
        print(f"{month}, proposed score at the tier threshold: paid {paid} of {len(y_m)},", end=" ")
        print(f"wrong {wrong} ({wrong / paid:.4f}, {low:.3f} to {high:.3f});", end=" ")
        print(f"among them predicted {p_m[pay].mean():.4f}")
    low, high = wilson_interval(int(y_shift.sum()), len(y_shift))
    print(f"September rate {y_shift.mean():.4f}, 95% interval {low:.3f} to {high:.3f}")
    print("September, mean cost per request:")
    for name, pay in (
        ("placeholder", np.ones(len(y_shift), dtype=bool)),
        ("proposed score, tier threshold", gate(proposed_s)),
        ("corrected score, tier threshold", gate(moved)),
        ("proposed score, review-cost rule", gate(proposed_s, a_shift)),
        ("corrected score, review-cost rule", gate(moved, a_shift)),
    ):
        mean, c_low, c_high = bootstrap(realized_cost(y_shift, a_shift, pay))
        print(f"  {name}: paid {int(pay.sum())}, wrong {int(y_shift[pay].sum())},", end=" ")
        print(f"cost {mean:.2f} ({c_low:.2f} to {c_high:.2f})")
    extra = realized_cost(y_shift, a_shift, gate(moved)) - realized_cost(
        y_shift, a_shift, gate(proposed_s)
    )
    mean, c_low, c_high = bootstrap(extra)
    print(f"  the correction at the tier threshold adds {mean:.2f} ({c_low:.2f} to {c_high:.2f})")
    print("September by bin, proposed score then corrected (requests, mean, wrong, interval):")
    for name, p_m in (("proposed", proposed_s), ("corrected", moved)):
        for n, mean, wrong in reliability(y_shift, p_m):
            low, high = wilson_interval(wrong, n)
            print(f"  {name}: {n}, {mean:.3f}, {wrong}, {wrong / n:.3f} ({low:.3f} to {high:.3f})")
    ratio = odds(y_shift.mean()) / odds(yv.mean())
    print(f"September's correction multiplies every odds by {ratio:.2f}")
    for q in (0.02, 0.15):
        print(f"doubling a risk of {q} multiplies its odds by {odds(2 * q) / odds(q):.2f}")
    print("September's frozen requests, after their one consultation:")
    print_record(september)
    print(f"fingerprints recompute after the reads: August {august.intact()},", end=" ")
    print(f"September {september.intact()}")
# endregion book:fnd.supervised-learning.numbers-01


# region book:fnd.supervised-learning.keys-01 output
# Keys to the chapter's numeric exercises; not included.
if __name__ == "__main__":
    pairs = log_loss(np.array([1, 0]), np.array([0.9, 0.9]))
    print(f"exercise 1: mean log loss of (1, 0.9) and (0, 0.9): {pairs:.4f};", end=" ")
    print(f"noise {noise:.4f}, squared bias {bias2:.4f}, variance {variance:.4f},", end=" ")
    print(f"total {noise + bias2 + variance:.4f}")
    on_sound, on_wrong = [0, REVIEW], [8.00, REVIEW]  # an $8 refund; actions: pay, refer
    adds = on_sound[1] - on_sound[0], on_wrong[0] - on_wrong[1]
    print(f"exercise 2a: an $8 refund: a needless review adds ${adds[0]:.2f}, a wrong", end=" ")
    print(f"payment adds ${adds[1]:.2f}, threshold {adds[0] / sum(adds):.3f};", end=" ")
    print(f"the mistakes themselves cost ${on_sound[1]:.2f} and ${on_wrong[0]:.2f}")
    page, wait = expected_costs([0.88, 0.12], [[2, 0], [0, 20]])  # actions: page, wait
    print(f"exercise 3a: page {page:.2f}, wait {wait:.2f} engineer-hours;", end=" ")
    print(f"page above {2 / (2 + 20):.4f}; at 0.12, page")
    for q in (0.08, 0.10, 0.15):
        call, leave = expected_costs([1 - q, q], [[3, 0], [0, 27]])  # actions: call, no call
        verdict = "tie" if math.isclose(call, leave) else "call" if call < leave else "no call"
        print(f"exercise 3b: call above {3 / (3 + 27):.2f}; at {q:.2f}, call {call:.2f}", end=" ")
        print(f"against {leave:.2f}: {verdict}")
    for amount in (8, 25, 45):
        low, high = band(REVIEW, amount, WRONG_DENIAL)
        own = "pay" if 0.15 < low else "refer"
        tier = "pay" if 0.15 < TIER_THRESHOLD else "refer"
        print(f"exercise 5: ${amount}: band {low:.3f} to {high:.3f}; at 0.15,", end=" ")
        print(f"{own} by its own edge, {tier} by the tier threshold")
    for q in (0.5, 0.9, 0.98):
        after = shift_base_rate(q, 0.5, 0.02)
        print(f"exercise 6: {q} becomes {after:.3f};", end=" ")
        print(f"at or above 0.10 before: {q >= 0.10}, after: {after >= 0.10}")
    weighted_v, weighted_a = (
        risk(models["class-weighted"], Xv),
        scores["class-weighted, as trained"],
    )
    planted = recalibration_report(
        AUGUST_WINDOW, AUGUST_WINDOW, weighted_a, y_test, weighted_a, y_test
    )
    honest = recalibration_report(
        VALIDATION_WINDOW, AUGUST_WINDOW, weighted_v, yv, weighted_a, y_test
    )
    print(
        f"exercise 7: reported error {planted['calibration_error']:.4f}, map fit on August itself;",
        end=" ",
    )
    print(f"refit on the validation window, {honest['calibration_error']:.4f}")
    print(
        f"exercise 7: honest bins {[(b['requests'], b['mean_prediction'], b['wrong_payments']) for b in honest['bins']]}"
    )
    exercise_8 = scores["class-weighted, with its map"]
    pay = gate(exercise_8)
    print(
        f"exercise 8: threshold {REVIEW / 49.99:.3f}; error {calibration_error(reliability(y_test, exercise_8)):.4f};",
        end=" ",
    )
    print(
        f"referred {int((~pay).sum())} of {len(y_test)}; cost {realized_cost(y_test, a_test, pay).mean():.2f}",
        end=" ",
    )
    print(f"against the placeholder's {placeholder_cost.mean():.2f}; denied on a score: 0")
    review_cost = realized_cost(y_test, a_test, gate(proposed, a_test))
    print(
        f"August: validation-calibrated proposed score with the review-cost rule saves ${np.mean(placeholder_cost - review_cost):.2f} per request"
    )
    if "--write-fixture" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(planted, indent=2) + "\n", encoding="utf-8")
# endregion book:fnd.supervised-learning.keys-01


# region book:fnd.supervised-learning.figures-01 output
# The data the chapter's three figures are drawn from; not included.
if __name__ == "__main__":

    def bins_with_intervals(y, p):
        return [[n, round(mean, 4), wrong, *(round(v, 4) for v in wilson_interval(wrong, n))]
                for n, mean, wrong in reliability(y, p)]  # fmt: skip

    figures = {
        "band": {"amount": 40.0, "review": REVIEW, "wrong_denial": WRONG_DENIAL,
                 "edges": [round(v, 4) for v in band(REVIEW, 40.0, WRONG_DENIAL)]},
        "cut_off": {
            "cut_offs": CUT_OFFS,
            "agent": [round(c, 4) for c in sweeps["boosted trees, with its map"]],
            "class_weighted": [round(c, 4) for c in sweeps["class-weighted, as trained"]],
            "placeholder": round(float(placeholder_cost.mean()), 4),
        },
        "reliability": {
            "class_weighted_august": bins_with_intervals(y_test, scores["class-weighted, as trained"]),
            "agent_august": bins_with_intervals(y_test, proposed),
            "agent_september": bins_with_intervals(y_shift, proposed_s),
        },
    }  # fmt: skip
    print(json.dumps(figures, indent=1))
# endregion book:fnd.supervised-learning.figures-01
