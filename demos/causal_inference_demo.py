"""Listings and outputs for causal inference: identify before estimating.

The hospital and two-arm examples are constructed. The refund planning figures use
chapter 5's boosted trees and map, fitted on its training and validation windows.
The map was fitted on validation itself: these figures plan a test and test nothing.
Only payment-status metadata enters the full-history census. No label later than
the validation window is read, and neither frozen month is consulted.

The printed listings, adjust-01 then assign-01, run in chapter order on their own.
Assignment is a rehearsal of a signed, capped exception that could start at stage 5.
It returns records, calls no agent or tool, and changes no rule. Importing runs
nothing. --write-fixture writes the constructed effect report for exercise 5.
"""

# ruff: noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import os

    for pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[pool] = "1"


def hospital_samples(size, seed=7):
    """Draw risk, call and readmission from the hospital's stipulated rates."""
    import numpy as np

    rng = np.random.default_rng(seed)
    high = rng.random(size) < 0.5
    call = rng.random(size) < np.where(high, 0.8, 0.2)
    rate = np.where(high, 0.4, 0.2) - 0.1 * call
    readmitted = rng.random(size) < rate
    return [(name, int((g & call).sum()), int((g & call & readmitted).sum()),
             int((g & ~call).sum()), int((g & ~call & readmitted).sum()))
            for name, g in (("high", high), ("low", ~high))]  # fmt: skip


def effect_intervals(counts, z=1.96):
    """Normal 95% intervals: sampling uncertainty, conditional on identification.

    Standardization includes estimation of stratum shares as well as arm means.
    These large-sample intervals cannot account for an omitted confounder.
    """
    import math

    def variance(n, k):
        if n < 2:
            raise ValueError("intervals need at least two units per arm and stratum")
        p = k / n
        return p * (1 - p) / (n - 1)

    total = sum(n1 + n0 for _, n1, _, n0, _ in counts)
    n1, k1, n0, k0 = (sum(c[i] for c in counts) for i in (1, 2, 3, 4))
    naive, adjusted = naive_effect(counts), standardized_effect(counts)
    v_naive = variance(n1, k1) + variance(n0, k0)
    v_adjusted = sum(((a + b) / total) ** 2 * (variance(a, x) + variance(b, y))
                     + ((a + b) / total) * (x / a - y / b - adjusted) ** 2 / total
                     for _, a, x, b, y in counts)
    return [(effect, effect - z * math.sqrt(v), effect + z * math.sqrt(v))
            for effect, v in ((naive, v_naive), (adjusted, v_adjusted))]


# region book:fnd.causal-inference.adjust-01 output
from demos.output_wrap import print_wrapped


def naive_effect(counts):
    """Rows: (stratum, action count, outcomes, no-action count, outcomes)."""
    n1, y1, n0, y0 = (sum(c[i] for c in counts) for i in (1, 2, 3, 4))
    if not n1 or not n0:
        raise ValueError("the comparison needs both actions")
    return y1 / n1 - y0 / n0


def standardized_effect(counts):
    """Compare strata; weight by each stratum's share of everyone."""
    total = sum(n1 + n0 for _, n1, _, n0, _ in counts)
    if not total or any(not n1 or not n0 for _, n1, _, n0, _ in counts):
        raise ValueError("every stratum needs both actions: positivity")
    return sum(
        (n1 + n0) / total * (y1 / n1 - y0 / n0) for _, n1, y1, n0, y0 in counts
    )


if __name__ == "__main__":
    from demos import causal_inference_demo as examples

    hospital = [("high", 400, 120, 100, 40), ("low", 100, 10, 400, 80)]
    print_wrapped("constructed hospital: 1,000 discharges")
    for name, n1, y1, n0, y0 in hospital:
        print_wrapped(f"{name}: phoned {y1}/{n1} = {y1 / n1:.3f};", end=" ")
        print_wrapped(
            f"not phoned {y0}/{n0} = {y0 / n0:.3f}; effect "
            f"{y1 / n1 - y0 / n0:+.3f}"
        )
    n1, y1, n0, y0 = (sum(c[i] for c in hospital) for i in (1, 2, 3, 4))
    print_wrapped(
        f"pooled: {y1 / n1:.3f} - {y0 / n0:.3f} = "
        f"{naive_effect(hospital):+.3f}"
    )
    print_wrapped(f"standardized: {standardized_effect(hospital):+.3f}")
    print_wrapped(
        "seed 7 samples; estimate [normal 95% interval], sampling error only"
    )
    for size in (1000, 10000, 100000):
        sample = examples.hospital_samples(size, seed=7)
        naive, adjusted = examples.effect_intervals(sample)
        cells = [
            f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            for e, lo, hi in (naive, adjusted)
        ]
        print_wrapped(f"{size:>7,}: naive {cells[0]}; standardized {cells[1]}")
# endregion book:fnd.causal-inference.adjust-01


import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

import numpy as np

from demos.supervised_learning_demo import FIT_BY, GO_LIVE, arrays, boosted, platt, risk
from refund_agent.data.history import generate_history

REVIEW, WRONG_DENIAL = 4.00, 20.00  # illustrative dollars
TIER_THRESHOLD = 0.080
VALIDATION_LAST_AT = GO_LIVE - timedelta(days=30)
FIXTURE = Path(__file__).with_name("data") / "causal_inference_effect_05.json"


def planning_history(seed=7):
    """Fit exactly chapter 5's proposed score, without reaching a frozen label.

    The census reads payment status, not outcomes, over all 36,000 requests. Date
    filtering precedes any label access, including the label-final timestamp.
    """
    ledger = generate_history(seed=seed)
    paid_count = sum(r.paid for r in ledger)
    census = len(ledger), paid_count, len(ledger) - paid_count
    allowed = [r for r in ledger if r.paid and r.at <= VALIDATION_LAST_AT]
    del ledger
    training, validation = [], []
    for r in allowed:
        final_at = r.label_final_at
        if final_at is None:
            raise ValueError("a paid planning request needs a final label")
        if final_at <= FIT_BY:
            training.append(r)
        elif final_at <= GO_LIVE:
            validation.append(r)
    Xt, yt, _ = arrays(training)
    Xv, yv, _ = arrays(validation)
    model = boosted().fit(Xt, yt)
    mapping, slope, intercept = platt(risk(model, Xv), yv)
    return {"census": census, "training": training, "validation": validation,
            "score": mapping(risk(model, Xv)), "map": (slope, intercept)}


def support_by_band(rows, score, edges=(0, .02, .04, .08, .12, .20, 1.0)):
    """Count observed referrals within fixed, left-closed score bands."""
    score = np.asarray(score)
    index = np.searchsorted(edges[1:-1], score, side="right")
    return [(edges[k], edges[k + 1], int((index == k).sum()),
             sum(r.policy_decision == "refer" for r, b in zip(rows, index) if b == k))
            for k in range(len(edges) - 1)]


def review_saving(y, amount, referred, q=0, f=0, review=REVIEW, wrong_denial=WRONG_DENIAL):
    """Dollars saved per request, under assumed missed-false and wrong-denial rates.

    A zero outcome is treated as sound for this calculation, as in chapter 5.
    This prices assumptions; the history contains no reviewer outcomes.
    """
    if not (0 <= q <= 1 and 0 <= f <= 1):
        raise ValueError("reviewer error rates must be probabilities")
    y, amount, referred = np.asarray(y), np.asarray(amount), np.asarray(referred, dtype=bool)
    n = len(y)
    if not n or len(amount) != n or len(referred) != n:
        raise ValueError("aligned, nonempty outcome, amount and selection arrays required")
    lost = float((y[referred] * amount[referred]).sum())
    sound = int((y[referred] == 0).sum())
    return ((1 - q) * lost - review * int(referred.sum()) - f * wrong_denial * sound) / n


def difference_in_differences(treated_pre, treated_post, control_pre, control_post):
    """A subtraction. It is the effect on the treated group only with one version of
    the treatment, an untreated comparison group, no spillover between the groups,
    the same kind of unit counted the same way in both periods, parallel trends and
    no anticipation."""
    return (treated_post - treated_pre) - (control_post - control_pre)


def requests_per_arm(sd, difference, z=NormalDist().inv_cdf(.975), z_power=NormalDist().inv_cdf(.8)):
    """Two equal-variance arms, two-sided 95% confidence and about 80% power.

    The count is of independent requests: one for each unit the coin is flipped for.
    """
    if not math.isfinite(sd) or not math.isfinite(difference) or sd < 0 or difference <= 0:
        raise ValueError("finite nonnegative standard deviation and positive difference required")
    return math.ceil(2 * (z + z_power) ** 2 * sd ** 2 / difference ** 2)


@dataclass(frozen=True)
class ThresholdExperimentPlan:
    assignment_unit: str = "customer (requester_id); arm fixed and analyzed at the first eligible request"
    pay_action: str = "pay with no review (D=0)"
    refer_action: str = "one reviewer with the evidence packet (D=1)"
    eligibility: str = "policy-payable tier 1; fresh evidence; review <= score * amount < cap"
    estimand: str = "mean cost by assigned arm, refer minus pay, in the set, with one half referred"
    outcome: str = "adjudicated cost per assigned request: review + wrong payment + wrong denial"
    label_window_days: int = 30
    review_cost: float = REVIEW
    wrong_denial_cost: float = WRONG_DENIAL
    cap: float = 10.00
    minimum_difference: float = 2.00
    refer_probability: float = .5
    seed: int = 7
    segments: tuple = ("reason", "prior refund: none or some", "amount: below $25 or at least $25")
    policy_version: str = "threshold-experiment-plan-1"
    first_assignment_at: datetime = datetime(2027, 1, 1, tzinfo=timezone.utc)
    last_assignment_at: datetime = datetime(2027, 12, 31, tzinfo=timezone.utc)
    analysis_at: datetime = datetime(2028, 1, 30, tzinfo=timezone.utc)
    signed_at: datetime = datetime(2026, 12, 20, tzinfo=timezone.utc)
    owner: str = "owner of the refund decision (illustrative signer; unsigned in practice)"
    dollar_ceiling: float = 100000.00
    denial_check: str = "blind second reading of every denial within 30 days; upheld appeals count"
    stop_rule: str = "halt before the next pay-arm amount would exceed the dollar ceiling; owner halts"
    versions: str = "one policy version; a change restarts the plan"
    interference: str = "shared review queue remains; no claim about referring the whole set"
    missing_outcomes: str = "report cost bounds for misjudged or unread denials and undiscovered false claims"
    counter_measure: str = "share referred; compare costs by assigned arm, including every denial"
    launch: str = "plan only; no earlier than stage 5; owner must sign the tier 1 exception"


def check_plan(plan):
    """Refuse missing design fields, absent support, immature labels or no loss bound.

    Acceptance checks a written design. Its illustrative signature cannot authorize
    a payment, and this module has no execution path.
    """
    if any(value is None or value == "" or value == () for value in asdict(plan).values()):
        raise ValueError("every plan field is required")
    if any(isinstance(v, str) and not v.strip() for v in asdict(plan).values()):
        raise ValueError("design fields cannot be blank")
    if type(plan.seed) is not int or type(plan.label_window_days) is not int:
        raise ValueError("seed and label window must be integers")
    if not 0 < plan.refer_probability < 1:
        raise ValueError("both actions need positive probability inside the set")
    for value in (plan.cap, plan.review_cost, plan.wrong_denial_cost, plan.dollar_ceiling,
                  plan.minimum_difference):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite costs, cap and dollar ceiling required")
    if plan.cap <= plan.review_cost or plan.label_window_days <= 0:
        raise ValueError("the set and label window must be nonempty")
    if not plan.segments or not all(isinstance(s, str) and s.strip() for s in plan.segments):
        raise ValueError("segments must be named in advance")
    for at in (plan.first_assignment_at, plan.last_assignment_at, plan.analysis_at, plan.signed_at):
        if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("plan dates must be timezone-aware")
    if plan.last_assignment_at < plan.first_assignment_at:
        raise ValueError("last assignment precedes the first")
    if plan.analysis_at < plan.last_assignment_at + timedelta(days=plan.label_window_days):
        raise ValueError("analysis must wait for the last assignment plus the label window")
    if plan.signed_at >= plan.first_assignment_at:
        raise ValueError("the signature must precede the first assignment")
    return "accepted"


def coin_range(n, probability=.5):
    """Central 95% binomial count range, exact for the independent customers here."""
    from scipy.stats import binom

    return tuple(int(v) for v in binom.ppf([.025, .975], n, probability))


def rehearse(plan, rows, score):
    from demos import causal_inference_demo as examples

    check_plan(plan)
    if len(rows) != len(score):
        raise ValueError("each request needs its frozen score")
    return [examples.assign(plan, r, float(p)) for r, p in zip(rows, score)]


def sizing(plan, rows, score, difference=None):
    """Size from validation pay outcomes, assuming equal refer-arm dispersion.

    Use a month's validation volume as one planning month's volume (30 days).
    Loss is gross pay-arm exposure at the target pay-arm count, not an
    incremental experiment loss or a ceiling on realized fixed-date counts.
    The formula counts independent requests, so the set must hold one request per
    customer; a customer's later requests share an arm and need a clustered analysis.
    """
    records = rehearse(plan, rows, score)
    difference = plan.minimum_difference if difference is None else difference
    selected = np.array([r["in_set"] for r in records])
    if selected.sum() < 2:
        raise ValueError("sizing requires at least two requests in the set")
    customers = [r["requester_id"] for r in records if r["in_set"]]
    if len(set(customers)) != len(customers):
        raise ValueError("sizing counts one eligible request per customer")
    _, y, amount = arrays(rows)
    costs = (y * amount)[selected]
    mean, sd = float(costs.mean()), float(costs.std(ddof=1))
    per_arm = requests_per_arm(sd, difference)
    n = int(selected.sum())
    perfect = float(costs.sum() - n * plan.review_cost)
    return {"validation_requests": len(rows), "set_requests": n, "monthly_volume": n, "mean": mean, "sd": sd,
            "difference": difference, "per_arm": per_arm,
            "months": math.ceil(2 * per_arm / n),
            "expected_loss": per_arm * mean, "worst_case_loss": per_arm * 49.99,
            "share": n / len(rows), "saving_per_referred": perfect / n,
            "population_saving": perfect / len(rows)}


def build_plan(planning, cap=10, difference=2):
    """Illustrative schedule and ceiling, written before a future run, never signed."""
    plan = replace(ThresholdExperimentPlan(), cap=cap, minimum_difference=difference)
    size = sizing(plan, planning["validation"], planning["score"], difference)
    last = plan.first_assignment_at + timedelta(days=30 * size["months"] - 1)
    return replace(plan, last_assignment_at=last,
                   analysis_at=last + timedelta(days=plan.label_window_days))


if __name__ == "__main__":
    planning = planning_history()
    rows, score = planning["validation"], planning["score"]
    _, y, amount = arrays(rows)
    referred = score * amount >= REVIEW
    plan = build_plan(planning)


# region book:fnd.causal-inference.support-01 output
if __name__ == "__main__":
    total, paid, unpaid = planning["census"]
    print(f"history census, payment-status metadata only: {total:,} requests")
    print(f"policy-payable tier 1: {paid:,} paid and labeled; {unpaid:,} unpaid, no outcome label")
    print(f"validation only: {len(rows)} requests; decisions {rows[0].at:%Y-%m-%d} to {rows[-1].at:%Y-%m-%d}")
    print("score band       paid  referred  share referred")
    for low, high, n, referred_count in support_by_band(rows, score):
        share = f"{referred_count/n:.3f}" if n else "undefined (empty band)"
        print(f"[{low:.2f}, {high:.2f}{']' if high == 1 else ')'}  {n:5}  {referred_count:8}  {share}")
    print("observed referral share: 0; positivity fails in every occupied band")
    extra = (score >= TIER_THRESHOLD) & ~referred
    print(f"tier threshold's extra reviews: {int(extra.sum())} (validation planning figures)")
    print(f"their mean pay cost ${float((y*amount)[extra].mean()):.2f} against a review's ${REVIEW:.2f}")
# endregion book:fnd.causal-inference.support-01


# region book:fnd.causal-inference.review-01 output
if __name__ == "__main__":
    ns, wrong = int(referred.sum()), int(y[referred].sum())
    lost, sound = float((y * amount)[referred].sum()), ns - wrong
    print("assumed reviewer errors; no reviewer effect was observed")
    print(f"validation {len(rows)}; referred by cost rule {ns}; found false {wrong}; no finding {sound}")
    print(f"wrong-payment dollars in that set ${lost:.2f}; review ${REVIEW:.2f}; wrong denial ${WRONG_DENIAL:.2f}")
    print("missed false  wrongly deny sound -> dollars saved per request; break-even denial share")
    for q in (0, .2, .4):
        break_even = ((1-q) * lost - REVIEW * ns) / (WRONG_DENIAL * sound)
        cells = [f"{f:.1f}: {review_saving(y, amount, referred, q, f):+.4f}" for f in np.linspace(0, .5, 6)]
        print(f"q={q:.1f}: {'; '.join(cells)}; break-even f={break_even:.4f}")
    print(f"q=0.2, f=0.1: ${review_saving(y, amount, referred, .2, .1):.4f} saved per request")
# endregion book:fnd.causal-inference.review-01


# region book:fnd.causal-inference.did-01 output
if __name__ == "__main__":
    treated_pre, treated_post, control_pre, control_post = .12, .18, .11, .13
    effect = difference_in_differences(treated_pre, treated_post, control_pre, control_post)
    before_after = treated_post - treated_pre
    after_only = treated_post - control_post
    untreated_post = treated_pre + (control_post - control_pre)
    print(f"Share resolved within the hour: switched {treated_pre:.2f} to "
          f"{treated_post:.2f}; others {control_pre:.2f} to {control_post:.2f}")
    print(f"Before and after at switched site {before_after:+.2f}; "
          f"after only, switched less others {after_only:+.2f}")
    print(f"Difference-in-differences: ({treated_post:.2f} - {treated_pre:.2f}) - "
          f"({control_post:.2f} - {control_pre:.2f}) = {effect:+.2f}")
    print(f"Switched site untreated after: {treated_pre:.2f} + ({control_post:.2f} - "
          f"{control_pre:.2f}) = {untreated_post:.2f}; gap to {treated_post:.2f} "
          f"= {treated_post - untreated_post:+.2f}")
    print("Switched site's effect if: one action version; others untreated, unaffected;")
    print("same kind of unit counted the same way; parallel trends; no anticipation.")
# endregion book:fnd.causal-inference.did-01


# region book:fnd.causal-inference.plan-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped("| Field | Illustrative plan; no run or real signature |")
    print_wrapped("|---|---|")
    summary = (
        ("Eligibility", plan.eligibility),
        ("Assignment", f"{plan.assignment_unit}; P(refer)={plan.refer_probability}; seed {plan.seed}"),
        ("Actions", f"{plan.pay_action}; {plan.refer_action}"),
        ("Estimand", f"{plan.estimand}; minimum difference ${plan.minimum_difference:.2f}"),
        ("Outcome", f"{plan.outcome}; {plan.label_window_days}-day label window"),
        ("Costs", f"review ${plan.review_cost:.2f}; wrong denial ${plan.wrong_denial_cost:.2f}"),
        ("Limits", f"cap ${plan.cap:.2f}; payout ceiling ${plan.dollar_ceiling:.2f}"),
        ("Stop rule", plan.stop_rule),
        ("Denial check", plan.denial_check),
        ("Segments", "; ".join(plan.segments)),
        ("Interference", plan.interference),
        ("Missing outcomes", plan.missing_outcomes),
        ("Assignments", f"first {plan.first_assignment_at:%Y-%m-%d}; last {plan.last_assignment_at:%Y-%m-%d}"),
        ("Analysis", f"{plan.analysis_at:%Y-%m-%d}; {plan.counter_measure}"),
        ("Policy", f"{plan.policy_version}; {plan.versions}"),
        ("Authority", f"{plan.owner}; illustrative signature {plan.signed_at:%Y-%m-%d}"),
        ("Launch", plan.launch),
    )
    for field, value in summary:
        value = value.replace("$", r"\$")
        print_wrapped(f"| {field} | {value} |")
# endregion book:fnd.causal-inference.plan-01


# region book:fnd.causal-inference.plan-full-01 output
if __name__ == "__main__":
    print(json.dumps(asdict(plan), indent=2, sort_keys=True, default=lambda at: at.isoformat()))
# endregion book:fnd.causal-inference.plan-full-01


# region book:fnd.causal-inference.plan-02 output
if __name__ == "__main__":
    print(f"written plan: {check_plan(plan)}; no launch authorization")
    for name, changed in (
        ("probability 1 inside the set", replace(plan, refer_probability=1)),
        ("analysis before the last label is final", replace(plan, analysis_at=plan.last_assignment_at + timedelta(days=29))),
        ("no dollar ceiling", replace(plan, dollar_ceiling=None)),
    ):
        try:
            check_plan(changed)
        except ValueError as exc:
            print(f"{name}: refused")
            print(f"  {exc}")
# endregion book:fnd.causal-inference.plan-02


# region book:fnd.causal-inference.assign-01 output
from hashlib import sha256


def assign(plan, request, score):
    """Return a stateless assignment; read no outcome, take no action."""
    if not 0 <= score <= 1:
        raise ValueError("score must be a probability")
    if not 0 < plan.refer_probability < 1:
        raise ValueError(
            "both actions need positive probability inside the set"
        )
    payable = (
        request.policy_decision == "approve"
        and 0 < request.amount_cents <= 4999
    )
    fresh = (
        max(
            request.order_evidence_age_seconds,
            request.shipment_evidence_age_seconds,
        )
        <= 1800
    )
    predicted_loss = score * request.amount_cents / 100
    inside = (
        payable and fresh and plan.review_cost <= predicted_loss < plan.cap
    )
    if inside:
        digest = sha256(
            f"{plan.seed}:{request.requester_id}".encode()
        ).digest()
        coin = int.from_bytes(digest, "big") / 2**256
        action = "refer" if coin < plan.refer_probability else "pay"
        probability = (
            plan.refer_probability
            if action == "refer"
            else 1 - plan.refer_probability
        )
    else:
        action = "pay" if payable and fresh and score < 0.080 else "refer"
        if request.policy_decision == "deny":
            action = "deny"  # the policy's denial, never a denial on a score
        probability = 1.0
    return {
        "request_id": request.request_id,
        "requester_id": request.requester_id,
        "action": action,
        "probability": probability,
        "in_set": inside,
        "policy_version": plan.policy_version,
        "seed": plan.seed,
        "decision_at": request.at.isoformat(),
        "score": score,
        "source_policy_version": request.policy_version,
        "inputs": {
            "amount_cents": request.amount_cents,
            "reason": request.reason,
            "prior_refund_count": request.prior_refund_count,
            "paid_cents": request.paid_cents,
            "delivery_age_days": request.delivery_age_days,
            "order_evidence_age_seconds": request.order_evidence_age_seconds,
            "shipment_evidence_age_seconds": (
                request.shipment_evidence_age_seconds
            ),
        },
    }


if __name__ == "__main__":
    from demos import causal_inference_demo as examples

    rehearsal = examples.planning_history(seed=7)
    rehearsal_plan = examples.build_plan(rehearsal)
    records = examples.rehearse(
        rehearsal_plan, rehearsal["validation"], rehearsal["score"]
    )
    selected = [r for r in records if r["in_set"]]
    pay = sum(r["action"] == "pay" for r in selected)
    low, high = examples.coin_range(len(selected))
    print(
        f"rehearsal only, seed {rehearsal_plan.seed}; cap "
        f"${rehearsal_plan.cap:.2f}"
    )
    print(
        f"set: {len(selected)}; pay {pay}; refer {len(selected) - pay};",
        "each logs probability 0.5",
    )
    print(
        f"fair coin's central 95% pay-count range: {low} to {high};",
        f"inside: {low <= pay <= high}",
    )
    outside = [r for r in records if not r["in_set"]]
    print(
        f"outside: {len(outside)};",
        f"pay {sum(r['action'] == 'pay' for r in outside)};",
        f"refer {sum(r['action'] == 'refer' for r in outside)};",
        "each logs probability 1",
    )
    print("records carry request, customer, action, probability, set,")
    print("policy versions, seed, decision time, score and inputs")
    print("no payment, referral, agent run or frozen-window consultation")
# endregion book:fnd.causal-inference.assign-01


def print_size(size, plan):
    from demos.output_wrap import print_wrapped

    print_wrapped(f"set {size['set_requests']} of {size['validation_requests']}, one request per customer;")
    print_wrapped(f"one planning month's volume {size['monthly_volume']}")
    print_wrapped(f"pay outcome: mean ${size['mean']:.4f}; sample standard deviation ${size['sd']:.4f}")
    print_wrapped("assume the refer arm's standard deviation equals the pay arm's;")
    print_wrapped("95% confidence, 80% power; requests counted as independent")
    for difference in (1, 2, 4):
        n = requests_per_arm(size["sd"], difference)
        months = math.ceil(2*n / size["monthly_volume"])
        print_wrapped(f"difference ${difference:.2f}: {n:,} per arm (independence-only benchmark); {months} assignment months + {plan.label_window_days} days")
        print_wrapped(f"  at the target pay-arm count: expected ${n*size['mean']:,.2f}; worst case ${n*49.99:,.2f};")
        print_wrapped(f"within the dollar ceiling: {n*49.99 <= plan.dollar_ceiling}")
    print_wrapped("worst case: every pay-arm request false at the tier 1 maximum, $49.99")
    print_wrapped(f"illustrative dollar ceiling ${plan.dollar_ceiling:,.2f}")
    print_wrapped(f"set share {size['share']:.6f} x perfect-review saving ${size['saving_per_referred']:.4f}")
    print_wrapped(f"= ${size['population_saving']:.4f} per validation request")
    print_wrapped("Months use expected equal-arm accrual;")
    print_wrapped("actual arm counts may fall short of the target on the fixed date.")


# region book:fnd.causal-inference.size-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    print_wrapped("The score's map was fit on this validation window: planning figures, not a test.")
    print_wrapped("June validation, with 1 July's boundary request, stands for one 30-day month.")
    size = sizing(plan, rows, score, difference=2)
    print_size(size, plan)
    print_wrapped("Loss is gross wrong-payment exposure, not an estimated incremental effect of review.")
# endregion book:fnd.causal-inference.size-01


def arm_costs(n, amount, pay_false, denials, refer_false, wrong_denials):
    """Constructed, adjudicated costs by assigned arm, not by who was paid.

    The denial and wrong-payment categories are disjoint. The found-false-payment
    contrast estimates stopped claims under equal prevalence only if every paid
    false claim is found; equal but incomplete detection does not suffice.
    The standard error and the interval treat each arm's n requests as independent
    draws: one request per customer, and no request's cost moved by another's arm.
    """
    pay = np.array([amount] * pay_false + [0.] * (n-pay_false))
    refer = np.array([REVIEW+amount] * refer_false + [REVIEW+WRONG_DENIAL] * wrong_denials
                     + [REVIEW] * (n-refer_false-wrong_denials))
    difference = float(refer.mean()-pay.mean())
    se = math.sqrt(float(refer.var(ddof=1)/n + pay.var(ddof=1)/n))
    return {"pay": float(pay.mean()), "refer": float(refer.mean()), "difference": difference,
            "se": se, "interval": (difference-1.96*se, difference+1.96*se),
            "pay_rate": pay_false/n, "refer_paid_rate": refer_false/(n-denials),
            "found_false_difference": pay_false-refer_false}


def effect_report():
    """Exercise 5's single planted error: compare selected paid denominators."""
    return {"scenario": "constructed refund threshold experiment; dollars illustrative",
            "amount": 40, "review_cost": REVIEW, "wrong_denial_cost": WRONG_DENIAL,
            "pay_arm": {"assigned": 400, "paid": 400, "found_false": 48},
            "refer_arm": {"assigned": 400, "denied": 80, "paid": 320,
                          "found_false": 16, "wrong_denials_on_second_reading": 50},
            "headline": {"claim": "review cut wrong payments from 12% to 5%",
                         "pay_rate": 48/400, "refer_paid_rate": 16/320}}


# region book:fnd.causal-inference.numbers-01 output
if __name__ == "__main__":
    patients = [(1, 1, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0)]  # D, Y(0), Y(1)
    print("constructed patients: patient D Y(0) Y(1) observed effect")
    for i, (d, y0, y1) in enumerate(patients, 1):
        print(f"{i} {d} {y0} {y1} {(y0,y1)[d]} {y1-y0:+d}")
    average = sum(y1-y0 for _, y0, y1 in patients)/len(patients)
    naive = sum(y1 for d, _, y1 in patients if d)/2 - sum(y0 for d, y0, _ in patients if not d)/2
    print(f"average effect {average:+.2f}; observed comparison {naive:+.2f}")
    arms = arm_costs(1000, 40, 100, 150, 30, 80)
    print("constructed arms: 1,000 each, amount $40; pay found false 100;")
    print("refer denied 150, paid 850, found false 30; stipulated wrong denials 80")
    print(f"estimated reduction in found false payments {arms['found_false_difference']}, a total; assumes equal prevalence")
    print("equals estimated stopped false claims only if every paid false claim is found; no denial is adjudicated by this contrast")
    print(f"cost per assigned request: pay ${arms['pay']:.2f}; refer ${arms['refer']:.2f};", end=" ")
    print(f"difference ${arms['difference']:.2f}; SE ${arms['se']:.3f};", end=" ")
    print(f"95% interval ${arms['interval'][0]:.2f} to ${arms['interval'][1]:.2f}")
    print(f"found false among paid: {100*arms['pay_rate']:.1f}% against {100*arms['refer_paid_rate']:.1f}%")
    unknown_denials = (REVIEW*1000+40*30)/1000
    print(f"without denial adjudication: refer cost ${unknown_denials:.2f} to ${unknown_denials+150*WRONG_DENIAL/1000:.2f}")
    print(f"sample size, sd $12, difference $2: {requests_per_arm(12, 2)} per arm")
    print(f"half the difference ($1): {requests_per_arm(12, 1)} per arm (fourfold before rounding)")
# endregion book:fnd.causal-inference.numbers-01


# region book:fnd.causal-inference.keys-01 output
if __name__ == "__main__":
    print("exercise 1a: effects -1, -1, 0, 0; average -0.50; observed outcomes 0, 0, 0, 0")
    print("exercise 1a: records hold D and Y(D), never both potential outcomes")
    print("exercise 1b: referral share 0; positivity fails; records price pay, not review")
    print("exercise 2a: call <- risk -> readmission; answering the phone follows the call, so leave it unadjusted")
    print("exercise 2b: randomize fresh payable tier 1 with $4 <= score x amount < cap, by customer")
    print("exercise 2b: log assigned arm and its probability; bound predicted loss per request and gross dollar exposure")
    incidents = [("critical", 200, 60, 50, 10), ("other", 50, 40, 200, 150)]
    print(f"exercise 3: naive {naive_effect(incidents):+.3f}; strata +0.100, +0.050;", end=" ")
    print(f"standardized {standardized_effect(incidents):+.3f}")
    from fractions import Fraction

    average = Fraction(1,4)*Fraction(3,10) - Fraction(3,4)*Fraction(1,10)
    print(f"exercise 3: 0.25 x 0.3 + 0.75 x (-0.1) = {float(average):.3f};", end=" ")
    print("zero hides one helped group and one harmed group")
    n = requests_per_arm(10, 2.5)
    months = math.ceil(2*n/80)
    print(f"exercise 4: {n} per arm; {months} assignment months + 30 days;", end=" ")
    print(f"at the target pay-arm count: worst case ${n*49.99:,.2f}; expected ${n*6:,.2f}")
    print(f"exercise 4: DiD {difference_in_differences(.050,.038,.048,.046):+.3f} per policy-payable request; parallel trends and no anticipation,")
    print("exercise 4: with one meaning for the release, untreated other regions, no spillover")
    print("exercise 4: and the same kind of request counted the same way")
    report = arm_costs(400, 40, 48, 80, 16, 50)
    print("exercise 5: headline uses 16/320 against 48/400; the reviewer selected the first denominator")
    print(f"exercise 5: assigned-arm costs ${report['pay']:.2f} and ${report['refer']:.2f};", end=" ")
    print(f"difference ${report['difference']:.2f}; estimated reduction in found false payments {report['found_false_difference']}")
    print("exercise 5: return the assigned-arm table, include wrong denials, reject the headline as a benefit claim")
    larger = build_plan(planning, cap=15)
    larger_size = sizing(larger, rows, score)
    trial = rehearse(larger, rows, score)
    selected = [r for r in trial if r["in_set"]]
    pay = sum(r["action"] == "pay" for r in selected)
    low, high = coin_range(len(selected))
    reseeded = rehearse(replace(larger, seed=8), rows, score)
    moved = sum(a["action"] != b["action"] for a, b in zip(trial, reseeded) if not a["in_set"])
    print(f"exercise 6: cap $15; plan {check_plan(larger)}; set {len(selected)}; pay {pay}; refer {len(selected)-pay}")
    print(f"exercise 6: fair coin range {low} to {high}; every set probability 0.5, outside 1; no tool or holdout read")
    print(f"exercise 6: actions outside the set that change with the seed (7 against 8): {moved}")
    print(f"exercise 6: {larger_size['per_arm']} per arm for $2; {larger_size['months']} months + 30 days;", end=" ")
    print(f"at the target pay-arm count: expected ${larger_size['expected_loss']:,.2f}; worst case ${larger_size['worst_case_loss']:,.2f}")
    if "--write-fixture" in sys.argv:
        FIXTURE.write_text(json.dumps(effect_report(), indent=2)+"\n", encoding="utf-8")
# endregion book:fnd.causal-inference.keys-01


# region book:fnd.causal-inference.figures-01 output
if __name__ == "__main__":
    grid = [i/100 for i in range(51)]
    curves = [{"missed_false": q,
               "saving": [round(review_saving(y, amount, referred, q, f), 6) for f in grid],
               "break_even": round(((1-q)*lost-REVIEW*ns)/(WRONG_DENIAL*sound), 6)}
              for q in (0, .2, .4)]
    figure_data = {
        "review_errors": {"wrongly_deny_sound": grid, "curves": curves,
                          "perfect_review": round(review_saving(y, amount, referred), 6)},
        "pay_probability": {"review": plan.review_cost, "cap": plan.cap,
                            "levels": [None, 1-plan.refer_probability, 0],
                            "below_review": "gate retained: pay if score < 0.080, otherwise refer",
                            "boundary": "left-closed: review <= expected loss < cap"},
    }
    print(json.dumps(figure_data, indent=1))
# endregion book:fnd.causal-inference.figures-01
