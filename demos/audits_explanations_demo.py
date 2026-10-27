"""Independent arithmetic and explicit record fixtures for chapter 36.

Canonical month captures are in audits_explanations_integrated_demo.py. Nothing
here substitutes these fixtures for chapter 34's regions or chapter 35's scope.
"""

from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb

from demos.audits_explanations_examples import (
    context_from_records,
    example_records,
)
from refund_agent.audit.audits_explanations_explain import explain, replay
from refund_agent.audit.audits_explanations_rates import (
    AuditRule,
    breaches,
    ceiling_test,
    exact_upper_count,
    fpr_from_identity,
    minimum_release_sample,
    power,
    wilson_interval,
    wrongful_denial_rate,
)
from refund_agent.rules import draft


def linear_attributions(weights, x, reference):
    if not len(weights) == len(x) == len(reference):
        raise ValueError(
            "weights, applicant and reference dimensions must agree"
        )
    return tuple(w * (a - b) for w, a, b in zip(weights, x, reference))


def print_identity():
    print(
        "Constructed hospitals, 1000 patients each. Shared PPV .5 and TPR .8."
    )
    rates = []
    for p in (0.10, 0.30):
        fpr = fpr_from_identity(p, 0.5, 0.8)
        rates.append(fpr)
        tp, fp, positives = (
            round(1000 * p * 0.8),
            round(1000 * (1 - p) * fpr),
            round(1000 * p),
        )
        print(
            f"base {p:.2f}: FPR {fpr:.9f}. TP {tp}, FP {fp}, FN {positives - tp}."
        )
        print(f"PPV=1 boundary: TP {tp}, FP 0, FN {positives - tp}. TPR .8.")
    print(f"FPR ratio second/first: {rates[1] / rates[0]:.6f}")
    for name, p in (("A", 0.8), ("B", 0.9)):
        ppv = 0.98 * p / (0.98 * p + 0.02 * (1 - p))
        print(
            f"Constructed not-delivered settings {name}: base {p:.2f}, PPV {ppv:.9f}."
        )
    print(
        "Shared error rates: TPR .98 and FPR .02. Realized counts are in month-01."
    )


def print_attributions():
    names = ("utilization", "recent inquiries", "years of history")
    weights, x, intercept = (3.0, 0.5, -0.2), (0.9, 4, 3), -3.0
    score = intercept + sum(w * v for w, v in zip(weights, x))
    print(f"Constructed lender. Weights {weights}, intercept {intercept:.1f}.")
    print(f"Applicant {x}. Log-odds {score:.1f}.")
    for name, ref in (("average", (0.3, 1, 8)), ("best possible", (0, 0, 20))):
        values = linear_attributions(weights, x, ref)
        reference_score = intercept + sum(w * v for w, v in zip(weights, ref))
        leader = names[max(range(3), key=lambda i: abs(values[i]))]
        print(f"Reference {name}: {ref}.")
        print("Attributions: " + ", ".join(f"{v:.1f}" for v in values))
        print(
            f"Sum {sum(values):.1f}, log-odds gap {score - reference_score:.1f}."
        )
        print(f"Principal reason: {leader}.")


def print_keys():
    print(
        "Exercise 1: compare oracle counts in month-01, not the settings alone."
    )
    print(
        "Exercise 2: replay tests fidelity and eligibility, not truth or usefulness."
    )
    print("Exercise 3: constructed branches, 3000 applicants each.")
    rates = []
    for p in (0.10, 0.25):
        fpr = fpr_from_identity(p, 0.6, 0.7)
        rates.append(fpr)
        tp, fp = round(3000 * p * 0.7), round(3000 * (1 - p) * fpr)
        print(f"base {p:.2f}: FPR {fpr:.9f}. TP {tp}, false declines {fp}.")
    ppv = 0.7 * 0.25 / (0.7 * 0.25 + rates[0] * 0.75)
    print(f"Second branch PPV at first branch FPR, TPR .7: {ppv:.9f}")
    lo, hi = wilson_interval(3, 80)
    estimate = wrongful_denial_rate(0.4, 3 / 80, 12 / 360)
    interval = tuple(wrongful_denial_rate(0.4, p, 12 / 360) for p in (lo, hi))
    K, chance = ceiling_test(3, 240, 348, 80, 0.01)
    print("Exercise 4: D, decided 600, denied 240, paid 360, found false 12.")
    print(f"Sample 3/80 = {3 / 80:.6f}. Wilson {lo:.9f} to {hi:.9f}.")
    print(f"Owed-denial estimate {estimate:.9f}.")
    print(f"Conditional range {interval[0]:.9f} to {interval[1]:.9f}.")
    print(
        f"Owed payments 348. Ceiling count {K}. Exact tail {float(chance):.9f}."
    )
    print(f"Family level .05/3. Breach {breaches(chance, 0.05, 3)}.")
    print("Exercise 5: reconstruct from blind readings, not appeals.")
    print("Canonical seed-11 report: exercises/audits-explanations/audit_report_05.json.")
    print(
        "Exercise 6: retain all clauses, evaluation clock, and required approvals."
    )


def print_boundaries():
    print(
        "Constructed finite-population boundaries, not a month's observed counts."
    )
    for k, N, owed, m in (
        (3, 239, 395, 80),
        (4, 239, 395, 80),
        (4, 239, 396, 80),
        (5, 5, 990, 5),
        (0, 5, 0, 5),
    ):
        K, chance = ceiling_test(k, N, owed, m, Fraction(1, 100))
        label = "unassessable" if owed == 0 and k == 0 else "assessable"
        print(
            f"k={k}, N={N}, owed={owed}, m={m}: Kceil={K}, tail={float(chance):.9f}."
        )
        print(f"Entitlement denominator status: {label}.")
    rule = AuditRule(
        signed_at=datetime(2026, 5, 1, tzinfo=timezone.utc) - timedelta(days=1)
    )
    for K in (21, 22, 23, 4):
        p = power(rule, 239, K, 395, 80)
        print(
            f"Given N=239 and owed=395: true wrongful={K}, exact power={float(p):.9f}."
        )
    print(f"Probability of all 3 in sample: {comb(80, 3) / comb(239, 3):.9f}.")


def print_release():
    print(
        "Constructed hold-release example: N=239 proposals, owed payments=395."
    )
    needed = minimum_release_sample(239, 395, 0.01)
    if needed is None:
        raise ValueError("the hold example has no assessable release sample")
    for m in (80, needed - 1, needed):
        upper = exact_upper_count(0, 239, m)
        rate = upper / (395 + upper)
        print(f"Clean readings {m}: exact 95% upper wrongful count {upper}.")
        print(f"Upper owed-denial share {rate:.9f}. Release: {rate < 0.01}.")
    print(f"Minimum clean sample for release: {needed}.")
    print("Nondetection with 80 reads keeps the hold. Review continues.")


def print_checks():
    print(
        "Explicit chapter-36 record fixtures. Partner integration is not asserted."
    )
    for kind in ("two-clause", "exhausted", "tier2", "tier3"):
        records = example_records(kind)
        own_order = records["current"]["order"]["order_id"]
        own_subject = records["current"]["request"]["requester_id"]
        ctx = context_from_records(
            records, may_act=lambda s, o: s == own_subject and o == own_order
        )
        state = ctx.current.state()
        e = explain(state, draft(state), state.refunded_cents, context=ctx)
        results = replay(e, state, context=ctx)
        print(f"{kind}: reasons {', '.join(e.reasons)}.")
        print(
            f"Counterfactual {e.counterfactual.outcome}, distance {len(e.counterfactual.changes)}."
        )
        print(
            f"Required approval {e.counterfactual.approval_required}, granted False."
        )
        print(f"Replay {sum(results.values())}/6. Execution asserted False.")
        if kind == "two-clause":
            delivered = max(
                event.at for event in ctx.current.shipment.events
                if event.status == "delivered"
            )
            print(
                f"Amounts {state.request.amount_cents} to {e.counterfactual.evaluation.request.amount_cents} cents."
            )
            print(
                f"Submission day {(state.request.at - delivered).days}, "
                f"execution day {(ctx.current.at - delivered).days}, "
                "counterfactual evaluation day "
                f"{(e.counterfactual.evaluation.at - delivered).days}."
            )


# region book:fnd.audits-explanations.identity-01 output
print_identity()
# endregion book:fnd.audits-explanations.identity-01

# region book:fnd.audits-explanations.attrib-01 output
print_attributions()
# endregion book:fnd.audits-explanations.attrib-01

# region book:fnd.audits-explanations.keys-01 output
print_keys()
# endregion book:fnd.audits-explanations.keys-01

# region book:fnd.audits-explanations.boundaries-01 output
print_boundaries()
# endregion book:fnd.audits-explanations.boundaries-01

# region book:fnd.audits-explanations.release-01 output
print_release()
# endregion book:fnd.audits-explanations.release-01

# region book:fnd.audits-explanations.checks-01 output
print_checks()
# endregion book:fnd.audits-explanations.checks-01
