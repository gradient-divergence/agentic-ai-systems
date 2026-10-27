"""Cell coverage of one capped offer replay, using learning requests only.

No stored log or package is an input. The two tables partition the same sends;
adding context does not rerun or change the reason-only logging policy.
The additive logistic design has no interactions and is counted, not fitted.
Its earlier-refund flag uses only strictly earlier paid learning requests, not
the history generator's prior-refund field or any false-claim outcome.
"""

from itertools import groupby, product

from refund_agent.offers.evaluation import EVALUATION_START
from refund_agent.offers.notices import OFFERS, eligible
from refund_agent.offers.policies import Capped, EpsilonGreedy
from refund_agent.offers.replay import TRAIN_START, replay, training_rows
from refund_agent.offers.response import NO_CONTACT_RATE

REASONS = tuple(NO_CONTACT_RATE)
AMOUNT_CUT = 2500


def earlier_refund_flags(rows):
    """A simultaneous request cannot count as an earlier refund."""
    seen, flags = set(), {}
    ordered = sorted(rows, key=lambda row: (row.at, row.request_id))
    for _, group in groupby(ordered, key=lambda row: row.at):
        simultaneous = list(group)
        flags.update((row.request_id, row.requester_id in seen) for row in simultaneous)
        seen.update(row.requester_id for row in simultaneous)
    return flags


def logistic_features(row, offer, earlier_refund):
    """Features for sigmoid(theta @ features); damaged/standard are references.

    One intercept, two reason contrasts, two offer contrasts, one amount-band
    coefficient and one earlier-refund coefficient. No requester identifier is
    a fitted input; it only supplies the decision-time earlier-refund flag.
    """
    return (
        1,
        *(int(row.reason == reason) for reason in REASONS[1:]),
        *(int(offer == action) for action in OFFERS[1:]),
        int(row.amount_cents >= AMOUNT_CUT),
        int(earlier_refund),
    )


def count_offer_cells(rows, sent):
    """Count one sent notice per eligible learning request, including zeros."""
    if not rows or any(
        not TRAIN_START <= row.at < EVALUATION_START
        or row.reason not in REASONS
        or not eligible(row)
        for row in rows
    ):
        raise ValueError("eligible learning-window requests required")
    requests = {row.request_id: row for row in rows}
    if len(requests) != len(rows):
        raise ValueError("unique learning request identifiers required")
    if len(sent) != len(rows) or {row.request_id for row in sent} != requests.keys():
        raise ValueError("one sent notice per learning request required")
    flags = earlier_refund_flags(rows)
    coarse = dict.fromkeys(product(REASONS, OFFERS), 0)
    refined = dict.fromkeys(product(REASONS, (False, True), (False, True), OFFERS), 0)
    for decision in sent:
        row = requests[decision.request_id]
        coarse[row.reason, decision.offer] += 1
        refined[row.reason, row.amount_cents >= AMOUNT_CUT, flags[row.request_id], decision.offer] += 1
    return dict(
        coarse=coarse,
        refined=refined,
        requests=len(rows),
        requesters=len({row.requester_id for row in rows}),
        earlier_refunds=sum(flags.values()),
        logistic_weights=len(logistic_features(rows[0], OFFERS[0], False)),
    )


def offer_cells():
    """Use chapter 17's learning window, seed, epsilon and exposure cap."""
    rows = [row for row in training_rows() if row.at < EVALUATION_START]
    settings = dict(seed=7, epsilon=0.3, cap=0.05)
    run = replay(rows, Capped(EpsilonGreedy(settings["epsilon"]), settings["cap"]), settings["seed"])
    return dict(count_offer_cells(rows, run.log.rows), settings=settings)


def print_offer_cells(result):
    print(f"learning window: [{TRAIN_START:%Y-%m-%d}, {EVALUATION_START:%Y-%m-%d})")
    settings = result["settings"]
    print(
        f"logger: seed {settings['seed']}; epsilon {settings['epsilon']:.2f}; "
        f"cap {settings['cap']:.2f}; reason-only learner"
    )
    print(f"amount band: under {AMOUNT_CUT} cents / {AMOUNT_CUT} cents and over")
    print("earlier refund: a strictly earlier paid learning row for this requester")
    print(
        f"requests {result['requests']}; requesters {result['requesters']}; "
        f"earlier-refund flags {result['earlier_refunds']}"
    )
    print("reason | standard | dated | referenced")
    for reason in REASONS:
        print(reason + " | " + " | ".join(str(result["coarse"][reason, a]) for a in OFFERS))
    coarse, refined = result["coarse"], result["refined"]
    print(
        f"{len(coarse)} cells: {sum(n == 0 for n in coarse.values())} empty; "
        f"{sum(n < 10 for n in coarse.values())} below 10 (includes empty)"
    )
    print("reason | amount cents | earlier refund | standard | dated | referenced")
    for reason, high, prior in product(REASONS, (False, True), (False, True)):
        label = f"{reason} | {'>=' if high else '<'}{AMOUNT_CUT} | {'yes' if prior else 'no'}"
        print(label + " | " + " | ".join(str(refined[reason, high, prior, a]) for a in OFFERS))
    print(
        f"{len(refined)} cells: {sum(n == 0 for n in refined.values())} empty; "
        f"{sum(n < 10 for n in refined.values())} below 10 (includes empty)"
    )
    print(
        f"logistic: {result['logistic_weights']} weights "
        f"(1 intercept + {len(REASONS) - 1} reason + {len(OFFERS) - 1} offer + 1 amount + 1 prior)"
    )
