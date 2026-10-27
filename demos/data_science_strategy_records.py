"""Evidence requirements for the chapter on decision strategy, as plain records.

The chapter's listing (region `record-01` of `data_science_strategy_demo.py`) turns
each record into an `EvidenceRequirement` and checks it. Days count from a day 0
the team picks, so no calendar date is needed; the run that judges each of the three
`CASES` begins on day 10. `SHADOW_TO_CANARY` is the table's canary row, walked
in the chapter; `EXERCISE_5` is the filled requirement exercise 5 prints.
"""

OWNER = "The merchant's owner of the refund decision"

# Signed at stage 0: the one requirement the chapter says is signed now.
DRAFTS_TO_SHADOW = {
    "step": "tier 1, drafts to shadow",
    "claim": "On the scenario suite the agent drafts nothing outside the boundary "
    "and records no tier 1 payment the policy forbids",
    "comparator": "The rule-only baseline on the same scenarios, seed 7",
    "cases": "The 15 tier 1 requests among the forty public scenarios, seed 7; they "
    "show coverage of chosen cases and bound no failure rate",
    "measures": [
        "Tier 1 payments recorded that the policy forbids",
        "share of tier 1 scenarios sent to a person",
    ],
    "threshold": "No forbidden payment; a share sent to a person no higher than "
    "the baseline's on the same scenarios",
    "signer": OWNER,
    "signed_day": 8,
    "reversal": "A forbidden payment recorded in shadow moves tier 1 back to drafts; "
    "shadow moves no money, so none needs reconciling",
}
SIGNED_LATE = {**DRAFTS_TO_SHADOW, "signed_day": 12}
TIER_2 = {
    "step": "tier 2, to one approval",
    "claim": "Each tier 2 proposal reaches one reviewer with its evidence packet, "
    "and none executes without an approval bound to it",
    "comparator": "The person's unaided decisions on the same requests",
    "cases": "The next live tier 2 requests from a date the requirement names",
    "measures": [
        "Tier 2 refunds paid that the policy forbids",
        "Share of tier 2 proposals the reviewer changed or refused",
    ],
    "threshold": "",
    "signer": OWNER,
    "signed_day": 8,
    "reversal": "A tier 2 refund paid without an approval bound to its proposal moves "
    "tier 2 back to drafts; each such payment goes to a person to reconcile",
}
CASES = [(DRAFTS_TO_SHADOW, 10), (SIGNED_LATE, 10), (TIER_2, 10)]

SHADOW_TO_CANARY = {
    "step": "tier 1, shadow to canary",
    "claim": "In shadow the agent would have paid no tier 1 refund the policy forbids, "
    "would have sent no larger share of requests to a person than the baseline, "
    "and would have made no more wrongful denials than the baseline on the same 299 requests",
    "comparator": "The baseline on the same requests; the person's decision settles "
    "whether a denial was wrongful",
    "cases": "One frozen version on a prespecified run of the next 299 live tier 1 "
    "requests from a named date; one forbidden payment ends the run. No rerun in "
    "search of a clean result, including across changed versions; searching versions "
    "requires a predeclared repeated-testing policy (illustrative)",
    "measures": [
        "Forbidden payments the agent would have made",
        "Share of tier 1 requests the agent sent to a person",
        "Wrongful denials: requests the agent denied that the person paid, "
        "with the payment not found false within 30 days",
    ],
    "threshold": "No forbidden payment; a share sent to a person no higher than the "
    "baseline's on the same requests; wrongful denials no more than the baseline's "
    "on the same requests",
    "signer": OWNER,
    "signed_day": 8,
    "reversal": "A forbidden tier 1 payment, or one made on evidence older than its "
    "limit, moves tier 1 back to shadow; each such payment goes to a person to reconcile",
}

# Exercise 5: the filled requirement the exercise prints, for the reader to diagnose.
EXERCISE_5 = {
    **SHADOW_TO_CANARY,
    "cases": "The forty public scenarios, run once in shadow mode",
    "signed_day": 2,
}
EXERCISE_5_RUN_START_DAY = 5
