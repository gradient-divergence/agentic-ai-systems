# Chapter 36: Outcome Audits and Decision Explanations

[Companion README](../../README.md)

Chapter 36 adds two checks to the refund agent's denials. A blind outcome audit
samples each segment's denials, tests them against a signed ceiling with an
exact finite-population test, and holds a breaching denial reason until the
evidence clears it. A denial explanation is generated from the actual record
and replayed before the denial is recorded. The month, its customers and their
outcomes are constructed; tools and rules never read the constructed truth.

## Files

| What | Where |
|---|---|
| Demos | `demos/audits_explanations_demo.py` (arithmetic and record fixtures); `demos/audits_explanations_integrated_capture.py` (the canonical month, read from a stored run) |
| Code | `refund_agent/audit/` (modules `audits_explanations_*`) |
| Stored run | `demos/data/audits_explanations_run.json`, with its figure data in `demos/data/audits_explanations_figures.json` |
| Exercises | `exercises/audits-explanations/` |
| Stored outputs | `book_outputs/_fnd.audits-explanations.*.txt` (11 files) |
| Tests | `tests/refund_agent/test_audits_explanations_{explain,holds,month,rates,stage6}.py` |

The package holds the source records, the constructed month, the exact ceiling
test, the safety bound, the append-only holds, the explanations and the stage-6
component. The book prints two listings from it: the audit core
(`audits_explanations_rates.py`) and the explanation core
(`audits_explanations_explain.py`).

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.audits_explanations_demo
AUDITS_EXPLANATIONS_RUN_PATH=demos/data/audits_explanations_run.json python -m demos.audits_explanations_integrated_capture
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/refund_agent/test_audits_explanations_explain.py tests/refund_agent/test_audits_explanations_holds.py tests/refund_agent/test_audits_explanations_rates.py tests/refund_agent/test_audits_explanations_month.py tests/refund_agent/test_audits_explanations_stage6.py
```

The first demo prints six stored outputs: `identity`, `attrib`, `keys`,
`boundaries`, `release` and `checks`. The second reads the stored month run,
which came from 2,000 real candidate runs, the forty public replays and mature
payment labels, with every Part VI component selected and the scope component
outermost. It refuses a missing run, or one whose selected code no longer
matches the current code, and prints the other five: `month-01`, `audit-01`,
`power-01`, `explain-01` and `explain-02`. The variable tells it where the
stored run is.

## What the output shows

- **The month** (`month-01`). Seed 7, three segments. A decides 592 requests
  and denies 217, none wrongly; B decides 611 and denies 218, none wrongly; C
  decides 613 and denies 226, of which 19 are wrong (false positive rate
  0.0475, precision 0.916). At an illustrative $20 per wrongful denial, C's
  mistakes cost $380.
- **The audit** (`audit-01`). The signed rule (`audits-explanations-rule-1`)
  reads 80 blind samples per segment against a ceiling of 0.01, at a family
  level of 0.05 with a Bonferroni correction, using a hypergeometric ceiling
  test. Morgan, the named approver, signs it before the period's first day.
  A and B show no wrongful denial in 80 readings and no breach. C shows 5 in 80
  (Wilson range 0.027 to 0.138) and breaches, so the rule holds C's
  `already_delivered` denials: 187 go to a person, and the carrier integration
  team is notified. The fewest clean readings that would release a hold are
  114, 115 and 119 for A, B and C.
- **Power** (`power-01`). An 80-reading audit has exact power 0.952 against C's
  actual errors and none against A's and B's (they have none to find); one
  wrongful denial above the ceiling is detected with probability below 0.018.
- **Explanations** (`explain-01`, `explain-02`). One denial explained in full:
  the clause, the evidence and its age, the customer's message, the
  counterfactual that would make the request eligible (with no payment
  authorized), the review contact and deadline, and six replay checks
  (completeness, counterfactual, denial, disclosure, evidence and version), all
  passing. Across the month 661 of 661 denial explanations pass, and 5 of 5 in
  the forty public scenarios.
- **Fixtures.** `identity-01` shows why equal precision and recall force
  unequal false positive rates when base rates differ (0.089 against 0.343 for
  two constructed hospitals). `attrib-01` shows that attributions in a
  constructed linear lender sum to the log-odds gap, and that the principal
  reason depends on the reference chosen. `boundaries-01` and `release-01`
  work the finite-population test: with 239 proposals and 395 owed payments, 80
  clean readings keep the hold (the upper wrongful count is 7), and 126 clean
  readings are the fewest that release it. `checks-01` replays explicit
  two-clause, exhausted-balance, tier 2 and tier 3 records. `keys-01` checks
  the exercise answers.

The rules behind these numbers:

- Audit sampling freezes identifiers before reading outcomes. During a hold,
  held referrals count as proposed denials. The ceiling count is capped at the
  denial population, a zero entitlement denominator is unassessable, and
  pending payment labels keep the audit open.
- A clean sample lifts a hold only when the exact one-sided 95% upper harm
  bound is strictly below the signed ceiling. Wilson ranges describe the sample
  estimate; they never decide a breach or a release.
- Counterfactuals establish policy eligibility only. They keep any required
  approval and assert no execution. Expiry uses `execution_at or request.at`,
  moving the clock requires actual policy and historical source records, and a
  spent balance admits no positive amount.
- Replay checks the complete failing set and its evidence, the policy version,
  disclosure, the denial and the counterfactual.

## How it joins the agent

`audits_explanations_component.selector(rule)` registers the component through
chapter 29's extension points; the candidate names the signed rule, the
explanation generator, the complete code closure and the append-only store.
`AuditInputs` requires chapter 34's region assignments, chapter 26's register
snapshot, the matching historical evidence, chapter 35's
`may_act(subject, order_id)` disclosure check, the review tickets and deadlines,
and the hold store; a missing input refuses composition. Two stage-6 steps, the
audit hold and the explanation, run before the core records a denial. A hold
keeps the proposed denial reason in a referral, and a failed replay becomes a
logged `explanation_failed` referral. Chapters 38 and 37 wrap these steps with
authority and tier routing.

## Exercises

Exercise 5 diagnoses an appeals-based audit: `audit_starter.py` deliberately
substitutes appeals for blind readings, and `audit_report_05.json` is the
seed-11 report to diagnose. Exercise 6's `explain_starter.py` deliberately keeps
only the first failing clause and its counterfactual; complete all six replay
checks. `reference.py` is the reference solution.

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider exercises/audits-explanations/test_acceptance.py
AUDITS_EXPLANATIONS_SOLUTION=explain_starter OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider exercises/audits-explanations/test_acceptance.py
```

The reference passes. The untouched starter fails completeness and its
counterfactual on `two_clause_06.json`.

## What it leaves out

- The month is constructed, and the $20 cost is illustrative.
- Counterfactual eligibility never supplies a reviewer's or the named
  approver's decision, and an explanation's replay tests fidelity and
  eligibility, not truth or usefulness.
- The fixtures in `audits_explanations_demo.py` stand alone; they do not claim
  the month's results.
