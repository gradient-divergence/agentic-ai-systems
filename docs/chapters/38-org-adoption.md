# Chapter 38: Organizational Adoption

[Companion README](../../README.md)

The chapter's code counts the work done around the agent. It holds a signed
rollout plan, reconciles a window's intake log, decision records, ledger
receipts and the payment provider's refunds, and decides whether a step widens,
holds or moves back. Every record in the demo is constructed; no deployed grant
is issued.

## Files

| What | Where |
|---|---|
| Demo | `demos/org_adoption_demo.py` (fixtures in `demos/org_adoption_fixtures.py`) |
| Code | `refund_agent/rollout/` (`org_adoption_plan.py`, `org_adoption_reconcile.py`, `org_adoption_rule.py`, `org_adoption_authority.py`) |
| Exercises | `exercises/org-adoption/` |
| Stored outputs | `book_outputs/_fnd.org-adoption.*.txt` (3 files) |
| Tests | `tests/refund_agent/test_org_adoption_rollout.py`, `tests/refund_agent/test_org_adoption_component.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.org_adoption_demo
python -m pytest tests/refund_agent/test_org_adoption_rollout.py
python -m pytest tests/refund_agent/test_org_adoption_component.py
```

Run the two test files one at a time; run together in one process, the
component test's resource guard makes the rollout tests fail.
`--write-data` rewrites the three exercise fixtures in `demos/data/`.

## What the output shows

`plan-01` prints the constructed rollout plan, signed by morgan. Step 1 covers
25% of requesters and step 2 everyone, both for tiers 1 and 2, each over a
fourteen-day window with a closing snapshot two days later. Each step names its
threshold (governed share at least 99%, no refund paid around the path), five
required records (budget, approvals, audit, queue, promotion), six owners with
their work, and the manual route. The structural fields are accepted, but
admission holds: `missing required record: org-adoption-step-1-budget`. No
predecessor record is supplied, so nothing is admitted.

The plan's rules are checked by `validate(plan, evidence)`: an audit signed
before its 30-day payment labels mature is refused, every denial is read when
fewer than 80 exist, and signatures must precede the step window.

`reconcile(intake, decisions, ledger, provider, window)` counts by the intake's
log: a request belongs to the window by its received time, and every payment made
inside the window is reconciled, including payments for older requests (those do
not enter the denominator). Pending decisions and referrals, late decisions,
payments after the window and unmatched payments stay visible; an ambiguous
payment is tagged as an exception without being attributed to a request. A
provider payment matches only an executed receipt with its audit confirmation.

`decide(...)` returns `widen`, `hold` or `move_back`. A refund paid around the
path holds. An unpaid missing decision counts against the denominator, so 99 of
100 can pass the count rule when its cause, owner and disposition are recorded.
After a hold, the next window needs a named change. Reversal conditions are a
forbidden payment, a spent budget and a tier 2 payment without proposal-bound
approval. `StageAuthority` signs steps, routes requests and withdraws authority;
a later step cannot restore withdrawn authority.

`balance-01` runs the unchanged refund gate on the text's case: a 4,000-cent
console refund that posts to the order's history at the 18:00 settlement. The
agent reads 4,000 cents, pays 4,000 more, and the reconciliation reports 8,000
refunded against 4,000 paid, an excess of 4,000 cents. When the console refund
posts at 10:30 instead, the agent reads 0 cents and denies the request.

`keys-01` holds the exercise keys: exercise 3 counts 20 received, 18 governed,
one refund paid around the path and one missing decision (90.00%, hold);
exercise 5 counts 79 of 80 (98.75%, hold), while the planted denominator of
decision records falsely reports 100.00%. Exercise 4's own case (9,000 cents
paid, a 6,000-cent console refund, a 4,500-cent request) gives 10,500 cents
refunded and 1,500 excess with delayed settlement; with early settlement the
agent reads 3,000 cents and does not pay. Exercise 6's rerun with a bank-only
refund of 2,500 cents gives 17 governed of 20 and two payments around the path:
hold. An unpaid gap of 99/100 (99.00%) widens under the count rule, conditional
on complete admission evidence and a recorded disposition.

## Exercises

`exercises/org-adoption/` has the starters for exercises 1 to 5
(`org_adoption_starter.py`) and exercise 6 (`reconcile_bank.py`), with an
reference solution. Run one test file per command:

```sh
python -m pytest exercises/org-adoption/test_org_adoption_acceptance.py
python -m pytest exercises/org-adoption/test_exercise_06.py
ORG_ADOPTION_REFERENCE=1 python -m pytest exercises/org-adoption/test_exercise_06.py
```

The untouched starters fail by design: exercises 1 to 5 on path classification,
named roles, the intake denominator, delayed settlement and the planted
denominator; exercise 6 on bank-only money and conflicting identities across
feeds. To check the reference for exercises 1 to 5, copy
`org_adoption_reference.py` over `org_adoption_starter.py` in a separate copy of
the folder. The printed fixtures are `demos/data/org_adoption_window_03.json`,
`org_adoption_record_05.json` and `org_adoption_balance_04.json`.

## What it leaves out

The full stage 6 window is not run. It needs chapter 37's demand and queue,
chapter 36's routing and audit, and the admission records of chapters 33 to 35,
so the plan holds on its missing budget record and the chapter's window counts
(1,213 requests, the tier counts, about 99.5% governed) are not reproduced here.
The rule, its persisted holds and retries, and the authority's withdrawals are
tested on constructed records only. The exercise fixtures are accounting inputs,
not a demand generator or review queue.
