# Chapter 37: Human Oversight and Risk Management

[Companion README](../../README.md)

Chapter 37 sizes and runs the human review that the refund agent's tier 2 and
tier 3 decisions require. The code models the review desk as a queue,
simulates a constructed month of arrivals against several staffing designs,
checks each design against a signed late ceiling month by month, plants
corrupted packets to test whether reviewers still read them, and defines the
review ticket's lifecycle: deadlines, escalation, notices and renewal. Arrivals
are seeded and reviewer behavior is planted; no real reviewer data is used.

## Files

| What | Where |
|---|---|
| Demo | `demos/human_oversight_risk_demo.py` |
| Code | `refund_agent/oversight/` (`human_oversight_risk_*`: queueing, demand, lifecycle, deadline, simulate, design, controller, monitor, desk, component, screen, partners) |
| Exercises | `exercises/human-oversight-risk/` |
| Stored outputs | `book_outputs/_fnd.human-oversight-risk.*.txt` (11 files) |
| Tests | `tests/refund_agent/test_human_oversight_risk_{component,lifecycle,math,simulation}.py` |
| Supplement | `supplement/human-oversight-risk-eu-ai-act.md`, a reading aid to the EU AI Act's scope, transitions and sector rules (checked October 2026; not legal advice) |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.human_oversight_risk_demo
python -m refund_agent.oversight.human_oversight_risk_queueing
python -m refund_agent.oversight.human_oversight_risk_deadline
python tests/refund_agent/test_human_oversight_risk_lifecycle.py
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/refund_agent/test_human_oversight_risk_component.py tests/refund_agent/test_human_oversight_risk_lifecycle.py tests/refund_agent/test_human_oversight_risk_math.py tests/refund_agent/test_human_oversight_risk_simulation.py
```

The demo prints seven stored outputs: `arrivals`, `queue`, `absence`, `catch`,
`sample`, `keys` and `wait-data` (the JSON behind the waiting-time figure). The
two modules print `mm1-01` and `timeout-01`. The lifecycle test file, run as a
script, prints `review-records-01` and `screen-01` from test-constructed
records. In `arrivals-01`, the candidate fingerprint line can differ from the
stored file; the other lines should match.

## What the output shows

- **The queue model** (`mm1-01`). Textbook M/M/1 results in hours: a reviewer
  with one arrival an hour and two services an hour is busy half the time, a
  request spends 1 hour in the system on average, and 1.8% take longer than 4
  hours. At 1.8 arrivals an hour the mean is 5 hours and 44.9% take longer than
  4. When arrivals outpace service there is no steady state.
- **Demand** (`arrivals-01`). A seeded 30-day stream (seed 7) of Poisson
  arrivals around the clock: 1,998 tier 1, 482 tier 2 and 108 tier 3 requests.
  Tier 1 requests run through the actual agent (960 approvals, 268 referrals,
  770 denials), giving a referral share of 13.41% and a payment share of
  48.05%. Tier 2 reviews (112 a week at 15 minutes), tier 3 reviews (28 a week
  at 30 minutes), referrals, rechecks, blind second reviews and planted cases
  offer the reviewers 43.19 hours of work a week. The stream never generates or
  reads labels. The shares come from the as-built candidate and serve as a
  sensitivity input, not as evidence about the fully assembled agent.
- **Staffing designs** (`queue-01`). Each design runs 100 seeds (7 to 106) over
  the same 30-day calendar from 1 October, with fixed and with exponential
  service times. The signed late ceiling binds every month on its own, so a
  good pooled share does not rescue a failing month:

  | Design | Hours a week | Breaching months, fixed | Breaching months, exponential | Accepted |
  |---|---|---|---|---|
  | D1 | 80 | 100 of 100 | 100 of 100 | no |
  | D2 | 80 | 97 of 100 | 94 of 100 | no |
  | D3 | 88 | 0 of 100 | 1 of 100 | fixed only |
  | D4 | 96 | 0 of 100 | 0 of 100 | yes |
  | D4 with rework | 96 | 0 of 100 | 0 of 100 | yes |

  D4 is a revised constructed staffing: the same named approver, Morgan, also
  works Saturday 09:00 to 17:00 UTC (48 hours a week) and handles the decision
  owner's follow-ups, and Morgan's overlapping roles share one capacity. The
  rework stress adds proposal preparation, renewed review, refusal escalation
  and blind adjudication. A simulated pass does not sign a deployment grant.
- **An absence** (`absence-01`). With Morgan away five working days and no
  deputy, 27 held tier 3 items all miss their original deadlines and escalate.
  After the return, the absence cohort clears in 1.73 working days at 8 hours a
  day and 3.38 at 4 hours a day. A deputy needs the corresponding policy
  version, so a deputy's decision means a new proposal and review.
- **A deadline** (`timeout-01`). A $120 tier 2 request arrives Friday 18:20 UTC
  while the reviewer is away Monday. At Monday 17:00 it escalates to Morgan
  with a new due time, Tuesday 13:00, and the customer is notified. The
  deadline records no approval, denial or payment; the item stays open.
- **Attention** (`catch-01`). A planted inattentive reviewer catches corrupted
  packets (a contradictory delivery scan, a payment one cent over the remaining
  balance, a false return-window denial) with chance 0.9 at five or fewer
  waiting, 0.7 at 6 to 15 and 0.4 above 15. Only the busy band, 6 to 15
  waiting, enters the alarm. Over 1,000 replications of D3 (seeds 7 to 1006),
  18% alarm by week 2 and 769 stay undetected by day 30; undetected runs are
  right-censored. Weekly looks repeat the test, so 5% is not a lifetime
  false-alarm bound.
- **Samples** (`sample-01`). Of 960 tier 1 payments in the month, a 25% recheck
  samples 227 and finds no policy-ineligible payment (95% upper bound 1.31%); a
  10% recheck samples 92 (bound 3.20%). Blind second readings agree with the
  adjudicated policy reading in 52 of 56 cases. These are labeled sensitivity
  results; constructed demand expectations are reported separately.
- **The review screen and records** (`screen-01`, `review-records-01`). One tier
  2 denial ticket as a reviewer sees it, and the records of fresh and stale
  execution, waiting expiry, refusal, escalation, notices, and policy and
  deputy renewal, including payments and confirmed denials.

The simulation keeps role constraints, absences, pauses at closing, independent
blind reviewers, deadlines and a finite calendar. It drains the month's
arrivals for fourteen more days to classify deadline outcomes, stops occupancy
at month close, censors unfinished stays there and counts the starting backlog
explicitly. Its conservation check is a finite-window identity, not a
steady-state Little's law.

## The review desk

`ReviewDesk` needs a full reviewer-authentication callback and chapter 36's
generated and replayed denial explanation. Authentication returns a
`ReviewerContext` only after chapter 35 verifies the credential's issuer, seal,
kind, scope, subject and role, time and revocation; the explanation arrives as a
`ProposedExplanation`, a required view of the actual record. Approving a denial
confirms it and pays nothing. Approving a payment calls the authenticated
`record_approval(approval, credential=..., at=...)` and the existing execution
gate. A stale read or expired eligibility escalates with its execution refusal,
and a policy change renews the proposal and its review. Tickets and their
events are in-memory teaching records, as in chapter 25. Shadow tickets are
hypothetical: recording a human outcome on one is refused, and the live review
and payment stores stay unchanged. The component registers
`human-oversight-risk.review@1` through chapter 29's release extension points.

## Exercises

Exercises 1 to 5 are a worksheet, `human_oversight_risk_answers_01_05.py`, whose
five answers carry deliberate mistakes: confusing accumulated work with
payments, ignoring the weekend, reporting service time as total waiting time,
sizing the insurer for seven staffed days, and counting escalations as catches.
Exercise 5's `log-05.csv` is a constructed week of 99 review items with 20
planted cases, ending with "Planted cases caught: 15 of 20 (75%). Review
healthy." Exercise 6's `timeout_06.py` writes an approval at the deadline; the
fix keeps the item open, escalates it and sends the new deadline.

```sh
python -m pytest -p no:cacheprovider exercises/human-oversight-risk/test_human_oversight_risk_answers_01_05.py
python -m pytest -p no:cacheprovider exercises/human-oversight-risk/test_timeout_06.py
```

Untouched, the first fails five tests and the second fails one.
`HUMAN_OVERSIGHT_RISK_REFERENCE=1` checks the reference solutions in
`reference/` instead.

## What it leaves out

- No credential issuer or explanation generator ships; the test credentials and
  explanations are confined to the tests, so the review screen and records are
  test-constructed, and a missing partner is refused with no unauthenticated
  retry.
- Counts for the fully assembled agent are pending; the month here measures the
  as-built candidate.
- Chapter 32's autonomous canary still limits payments to tier 1.
