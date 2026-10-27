# Chapter 34: Observability and Reliability

[Companion README](../../README.md)

This chapter's code adds traces, service level objectives and a signed error
budget to the refund agent, reconstructs an incident from the records alone, and
builds a silence detector for a feed that stops sending. The demo runs a tier 1
shadow control and chapter 32's cohort rehearsal over the same constructed
fortnight, in which one region of the carrier's scans falls silent for 48 hours,
and shows what the budget, the incident record and the detector make of it.

## Files

| What | Where |
|---|---|
| Demo | `demos/observability_reliability_demo.py`; `demos/observability_reliability_exercises.py` holds the exercise-fixture writer and prints one more output |
| Code | `refund_agent/observability_reliability/`: spans, reason classes, objectives, exact binomial count rules, the error budget and its monitor, restart accounting, regional watermarks, the silence detector and the incident reconstruction |
| Exercises | `exercises/observability-reliability/` |
| Stored outputs | `book_outputs/_fnd.observability-reliability.*.txt` (11 files) |
| Tests | `tests/test_observability_reliability.py`, `tests/test_observability_reliability_incident.py`, `tests/test_observability_reliability_records.py`, `tests/test_observability_reliability_exercises.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.observability_reliability_demo
python -m demos.observability_reliability_exercises
python -m pytest tests/test_observability_reliability.py
python -m pytest tests/test_observability_reliability_incident.py
```

The second command prints `prebatch-01`. Run the two test files separately; in one
pytest session, the incident tests fail. The demo's `--write-exercises` flag
rewrites the exercise fixtures. To select the component in your own run, use
`component.selector(region_finality=True)` after the data-contract and event-time
components of chapters 30 and 31.

## What the output shows

- **The budget's arithmetic** (`budget-01`), from a constructed forecast of 2,000
  tier 1 requests in 30 days and a 28-day window: about 1,867 requests and 1,120
  payments, so allowances of 37.3 untimely answers and 11.2 corrections, and 0.56
  corrections for the cohort's 56 payments. It also gives the burn-rate
  thresholds, the chance of a false page, exact binomial count rules (a window of
  17 at a 2% rate pages at 3 bad), and a latency example: one 2,100-second hang
  among 100 runs moves the mean to 22.98 seconds while the 99th percentile stays
  at 2 seconds. A signed forecast spend and the final realized ratio are separate
  checks.
- **A trace** (`trace-01`) of one cohort request, `live-r-34-1310`, paid 4,999
  cents and later corrected: the held run, the watermark's advance, the second
  run that releases it, the gate, the ledger and the correction, with causal
  links between spans. Durations are virtual and token counts are null.
- **The objectives by day** (`objectives-01`), as built and with the fix: the fix
  trades two corrected payments on day 7 for five late answers on days 4 and 5.
- **The incident record** (`incident-01`). Region 2's last scan arrived at 20:13
  on day 5 and nothing more came until a batch of 267 scans 48 hours later. The
  record finds 59 affected requests, among them the two corrected payments, 9
  claims held until after the batch, and 7 `delivery_unknown` referrals. Three
  costs are
  never added: 4,999 cents of cash lost, 7,498 cents of hypothetical shadow
  exposure (no money moved), and 2,800 cents of illustrative review. Corrections
  stand at 2 bad of 442 eligible against an allowance of 11.2, and timeliness at 0
  of 936. Every lineage record carries its bundle stamp, and chapter 33's registry
  shows no promotion or rollback. The cohort's exposure is the trigger payment
  alone; the stop paused the canary with no payment after it. The paired-denial
  check (D6) is unavailable, since the fortnight has neither paired baseline
  decisions nor mature labels. The record lists what it cannot know (the batch's
  completeness and the cause of the silence) and assigns two actions.
- **The silence detector** (`detector-01`). With three regions at 5.56 scans an
  hour and 0.1 false pages a region in a 672-hour window, the quiet threshold is
  1.89 hours: a 1-hour threshold would expect 14.4 false pages, a 2-hour one
  0.056. Replayed, it pages once, for region 2, two hours after its last arrival.
- **The fix** (`fix-01`). With regional finality the detector pages 40.15 hours
  before the first shadow payment: no corrections, five late answers, and a cohort
  that runs with no stop.
- **Losing records** (`loss-01`, `sampling-01`). A sink that drops one span in 50
  loses 351 of 17,591, and the audit-coverage check fails (25 missing). Keeping a
  trace with probability 0.1 keeps at least one of three related traces with
  chance 0.271.
- `prebatch-01` counts late scans: 21 of 2,462 before the batch, 272 of 5,600
  (4.86%) over the fortnight. `keys-01` holds the exercises' numbers, and
  `figure-01` the data behind the chapter's figures.

`Monitor(record, path=None)` takes a signed `ErrorBudgetRecord` bound to a canary,
a candidate fingerprint and a window; its optional JSON-lines file is append-only,
and a reload rebuilds the observations, payments, corrections, closed alert
windows and spend. `status()` reports each objective's eligible, bad and pending
counts, its signed and remaining allowance, and recent burn rates.
`register_budget_spent(name, callback)` fires when spend passes either allowance,
and `Monitor.for_canary(session, candidate)` connects that to chapter 32's end
transition. `register_breach(callback)` reports any forbidden or stale payment;
there is no setting to suppress it. Unanswered requests turn bad at their
deadlines without new spans; `close_windows(at)` tests each disjoint count window
once, at its close. `reconstruct` reads the evaluation and sign-off journal of
chapter 29, which says which software was reviewed but not when it was deployed,
and, given `registry=`, chapter 33's signed promotions and rollbacks.

## Exercises

From `exercises/observability-reliability/`, with the companion folder on
`PYTHONPATH`, run `python -m pytest test_acceptance.py`. Starters 1 to 5 omit
records, confuse policy referrals, use the wrong traffic window or population, or
trust a frozen audit age. The untouched `detector_06.py` fails at a market close
and misses the twenty-second requirement: change `rate_at` to give an hourly rate,
with no expected quotes outside the market session. `trace-01.json` comes from the
chapter's cohort rehearsal; the other fixtures are teaching cases. `reference.py`
is the reference solution.

## What it leaves out

- The fortnight, the silence and the forecast are constructed; the cohort pays
  into a simulated ledger.
- The rehearsal issues no admission, so no canary starts, and the full canary
  admission with the budget's end wired in is not done here.
- The records cannot say whether the batch was complete or why the region fell
  silent.
