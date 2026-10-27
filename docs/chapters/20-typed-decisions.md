# Chapter 20: Typed Decision Models and Calibrated Decisions

[Companion README](../../README.md)

`refund_agent/typed.py` declares the refund risk question, validates its answer
(a probability vector, or a declared `cannot_tell`), and maps three bands to
actions. A score below 0.080 may pay, but only if every gate condition holds; a
score through 0.80, ties included, goes to review; above 0.80 is the third band.
Review, the third band and `cannot_tell` all refer to a person, and no answer
maps to denial. An amount of $50 or more is outside the score's training
population. The agent still only drafts, a person executes, and `rules.py` keeps
its placeholder risk rule.

## Files

| What | Where |
|---|---|
| Demo | `demos/typed_decisions_demo.py` |
| Code | `refund_agent/typed.py` |
| Exercises | `exercises/typed-decisions/` |
| Exercise data | `demos/data/typed_decisions_acceptance_05.json` |
| Figure data | `demos/data/typed_decisions_figures.json`, `demos/data/typed_decisions_record.json` |
| Stored outputs | `book_outputs/_fnd.typed-decisions.*.txt` (8 files) |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.typed_decisions_demo
python -m demos.typed_decisions_demo --exercise-july
```

The demo sets its own numerical libraries to one thread. The printed listings
`typed-01`, `scores-01` and `conformal-01` run in that order without hidden
setup, and importing the module generates, fits and prints nothing. Each run
logs its reads of the two frozen months in append-only journals under
`golden-sets/` in the working directory; set `BOOK_GOLDEN_DIRECTORY` to use
another directory. The stored outputs come from a run with a fresh directory.
`--exercise-july` prints only the scripted stand-in's July diagnostics, without
consulting or freezing either month. `--write-fixtures` rewrites the exercise
report and the figure data in `demos/data/`.

## What the output shows

The demo rebuilds chapter 5's boosted score and its Platt map from seed 7's
training and validation data only, plus the class-weighted score as trained. It
then draws 36,000 new requests at seed 20 and keeps the paid tier 1 requests, by
decision date. The redrawn July calibrates the 90% marginal and class-conditional
conformal sets. August and September are frozen right after the draw, before any
threshold is set, and each is consulted once per run. These are new months with
new journals, never a reread of chapter 4's golden set or seed 7's frozen months.

- `retest-01`: August has 1,022 paid tier 1 requests with 32 wrong payments
  (0.031); September has 952 with 69 (0.072). The proposed score's Brier score
  is 0.0280 in August and 0.0602 in September, against 0.0304 and 0.0685 for the
  base rate and 0.161 and 0.168 for the class-weighted score. The retest also
  reports log loss, five equal-count bins with their decomposition residual,
  expected scores from the generator, coverage with 95% intervals, and referral
  shares with their population and window.
- `bands-01`: each band's count, mean forecast and wrong-payment rate with a
  Wilson 95% interval beside the planted risk. In August the proposed score puts
  899 requests in the pay band (wrong rate 0.0122) and 123 in review (0.1707),
  and none in the third band.
- The coverage verdict uses a 0.03 tolerance, with exact Clopper–Pearson bounds
  below 50 cases and Wilson bounds otherwise. Per-class coverage is a diagnostic
  and never decides the marginal verdict. Queue shares describe a later stage;
  at stage 3 every draft still reaches a person.
- `keys-01`: reference solutions for the exercises.

## Exercises

- Exercise 5 reads an agent's report, `demos/data/typed_decisions_acceptance_05.json`,
  that accepts a scripted stand-in as the refund risk signal. Its one planted
  error is reading marginal coverage as coverage of wrong payments. Its counts
  are measurements of a constructed scripted stand-in, with no product name or
  claim.
- Exercise 6: with the starter in `exercises/typed-decisions/`, write the
  acceptance test for `ScriptedVendorModel` on the redrawn July, as a completed
  `accept(rows, model)`. The illustrative rule, fixed before reading July,
  requires reliability at most 0.0005 and no negation gap above 1e-9. Point the
  tests at your file; the folder also holds the reference
  solution `reference_solution.py`:

```sh
TYPED_DECISIONS_SOLUTION=path/to/your_starter.py \
  python -m pytest exercises/typed-decisions/test_acceptance.py
```

## What it leaves out

The requests, months and stand-in are generated; no product is measured. The
score's population stops below $50. Passing the retest changes no authority: the
agent drafts, and a person decides every refund.
