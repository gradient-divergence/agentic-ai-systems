# Chapter 17: Exploration, Bandits and Exposure Budgets

[Companion README](../../README.md)

The chapter's code chooses among three notices for a refund request (standard,
dated and referenced) and learns which one avoids a repeat contact, inside an
exposure budget. `refund_agent/offers/` separates the notices, the planted
answers, the strategies (epsilon-greedy, UCB1, Thompson sampling and a capped
learner), the exposure arithmetic, an append-only decision and maturity log, and
the replay. NumPy and the standard library implement the strategies.

## Files

| What | Where |
|---|---|
| Demo | `demos/exploration_bandits_demo.py` |
| Code | `refund_agent/offers/` |
| Exercises | `exercises/exploration-bandits/` |
| Exercise data | `demos/data/exploration_bandits_log_05.json` |
| Figure data | `demos/data/exploration_bandits_figures.json` |
| Stored outputs | `book_outputs/_fnd.exploration-bandits.*.txt` (10 files) |
| Tests | `tests/refund_agent/test_offers.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.exploration_bandits_demo
```

The two printed listings, `epsilon-01` and `cap-01`, run together in that order
without hidden setup. Importing the demo prints and writes nothing.

## What the output shows

The replay projects decision-time inputs from chapter 5's training window. It
reads no false-claim label and no August, September or golden outcome. Each
answer is used once, after its full fourteen-day window, and the response draw is
fixed by request and offer, independent of the policy. The capped learner uses
epsilon 0.3 inside a 5% coin, with separate tables by reason.

- `epsilon-01`: four answers averaging 0.3 and a fifth answer of 1 give 0.44. At
  epsilon 0.1 the probabilities of standard, dated and referenced are 0.033,
  0.933 and 0.033.
- `cap-01`: a budget of 0.01 against a baseline of 0.80 gives a cap of 0.0125
  with an unsigned floor of 0, and 0.05 with a signed floor of 0.60. A propensity floor of 0.005
  bounds the largest weight at 200.
- `trace-01`: five rounds' cumulative pseudo-regret, 2 × 0.06 + 1 × 0.02 = 0.14.
- `ucb-01`: at t = 100, UCB1's bonus is 0.6070, 2.48 times the fixed-count
  radius, and the next notice is referenced.
- `thompson-01`: each notice's posterior probability of being best, by
  numerical quadrature and by 300,000 Monte Carlo draws, which agree within
  0.001.
- `run-01`: the replay at seed 7. The learning window, April 2025 to January
  2026, holds 9,837 requests. Each reason and notice gets its no-contact rate with
  a Wilson 95% interval beside the planted true rate. The learner beats the coin
  baseline by 0.033, with an approximate 95% interval of [−0.0007, 0.0666] that
  includes zero. Cumulative pseudo-regret is 500.16, or 0.051 a request. The
  months each notice needs to reach 550 answers run from under one to 550.
- `regret-01`: twenty policy seeds, 7 to 26, on the same planted answers. Mean
  final pseudo-regret is 110.00 for epsilon-greedy, 174.41 for UCB1, 80.48 for
  Thompson sampling and 502.90 for the capped learner, with each strategy's
  allocation of not-delivered requests.
- `delay-01`: with a 0.20 repeat-contact rate and a 14-day window, counting
  silence as success or using only the answers that have arrived misstates the
  rate before day 14.
- `adaptive-01`: on adaptively collected data the plain averages, 0.432 and
  0.392, sit below the true 0.5; reweighting brings them to about 0.50.
- `keys-01`: reference solutions for the exercises.

`demos/data/exploration_bandits_figures.json` keeps the twenty seeds' paths,
means and percentiles for all four strategies, and the not-delivered allocation
and delay data. The seed bands describe policy randomness in this constructed
world.

## Exercises

- Exercise 5 reads an agent's report on a capped run,
  `demos/data/exploration_bandits_log_05.json`: 2,000 rows at a 5% cap. Its only
  planted defect is logging the learner's uncapped probabilities instead of the
  applied ones.
- Exercise 6: with the starter in `exercises/exploration-bandits/`, add an
  amount band (under $25, or $25 and over) beside the reason, and rerun the
  replay at caps of 5% and 20%. The starter leaves `amount_context` for you.
  Point the acceptance tests at your file; the demo holds a reference
  implementation:

```sh
EXPLORATION_BANDITS_SOLUTION=path/to/your_starter.py \
  python -m pytest exercises/exploration-bandits/test_acceptance.py
EXPLORATION_BANDITS_SOLUTION=demos/exploration_bandits_demo.py \
  python -m pytest exercises/exploration-bandits/test_acceptance.py
```

## What it leaves out

The run grants no autonomy and names no winner to adopt. The capped learner
learns slowly, and the interval of its gain over the coin includes zero. The
answers are planted, so regret can be computed only because this world is
simulated.
