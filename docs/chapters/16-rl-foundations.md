# Chapter 16: Reinforcement Learning Foundations

[Companion README](../../README.md)

The chapter's code builds a constructed late-parcel claim. A parcel with no scan
may be lost, and each day the desk either pays the claim or waits for tomorrow's
scan. The demo solves the finite Bellman equations, compares model-free table
learning with planning from simulated carrier follow-up logs, and checks beliefs
and reward shaping. It reads no project history, frozen set or customer.
`refund_agent/learning_case.py` writes two evidenced cases for not learning yet,
beside the unchanged agent. The policy owner decides whether a claim may wait; a
value supplies no payment authority.

## Files

| What | Where |
|---|---|
| Demo | `demos/rl_foundations_demo.py` |
| Code | `refund_agent/learning_case.py` |
| Exercises | `exercises/rl-foundations/` |
| Exercise data | `demos/data/rl_foundations_incident_05.json` |
| Stored outputs | `book_outputs/_fnd.rl-foundations.*.txt` (10 files) |
| Tests | `tests/refund_agent/test_learning_case.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.rl_foundations_demo
OMP_NUM_THREADS=1 python -m refund_agent.learning_case
python -m pytest tests/refund_agent/test_learning_case.py
```

The demo prints nine of the stored outputs; the learning-case module prints the
tenth, `case-01`. Importing either module computes and prints nothing.
`--write-data` rewrites the chapter's plot data and the constructed exercise
report in `demos/data/`.

## What the output shows

- `model-01`: the constructed parcel. It is lost with probability 0.100 and
  delivered with probability 0.600 a day; the claim is $49.99, a day of waiting
  costs $3.00, payment is compulsory on day 10, and the discount is 1. With no
  scan, the belief that the parcel is lost rises from 0.100 on day 0 to 0.916 on
  day 5.
- `values-01` and `vi-01`: the best rule waits on days 0 to 4 and pays from
  day 5, at an expected claim cost of $11.41. Value iteration last changes at
  sweep 11. A one-stage rule finds the same pay day for every amount and waiting
  price in the table.
- `stopping-01`: a counterexample. When scans rise and then fall, the one-stage
  rule pays on day 0 at cost 10.00, while the optimal rule waits at cost 3.00.
- `td-01`: one temporal-difference update and one Q-learning update by hand,
  nonterminal and terminal.
- `learn-01`: live Q-learning on seeds 0 to 19, 1,000 claims each, with a random
  action one time in ten. On days 0 to 4 all twenty seeds stay greedy on the
  planned rule, yet none of the twenty final greedy rules matches the whole rule,
  and the median sampled cost above the planned rule is 3,736.56 over 1,000
  claims. Planning from 200 pay-at-once claims with a 10-day carrier follow-up
  costs 0.096 extra a claim on average.
- `state-01`: with two carriers mixed half and half, the best rule that sees only
  the day count pays on day 10 and costs $13.98 a claim, 54 cents more than a
  rule that knows the carrier.
- `reward-01`: reward shaping. A zero waiting price pays on day 10; three
  potential-based shapings keep the day-5 rule; a flat waiting bonus of 1.00
  moves it to day 6.
- `case-01`: the two learning cases, each asked as five questions (sequential,
  feedback, choice, safety, value). The refund agent should not learn approve,
  deny or refer: every answer is no. Whether a late-parcel claim is payable is
  sequential, but its feedback, choice and safety are unknown and its value is
  no. Both verdicts are "not yet", with autonomy unchanged: the agent drafts and
  a person executes.
- `keys-01`: reference solutions for the exercises.

## Exercises

- Exercise 7 reads an agent's report on a policy for closing IT incidents,
  `demos/data/rl_foundations_incident_05.json`, which has one planted reward
  error. Its score curve is an exact policy-mixture illustration, not measured
  training.
- Exercise 8 asks for a `LearningCase` on whether the refund agent should learn
  its tier 1 threshold from outcomes. Implement `submission()` in
  `exercises/rl-foundations/starter.py` to return the case and its rendered
  table. The checker resolves every source locator, validates the JSON round
  trip, the verdict and the unchanged execution authority; a person must still
  check that each cited passage supports its answer. The starter fails on
  purpose; `reference_solution.py` is the reference solution.
- `practice.md` holds optional practice; its numerical key is in
  `book_outputs/_fnd.rl-foundations.keys-01.txt`.

## What it leaves out

The parcel, its carriers and its costs are constructed. Carrier arrivals are
independent of payment, and customers' reactions to waiting are absent. No
historical window is read. Agreement on days 0 to 4 does not mean agreement with
the whole rule. No learned value changes eligibility, approvals or who executes
a refund.
