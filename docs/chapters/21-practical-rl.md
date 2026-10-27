# Chapter 21: Practical RL and Off-Policy Evaluation

[Companion README](../../README.md)

The chapter's code judges a new policy from another policy's log.
`refund_agent/offpolicy/` implements the four estimates (inverse propensity
scoring, its self-normalized form, the direct method and the doubly robust
estimate), paired improvement terms, normal and delta-method intervals,
full-distribution support tables, smoothed reward models fitted on learning data
only, an adapter for a frozen log, a rule declared before the data is read, and
the JSON evaluation record. A zero sum of ratios gives an effective size of zero
and an undefined self-normalized estimate. Raw estimates can be computed without
support, but the record then marks the policy not judged and holds it.

## Files

| What | Where |
|---|---|
| Demo | `demos/practical_rl_demo.py` |
| Code | `refund_agent/offpolicy/` |
| Exercises | `exercises/practical-rl/` |
| Exercise data | `demos/data/practical_rl_reports_05.json` |
| Stored outputs | `book_outputs/_fnd.practical-rl.*.txt` (8 files) |
| Tests | `tests/refund_agent/test_offpolicy.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.practical_rl_demo
```

The printed listings `bank-01`, `dr-01` and `refund-01` run together in that
order without hidden setup. Importing the demo reads, prints and writes no
chapter data. The bank, box and exercise arithmetic use fixed counts.
`--write-data` rewrites the coverage and support figure data, the evaluation
record and the constructed exercise 5 reports.

## What the output shows

- `bank-01`: a bank's 80 text alerts and 20 calls. The logged mean is 0.640;
  the inverse-propensity and self-normalized estimates of the target policy are
  both 0.720, from an effective sample of 50. The improvement of 0.080 has the
  interval [−0.091, 0.251], so the log cannot yet tell. The bank passes the rule
  at 1,000 alerts and is held at 100.
- `box-01`: three records with ratios 2, 1 and 0.5 and rewards 1, 0 and 1 give
  an inverse-propensity estimate of 0.833, a self-normalized one of 0.714 and an
  effective size of 2.333.
- `dr-01`: with a reward model of 0.55 for texts and 0.85 for calls, the direct
  method gives 0.730, its correction −0.010, and the doubly robust estimate
  0.720.
- `coverage-01`: a simulated bank with a true value of 0.735, 500 seeded logs of
  2,000 alerts per logging chance, each pooled reward model fitted on its own
  independent training log. As the logging chance falls from 0.5 to 0.01, the
  inverse-propensity, self-normalized and doubly robust means stay near 0.735,
  the direct method drifts to 0.713, and the median effective size falls from
  1,923 to 107.
- `edge-01`: 1,000 records of which only 5 match the candidate, all answered.
  Both estimates are 1.000, the self-normalized interval collapses to
  [1.000, 1.000], and the rule holds the candidate because its effective size of
  5 is below 400.
- `horizon-01`: 1,000 student-support trajectories. Trajectories matching a
  candidate fall from 523 at one step to 73 at four and none at ten.
- `refund-01`: the refund judgment. Two candidate offer policies are judged
  against the fixed offer rule on chapter 17's offer log, by a rule recorded
  first: doubly robust, 95% interval, an effective-size floor of 400 and an exact
  level of 0.025. The reward model is fitted on the 9,837 learning decisions
  (April 2025 to January 2026) only, and the 3,929 evaluation decisions (February
  to May 2026) have their outcomes consulted once. Before that consultation the
  rule, policies and model are recorded in
  `demos/data/practical_rl_evaluation.access.jsonl`. Candidate A keeps the
  exported policy, including its unfavorable choice for not-delivered requests;
  B changes only that segment. Adapters keep the upstream policy fingerprints
  beside their own. Both are held: A has only 55 effective records, and B has
  1,309.95 but its improvement interval and its randomization check fail.
- `keys-01`: reference solutions for the exercises.

The randomization check keeps the reward model fixed and redraws the applied
actions under the sharp null of no action effect. It enumerates every redraw
when at most twenty rows affect the statistic, and otherwise uses 100,000 seed-7
redraws, a Monte Carlo approximation whose uncertainty is in the JSON record.
It holds a sparse, balanced five-answer example despite that example's favorable
normal interval.

## Exercises

- Exercise 5 reads `demos/data/practical_rl_reports_05.json`: two agents'
  reports on a constructed offer window whose logger sent not-delivered requests
  only the standard notice.
- Exercise 6: `exercises/practical-rl/candidate_c.json` sends the referenced
  notice to not-delivered requests and the standard notice elsewhere, fixed
  before the frozen window is read. Complete `candidate_c(frozen_at)` in
  `starter.py` to return that policy. The provided evaluator writes the rule
  before consulting outcomes once, delivers
  `candidate-c-output/candidate-c-record.json` and
  `candidate-c-output/candidate-c-output.txt`, and follows the rule's decision.
  The candidate has support; the exercise does not promise that it will pass.

```sh
OMP_NUM_THREADS=1 python exercises/practical-rl/starter.py
python -m pytest exercises/practical-rl/test_acceptance.py
```

The untouched starter fails with `NotImplementedError`. The tests check the
reader's starter by default; set `PRACTICAL_RL_SOLUTION` to another file's path
to check it. They check Candidate C's policy before the read, the rule's order,
the full support table, every estimate, the interval, the decision and both
delivered files. An identical second evaluation reuses the first read's receipt
without changing any bytes.

## What it leaves out

The judgment creates no new refund history or evaluation log, and reads none of
chapter 4's golden or validation data. The bank, the box and the student log are
constructed. Holding a candidate is the rule's decision, not a verdict that it is
worse, and the agent still only drafts.
