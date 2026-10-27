# Chapter 18: Deep Reinforcement Learning

[Companion README](../../README.md)

The chapter's code is a set of deterministic, offline experiments in NumPy: a
frictionless cart-pole simulator, a two-hidden-layer network with per-array Adam,
replay and target-network ablations, REINFORCE with its `gamma**t` factor, a
separately fitted state-value baseline, PPO's clipped objective, and a finite
parcel depot solved both by a DQN and by backward induction. The four printed
listings (`target-01`, `dqn-01`, `pg-01`, `ppo-01`) live in the demo and run in
chapter order; the `demos/deep_rl/` package imports those same functions rather
than keeping its own copies.

## Files

| What | Where |
|---|---|
| Demo | `demos/deep_rl_demo.py` |
| Code | `demos/deep_rl/` (cart-pole, network, DQN, policy gradient, PPO, depot) |
| Exercises | `exercises/deep-rl/` |
| Figure data | `demos/data/deep_rl_figures.json` |
| Stored outputs | `book_outputs/_fnd.deep-rl.*.txt` (13 files) |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.deep_rl_demo
```

Importing the demo does no work. Every run recomputes its results from seeds,
including the long training runs, so it takes minutes rather than seconds;
stored outputs and figure data are never inputs.

## What the output shows

- `target-01`: one transition's target, TD error, half-squared loss and step;
  a table of values is the special case.
- `diverge-01`: the two-state example in which shared weights, bootstrapping and
  off-policy updates make the weight grow, and what holding the target changes.
  The maximization bias is 0.564 exactly and 0.566 in simulation.
- `runs-01`: cart-pole DQN. Five seeds (7, 11, 19, 23, 31) each get 50,000 steps
  under four conditions (full, no target network, no replay, neither), evaluated
  on the same ten starting states every 2,500 steps. Batches hold 64 transitions,
  with one update per 64 new steps under every condition (781 updates a seed);
  without replay a batch is used once and discarded. Target weights refresh every
  100 updates. Behavior and replay use separate random streams, and all settings
  are fixed before evaluation. Termination masks a DQN target; the 500-step time
  limit does not. For comparison, a random policy's mean return is 21.20 and a
  fixed rule's 41.40.
- `pg-01`: one REINFORCE step by hand; the baseline's expected contribution is
  zero.
- `variance-01` and `variance-02`: the baseline is fitted on 400 separate
  episodes and the gradient's variance measured on 2,000 others. The fitted
  state value cuts total variance to 0.615 of the variance without a baseline; a
  poor constant baseline can raise it (0.25 becomes 1.00).
- `ppo-01` and `ppo-02`: the four cases of PPO's clipped objective at a clip of
  0.2, with their slopes.
- `batch-01`: ten epochs on one frozen batch of 579 samples, clipped against
  unclipped: the share of ratios outside [0.8, 1.2], the largest ratio and the
  mean KL divergence.
- `depot-01` and `depot-02`: the parcel depot. The agent chooses a checkpoint
  from the network's own estimates and claims a saving of 377.65 a shift (91.7%)
  against the fixed rule, benchmarked on 100 training shifts. On 200 shared
  held-out shifts that checkpoint costs 1,354.72 a shift against the fixed
  rule's 414.27. Only seed 31 beats the rule, by 2.755, and its paired interval
  includes zero. The exact policy from backward induction costs 407.37, a saving
  of 6.90 with interval [5.33, 8.47]. Adoption needs all five seeds to improve,
  each with a paired 95% interval above zero, so the learned policies fail; and
  because a table solves this simulator, its exact policy stays the decision even
  if a learned one improves.
- `offer-cells-01`: chapter 17's learning-window offers (9,837 requests) counted
  by reason, amount band and earlier refund. No request has an earlier refund,
  so half of the 36 finer cells are empty.
- `keys-01`: reference solutions for the exercises.

## Exercises

- Exercise 5 reads `exercises/deep-rl/ppo-05.json`, a coding agent's cart-pole
  PPO update. Its only planted implementation error is
  `clip(ratio) * advantage` without the minimum; its batch records old
  probabilities, advantages, time weights and ratios after ten epochs of that
  update.
- Exercise 6: copy `exercises/deep-rl/starter.py` and implement `double_target`,
  so that the DQN chooses each next action with the online network and evaluates
  it with the target network. Keep every other setting fixed, run the five
  training seeds under each target on the same ten evaluation episodes, and save
  the JSON report described in `exercises/deep-rl/README.md`. Then check it:

```sh
OMP_NUM_THREADS=1 python exercises/deep-rl/acceptance.py candidate.py report.json
```

Apply the paired rule even if one seed looks better: five of five must improve,
each with an interval of episode-wise improvement above zero. Ten episodes give
only a small-sample comparison; the normal approximation is a teaching
calculation, not a deployment guarantee.

## What it leaves out

No refund data and none of chapter 4's training, validation or golden data is
used. The cart-pole and the depot are simulators, and the settings, seeds and
evaluation episodes are the chapter's choices.
