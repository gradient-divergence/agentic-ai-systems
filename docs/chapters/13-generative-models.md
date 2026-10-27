# Chapter 13: Generative Models and Simulation

[Companion README](../../README.md)

This chapter's code works through sampling, likelihood, the evidence lower bound,
a two-point GAN and diffusion steps with small exact examples. It then builds a
scenario generator for the refund agent: a model counted on training requests
that proposes test cases, which an independent policy function labels.

## Files

| What | Where |
|---|---|
| Demo | `demos/generative_models_demo.py` |
| Code | `refund_agent/scenarios/` (tokens, model, checks, suites) |
| Exercises | `exercises/generative-models/` |
| Stored outputs | `book_outputs/_fnd.generative-models.*.txt` (11 files) |
| Tests | `tests/refund_agent/test_scenarios.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.generative_models_demo
python -m pytest tests/refund_agent/test_scenarios.py
```

The numerical experiments need NumPy only. No network, model service or trained
image denoiser is involved, and importing the demo prints, fits and writes
nothing. `--write-data` writes the scenario suites, plot data and exercise 6's
report to `demos/data/generative_models_*.json`.

## What the output shows

Small exact examples:

- `sample-01`: five scored tokens under greedy decoding, temperatures 0.5, 1
  and 2, top-2 and nucleus 0.8 sampling, with the chance each gives the token
  "certificate".
- `likelihood-01`: the scenario model's add-one-smoothed conditionals, counted
  on 395 failure-free training requests (179 distinct cases); a common case has
  probability 0.0083 and a rare one 0.00027.
- `elbo-01`: one data point where the bound is -2.419, the log-likelihood
  -2.266, and the gap, 0.153, is the KL divergence from the encoder to the true
  posterior.
- `vae-01`: a linear Gaussian VAE with analytic expectations and bound
  gradients on 512 seeded two-dimensional points (seed 7). It records the
  expected log-likelihood, KL to the prior, bound, exact log-likelihood and gap;
  by step 2,000 the gap is 0.000024.
- `gan-01`: a two-point GAN with its best discriminator. With the generator at
  (0.8, 0.2), the objective is -1.285 against a floor of -1.386 and the
  Jensen-Shannon divergence is 0.051; when the generator matches the data,
  both reach the floor and zero.
- `diffusion-01`, `schedule-01` and `reverse-01`: one forward noising step, the
  jump to step 3, a linear schedule of 1,000 steps (beta 0.0001 to 0.02) and one
  illustrative reverse step with a given noise prediction; no network is
  trained.
- `keys-01` and `keys-gan-01`: answers to the exercises.

The refund scenarios (`scenarios-01`):

`refund_agent/scenarios/` counts conditionals on the 395 failure-free requests in
chapter 4's training window, before 1 July. Its six tokens are reason, delivery,
delivery-age band, amount, earlier refund and proposed decision. The declared
parents exclude the date, so generation keeps the training policies' habits.
Only request times and ids are read to select the window; no later feature or
label is fitted, and no golden set is consulted. Each kept row carries a label
from the independent policy function, and keeps the generator's decision
separately. A schema rejection is recorded with its draw number.

For 15 September, under the 1 September policy:

| Set | Proposed | Rejected | Overruled | Distinct | Unseen in training | Kept |
|---|---|---|---|---|---|---|
| Listed | 252 | 0 | 0 | 252 | 73 | 252 |
| Temperature 1 | 200 | 4 | 88 | 121 | 23 | 196 |
| Temperature 2 | 200 | 28 | 101 | 121 | 29 | 172 |

The listed set labels all 252 valid cases. Kept draws may repeat a case, so the
distinct count, over the five case tokens, is reported separately. "Overruled"
counts proposed decisions the policy function changes. Neither count is a
traffic pass rate or a release decision. Human review is pending:
`review_sample` selects five kept draws (seed 31) for a person to compare with
the policy text; selecting them is not a review. Each suite's JSON records its
date, policy, sampling parameters, fingerprints, rejection reasons and check
counts. `ScenarioSet.read` checks the fingerprint and provenance, and
`regenerate(manifest)` must reproduce the suite byte for byte.

## Exercises

Exercise 6 audits `demos/data/generative_models_scenarios_06.json`, a report
dated September that cites the wrong policy version. All 196 kept rows cite the
July policy; under September, 22 labels change (9 denials become approvals and
13 become referrals), and 12 of the wrong labels agree with the generator.

Exercise 7, in `exercises/generative-models/`, adds a seventh token: no failure,
or one of the five injected failures, which the policy label turns into a
referral. Complete `starter.py`, then run its acceptance test and the unchanged
scenario tests:

```sh
GENERATIVE_MODELS_SOLUTION=path/to/your_solution.py \
  python -m pytest exercises/generative-models/test_acceptance.py
python -m pytest tests/refund_agent/test_scenarios.py
```

The starter raises `NotImplementedError` on purpose and is outside the default
test run. `reference.py` is the reference solution, `test_reference.py` its
regression checks, and `run-07.json` with the two `manifest-tau*.json` files the
record of its run (at temperature 1: 200 proposed, 4 rejected, 89 overruled,
121 distinct, 196 kept).

## What it leaves out

The scenario counts measure coverage of constructed cases, not traffic, and
grant no release authority. The human review of sampled rows has not happened.
The diffusion and GAN examples are exact toy calculations; no image model or
denoiser is trained.
