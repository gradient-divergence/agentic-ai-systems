# Chapter 19: RLHF and Preference Alignment

[Companion README](../../README.md)

The chapter's code uses a constructed telecom desk, with no language model and
no external service: five kinds of fault report, each with two possible causes,
and a policy that names a cause with a confidence. The demo turns comparisons
into rewards,
solves the KL-regularized objective exactly, trains by direct preference
optimization (DPO), and then checks what preference accuracy leaves unanswered.

## Files

| What | Where |
|---|---|
| Demo | `demos/preference_demo.py` |
| Exercises | `exercises/rlhf-alignment/` |
| Exercise data | `demos/data/rlhf_alignment_report_05.json` |
| Figure data | `demos/data/rlhf_alignment_figures.json`, `demos/data/rlhf_alignment_delivered.json` |
| Stored outputs | `book_outputs/_fnd.rlhf-alignment.*.txt` (6 files) |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.preference_demo
```

The three printed listings, `pair-01`, `optimum-01` and `dpo-01`, run in that
order without hidden setup. `--write-data` rewrites the figure data and the two
constructed agent reports in `demos/data/`.

## What the output shows

- `pair-01`, `optimum-01` and `dpo-01`: stable Bradley–Terry and DPO losses,
  token log-probabilities summed into a response's, and the exact solution of the
  finite KL-regularized objective.
- `train-01`: seed 7, with four independently seeded windows of 2,000 records
  per kind (labeled reports, training comparisons, held-out reports, held-out
  comparisons). Labeled training reports fit the frozen reference by log loss;
  training comparisons fit the reward gaps or DPO. Held-out comparisons measure
  preference accuracy, with ties against the reference worth one half; held-out
  outcome labels measure correctness and calibration, one bin per kind. DPO at
  β = 1 and β = 0.25 raises preference accuracy from 0.500 to 0.739 and leaves
  correctness at 0.7356, while calibration error grows from 0.0128 to 0.112 and
  0.218. Honest preferences raise confidence without changing the most likely
  cause.
- `delivered-01`: an agent's report on the β = 0.25 model, checked claim by
  claim. Preference accuracy rose, but correctness did not. The 6,000 reports
  above 0.9 confidence average 0.999 confidence and are correct only 0.8225 of
  the time, so "replies above 0.9 can go without review" fails. The KL penalty
  (mean KL 0.195) prices distance and forbids nothing. No credit promised in 300
  test drafts still allows a credit rate up to 0.0099 at 95%; permission belongs
  at execution.
- `keys-01`: reference solutions for the exercises.

## Exercises

- Exercise 5 reads a constructed hospital report,
  `demos/data/rlhf_alignment_report_05.json`. Its one evaluation defect: it calls
  the model calibrated after selecting only messages labeled urgent.
- Exercise 6: with the starter in `exercises/rlhf-alignment/`, train the toy by
  DPO at β = 0.5, then fit and check a calibration map on separate generated
  windows. `RLHF_ALIGNMENT_SOLUTION` selects the file to test; the reference
  solution is `exercise_six` in `demos/preference_demo.py`.

```sh
RLHF_ALIGNMENT_SOLUTION=path/to/your_starter.py \
  python -m pytest exercises/rlhf-alignment/test_acceptance.py
RLHF_ALIGNMENT_SOLUTION=demos/preference_demo.py \
  python -m pytest exercises/rlhf-alignment/test_acceptance.py
```

The untouched starter fails. The tests accept the fitted-sample optimum, which
passes the logit check. The report also tests the population formula against a
0.02 tolerance and prints FAIL for seed 7, because sampling noise makes its
largest logit gap 0.310249; the tests confirm that diagnosis without changing
the samples or hiding the failed check.

## What it leaves out

There is no language model: the desk, its kinds and its labels are constructed.
No chapter 4 frozen, golden or validation data is read. Training grants no new
permission; a tool such as a credit stays gated at execution.
