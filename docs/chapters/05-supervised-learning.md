# Chapter 5: Supervised Learning and Calibrated Risk

[Companion README](../../README.md)

Chapter 5 learns the probability that paying a refund is a mistake. The demo trains a
risk score on a generated refund history whose label is an outcome: a paid claim found
false within 30 days, a *wrong payment*. It fits a logistic regression and boosted
trees, calibrates them, prices each action, and judges the proposed score once on
August and once on September, the month after a policy change. Nothing here changes
the refund agent, whose `risk_score` is still the text rule in `refund_agent/rules.py`.

## Files

| What | Where |
|---|---|
| Demo | `demos/supervised_learning_demo.py` |
| Code | `refund_agent/data/history.py` (the refund history), `refund_agent/model/history_baseline.py` (a separate time-split baseline), `refund_agent/split.py` (`freeze`) |
| Exercise data | `demos/data/supervised_learning_recalibration_07.json` (exercise 7's report) |
| Stored outputs | `book_outputs/_fnd.supervised-learning.*.txt` (21 files) |
| Tests | `tests/test_supervised_learning.py`, `tests/refund_agent/test_history.py`, `tests/refund_agent/test_history_baseline.py` |

This chapter has no `exercises/` folder.

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.supervised_learning_demo
python -m pytest tests/test_supervised_learning.py tests/refund_agent/test_history.py tests/refund_agent/test_history_baseline.py
```

The demo sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS`
to 1 itself, so every run prints the same digits. Importing it runs nothing.
`--write-fixture` rewrites exercise 7's report. The tests refit the models, so they
take a minute or more.

## What the output shows

The history holds 36,000 generated requests, of which the policy paid 17,704, all at
tier 1. Every window is cut by the day a label became final, so nothing fitted uses a
label still open on 1 August 2026, the day the score starts work.

- `split-01` counts the windows: training 13,766 paid requests, validation 988, an
  open pool of 985 whose labels were still open, test 999 (August) and shifted 966
  (September). Before any model is fit, August and September are frozen with
  `refund_agent.split.freeze`: each gets a manifest, a fingerprint and an access log,
  and a held-out request is reached only through a logged read.
- `fit-01` and `fit-02` give log losses beside the base rate's: on validation, 0.1830
  for the base rate, 0.1685 for the logistic regression and 0.1683 for the boosted
  trees. Wrong payments are 3.62% of training and 4.45% of validation.
- `boost-01` builds three one-split trees by hand, each fit to what the trees before
  it left: the squared error falls from 16 to 4 to 1.
- `biasvar-01` estimates bias and variance of three models at two fixed requests,
  over training sets redrawn from the generator's planted risk; one deep tree has the
  largest variance.
- `decide-01` computes the expected cost of each action, first for three incident
  tickets and then for a refund at risk 0.12. A review is cheapest between 0.080 and
  0.800 for a $49.99 refund, and the band narrows as the amount falls.
- `calibrate-01` measures calibration error on validation and fits a two-parameter map
  there: the boosted trees' error is 0.0132, the class-weighted score's 0.3201.
- `august-01` prints August's manifest and its one consultation, after every fit.
  `record-01` gives 37 wrong payments among 999 requests (0.027 to 0.051) and four
  scores at the tier threshold; `record-02` the proposed score's calibration by score
  bin and by amount.
- `gate-01` compares three drafts on August. The placeholder rule pays all 999 at a
  cost of 1.07 per request. The tier threshold of 0.080 refers 119 and costs 0.80; the
  review-cost rule ($4 review) refers 59 and costs 0.67. `sweep-01` prices every
  cut-off and `bootstrap-01` resamples the saving 2,000 times: the threshold saves
  $0.27 a request against the placeholder, with a 95% interval from -$0.01 to $0.59.
- `shift-01` corrects a probability for a changed base rate and for a class weight;
  `shift-02` makes the one consultation of September: 966 requests, 63 wrong payments,
  calibration error 0.0231 for the proposed score and 0.0181 after the correction.
- `trace-01` to `trace-03` follow one request through the fitted models to its drafts,
  with its later outcome kept apart. Promotion granted: none; the score drafts, and a
  person executes.
- `numbers-01`, `keys-01` and `figures-01` hold the arithmetic the chapter quotes, the
  keys to its exercises and the data behind its three figures.

"The proposed score" is the boosted trees followed by their recalibration map.

## The refund history and its separate baseline

```sh
python -m refund_agent.data.history --seed 7
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m refund_agent.model.history_baseline --seed 7
```

`generate_history(seed=7, rows=36000)` returns typed rows; the first command writes
them to `refund-history/history.csv` (`--rows` and `--out` override the defaults), and
the second writes `refund-history/baseline.json`. Git ignores both. Requests span
2025-04-01 through 2026-09-30. The history is independent of the refund agent's
generated data: the 600 labeled requests and forty public scenarios are unchanged.

The history's ledger pays only policy-payable tier 1 claims. `found_false_30d` is 0 or
1 only for paid claims, final 30 days after the request (`label_final_at`); both
fields are empty for unpaid claims. A zero means nothing was found within that window,
not that the claim was true. `FEATURE_COLUMNS` is the model's input allowlist, built
only from the request and the order and shipment evidence available before the
decision; payment status, the policy decision, identifiers, timestamps and outcome
fields are excluded. The signal comes from the amount, the reason, the delivery age,
prior refunds and a damaged-claim and prior-refund interaction; the original payment
and the source ages are noise. Conditional risk doubles on 2026-09-01, with feature
generation unchanged. The target bands, 1.5% to 4% before the drift and 3% to 8%
after, are constructed teaching rates, not merchant statistics.

The baseline trains on the first 12 months, purging the final 30 days whose labels are
not final at its 2026-04-01 cutoff, and tests on the final six months. It fits an
unweighted logistic regression and shallow histogram gradient boosting, with no tuning
on the test window and no recalibration. Predictions below 0.080 pay; higher ones are
referred for an illustrative $4 review that is assumed perfect. A false claim paid
costs its amount; a wrong denial would cost $20, but the score has no authority to
deny. The JSON reports predicted expected cost and realized cost separately.

## Exercises

Exercise 7 hands you a report, `demos/data/supervised_learning_recalibration_07.json`,
in which an agent recalibrated the class-weighted score with isotonic regression and
reported a calibration error of 0.0000. It holds one planted error; the key is in
`book_outputs/_fnd.supervised-learning.keys-01.txt`.

## What it leaves out

- The score is demonstrated offline, on requests the policy paid. Only a paid request
  has an outcome; a referral's outcome and cost are never recorded.
- The costs are illustrative and await the decision owner's signature; the history and
  its rates are constructed.
- The refund agent keeps its placeholder text rule; no score here drafts or pays a
  live refund.
