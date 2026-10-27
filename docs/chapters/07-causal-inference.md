# Chapter 7: Causal Inference and Experiments

[Companion README](../../README.md)

Would sending more refund requests to review save money? The refund history
cannot say: in its validation window no request was referred, so there is
nothing to compare. This chapter's demo identifies before it estimates: it adjusts for a confounder in a
constructed hospital, checks whether the refund history supports the question,
and then plans, sizes and rehearses a small randomized experiment instead of
claiming an effect.

## Files

| What | Where |
|---|---|
| Demo | `demos/causal_inference_demo.py` (including `ThresholdExperimentPlan`, `check_plan`, `assign` and `requests_per_arm`) |
| Exercise fixture | `demos/data/causal_inference_effect_05.json` (exercise 5) |
| Stored outputs | `book_outputs/_fnd.causal-inference.*.txt` (12 files) |
| Tests | `tests/test_causal_inference.py` |

The chapter has no exercises folder; its exercises are in the book, and
exercise 5's fixture sits with the demo's data.

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.causal_inference_demo
python -m pytest tests/test_causal_inference.py
```

Run as a script, the demo sets its numerical libraries to one thread. Importing
it runs nothing.

## What the output shows

The two printed listings, `adjust-01` and `assign-01`, run together in that
order without setup between them.

- **Adjustment** (`adjust-01`). In a constructed hospital of 1,000 discharges, a
  follow-up call lowers readmission by 0.100 in each risk group, yet the pooled
  comparison shows +0.020; standardizing over risk recovers -0.100. Seed-7
  samples of 1,000, 10,000 and 100,000 discharges then show the naive and
  standardized estimates with their sampling intervals.
- **Potential outcomes** (`numbers-01`) and **difference-in-differences**
  (`did-01`) work small constructed examples, with the assumptions each estimate
  needs printed beside it.
- **Support** (`support-01`). The history census reads payment-status metadata
  only: of 36,000 requests, 17,704 policy-payable tier 1 requests were paid and
  labeled, and 18,296 were unpaid and have no outcome label. In the validation
  window (988 requests, decisions from 1 June to 1 July 2026), no request in any
  score band was referred, so positivity fails in every occupied band.
- **Review costs** (`review-01`) price assumed reviewer errors. No reviewer
  effect is observed.
- **The plan** (`plan-01`, `plan-02`, `plan-full-01`). `ThresholdExperimentPlan`
  and `check_plan` build and check an illustrative plan: customers are assigned,
  not requests, with probability one half of a referral, for policy-payable
  tier 1 requests where $4 <= score times amount < $10. The cap is exclusive
  ($15 in exercise 6). The tier gate stays outside the randomized set, including
  extra referrals below the $4 cost edge. Signature and dates are illustrative
  fields, not authorization.
- **Sizing** (`size-01`). `requests_per_arm` sizes the experiment with exact
  normal quantiles, equal arm standard deviations, independent customers, and
  the validation volume standing for one 30-day planning month. Detecting a $2
  difference needs 982 requests per arm, which at that volume means 34 assignment
months plus 30 days. The output reports the gross expected
  wrong-payment loss and a worst-case pay-arm ceiling, not the incremental loss
  an experiment would cause.
- **Assignment** (`assign-01`). The rehearsal returns records by customer, with
  their probabilities, and never calls an agent or a tool: in the validation
  window, 59 requests fall in the set (31 pay, 28 refer, each logged with
  probability 0.5), and the 929 outside it each log probability 1.
- `figures-01` holds the data behind the chapter's figures, and `keys-01` the
  exercise keys.

The demo fits chapter 5's proposed boosted score and its map on the training
and validation windows only. It excludes later requests before reading any
label, reaches no August or September outcome, and consults no golden set.
Validation includes decisions through 1 July, whose 30-day labels are final by
1 August. Because the map was fitted on that same window, every refund figure
here is a planning figure: these numbers plan a test and test nothing.

## Exercises

Exercise 5's `demos/data/causal_inference_effect_05.json` is a constructed
effect report with one planted error; its key is in `keys-01`. `--write-fixture`
rebuilds it.

## What it leaves out

The agent is unchanged, and no experiment is run. Real repeat customers, queue
interference, changed rates, unadjudicated denials and undiscovered false claims
would need further evidence before any of these plans became a test.
