# Chapter 4: ML Data and Evaluation

[Companion README](../../README.md)

Chapter 4 prepares the refund agent's data for learning and evaluation. The demo
writes down a label policy and relabels the requests under a later policy version,
cuts a time-safe split, freezes the golden set before any check or model, plants two
leaks and catches them, and scores the baseline and a small classifier on the golden
set through one logged read. It also computes the measures and intervals the chapter
uses. Nothing here changes the refund agent.

## Files

| What | Where |
|---|---|
| Demo | `demos/ml_data_evaluation_demo.py` |
| Code | `refund_agent/split.py` (`time_split`, `freeze`), `refund_agent/labels.py` (the label policy) |
| Exercise data | `demos/data/ml_data_evaluation_split_05.json` (exercise 5's manifest) |
| Stored outputs | `book_outputs/_fnd.ml-data-and-evaluation.*.txt` (9 files) |
| Tests | `tests/test_ml_data_and_evaluation.py` |

This chapter has no `exercises/` folder.

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.ml_data_evaluation_demo
python -m pytest tests/test_ml_data_and_evaluation.py
```

The agent and the baseline decide each request on a dataset that holds only that
request's order, shipment and policy retrieval, with the policy versions
(`one_request_data`). They decide as they would on the full dataset, which would take
minutes. Importing the module runs nothing.

## What the output shows

- `counts-01` computes the measures from the counts of a constructed bank (18 true
  positives, 6 false positives, 2 false negatives, 74 true negatives): precision
  0.75, recall 0.90, F1 0.8182, accuracy 0.92, false-positive rate 0.075. With the
  same recall and false-positive rate, precision is 0.0569 at a base rate of 0.005,
  0.1081 at 0.01 and 0.9231 at 0.5. It compares Wald and Wilson intervals: at 0 in 15
  the Wald interval collapses to [0, 0] while Wilson's is [0, 0.2039]. A half-width of
  0.05 at a rate of 0.1 needs 139 cases, about 13,900 payments at a 1% base rate.
- `counts-02` prints the figures the chapter quotes beside it: a score that never
  flags one positive in 100 has accuracy 0.99 and recall 0; the one-sided 95% bound
  at 0 errors in 15 is 0.1810; and a nominal 95% Wilson interval at n = 15 actually
  covers 0.8601 of the time at a rate of 0.01 and 0.9819 at 0.2.
- `label-01` states label policy 1 (effective 2026-01-20, final when decided) and
  relabels the 600 generated requests under policy version 2026-09-01: 70 decisions
  change (21 deny to approve, 49 deny to refer). For example r-7-0504, decided 39 days
  after delivery when the window was 14 days, becomes an approval under the later
  45-day window.
- `windows-01` cuts the split in time, with no gap because a label is final when
  decided: 398 training requests (2026-01-21 to 2026-06-30), 81 validation requests
  (July) and 121 golden requests (2026-08-01 to 2026-09-27). It also counts the
  shipments with no delivered event (399 of 2,000) and the requests that had none when
  they arrived (122 of 600). The golden window is then frozen.
- `leak-01` works on the 479 development requests only. A value check catches a
  planted copy of the label; only a time check catches `paid_amount_cents`, copied from
  the agent's audit records, because the decision writes it. The classifier's flag
  answers whether the policy forbids paying a request. With the audit field as
  written, a shuffled split agrees on 81 of 81 held-out requests; a time split on July
  agrees on 78 of 81, the same model given the field as it stood at decision time on
  77, and a model without the field on 79. The leaky input looks best offline.
- `golden-01` prints the frozen set's manifest (121 requests; 38, 34 and 49 at tiers 1
  to 3; a fingerprint) and its whole access log, one entry. The baseline makes 0
  errors on the 121; the classifier misses no forbidden payment and raises 17 false
  flags.
- `golden-02` is the per-tier table with 95% intervals. Over all 121 requests: 0
  errors (0.000 to 0.031), 83 forbidden payments, recall 1.00 (0.96 to 1.00) and
  precision 0.83 (0.74 to 0.89).
- `trace-01` follows a tier 3 golden request, r-7-0523 ($650.00): the label refers it
  for approval, the agent only prepares the packet, and the named approver decides,
  denials included.
- `exercises-01` holds the keys to exercises 3 to 6; the chapter doesn't print it.

Between the freeze and the one consultation, no check or model uses a golden request
or label.

## Exercises

Exercise 5 gives an agent-built manifest, `demos/data/ml_data_evaluation_split_05.json`,
with one planted error to find. Exercise 6 freezes the September slice of the golden
set (60 requests, its parent `golden-1`) with `freeze` and reports the baseline's
errors and the classifier's recall and precision per tier, each with a Wilson
interval. The keys are in `book_outputs/_fnd.ml-data-and-evaluation.exercises-01.txt`.

## What it leaves out

- The labels are decision labels: they repeat the generator's fictional rules, so the
  baseline's 0 errors show agreement with those rules, not that the decisions were
  right. Whether a paid claim was true is the outcome label of chapter 5.
- The counts and windows before the freeze describe the labels the golden set is built
  from; they score nothing.
