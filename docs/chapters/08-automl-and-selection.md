# Chapter 8: Delegated Model Building

[Companion README](../../README.md)

What should you check when someone else, a search procedure or an agent, builds
the model for you? This chapter's demo commissions candidate models for chapter
5's refund-risk score, inspects each delivery, and lets a single frozen month
decide under a rule fixed in advance. The baseline survives.

## Files

| What | Where |
|---|---|
| Demo | `demos/delegated_model_demo.py` (the commission, inspections and selection record) |
| Search lab | `demos/delegated_search_lab.py` (grid, random and model-based search; stacking; input selection) |
| Constructed agent pipeline | `demos/delegated_model_pipeline.py` |
| Commissions and selection records | `refund_agent/selection.py` |
| Exercise fixtures | `demos/data/delegated_model_agent_delivery.json`, `delegated_model_agent_clean.json`, `delegated_model_delivery_05.json` |
| Stored outputs | `book_outputs/_fnd.automl-and-selection.*.txt` (16 files: 11 from the demo, 5 from the lab) |
| Tests | `tests/test_delegated_model.py`, `tests/test_delegated_search_lab.py` |

## Run it

From the companion folder, after the install in the README:

```sh
BOOK_GOLDEN_DIRECTORY="$(mktemp -d)" python -m demos.delegated_model_demo
python -m demos.delegated_search_lab
python -m pytest tests/test_delegated_model.py tests/test_delegated_search_lab.py
```

Run as a script, the demo sets its numerical libraries to one thread. Importing
it fits nothing. `--write-fixtures` rebuilds the three JSON reports listed above.

The demo consults the August golden set and records that consultation in a
durable journal, in `golden-sets/` under the working directory or in the folder
`BOOK_GOLDEN_DIRECTORY` names. A new journal replays the two reads that chapters
5 and 6 actually made, from their stored outputs, and then appends entry 3 for
the selection record, which is the book's record (`frozen-01`). An existing
journal resumes at its next entry, so a second run in the same folder appends a
fourth read; use a new folder each time to reproduce the book. September is
never consulted.

## What the output shows

**The commission** (`commission-01`) fixes, before any fit, the seven allowed
inputs, the label policy, the measure (raw validation log loss), a budget of 20
trials, seed 7, the search space and the acceptance rule. Commissions and
selection records are typed, with canonical fingerprints; the seal made before
the consultation also binds the fitted validation predictions.

**The search** (`search-01`) tries twenty distinct logistic and boosted-tree
configurations on chapter 5's history. Raw validation log loss ranks the trials
(the winner's 0.1666 against the baseline's 0.1683, a lead smaller than its
standard error); every model then receives the same validation-fitted
recalibration map before costs are compared at threshold 0.080.

**The agent's delivery** comes from the constructed pipeline, a fixture rather
than a live agent. It builds a group input from training and validation labels,
each row's own label included. Its clean copy fits that statistic on training
only and leaves each training label out of both the group rate and its smoothing
prior.

**The inspection** (`inspect-01`), the chapter's only printed listing, runs on
its own and reads no frozen case. It applies four checks to each delivery:
allowed inputs, where each fitted statistic came from, reproduced settings, seed
and loss, and the trial budget. The agent's delivery fails the first two; its
reported loss of 0.1506 becomes 0.1754 on an honest refit.

**The decision** (`frozen-02`, `record-01`, `record-full-01`). The acceptance
rule uses paired cost intervals: failures are sent back, and an improvement that
is not proven keeps the baseline. On August, the search winner's lower
validation loss comes with a higher cost per request (0.9016 against the
baseline's 0.8149), and `truth-01` shows the same on a fresh draw from the
generator, where true expected cost is known. The baseline, boosted trees with
the map, is kept; the agent's delivery is sent back.

`truth-01` holds the scatter data for the chapter's figure, `optimism-01` the
normal-maximum curve (the expected best of many equal candidates), and `keys-01`
the exercise keys.

**The search lab** gives grid, random and Gaussian-process search equal budgets
of full trials, with expected improvement over unseen configurations as the
acquisition rule (`lab-01`, `lab-trace-01`, `ei-01`). It equalizes the number of
trials, not compute. It also shows time-safe out-of-fold stacking (`stack-01`)
and input selection ranked on validation against selection ranked inside the
training window (`select-01`). Nothing in the
lab enters the selection record or consults a frozen month.

## Exercises

Exercise 5's `delegated_model_delivery_05.json` is a search delivery to inspect;
its trial log holds sixty trials under a budget of twenty. Its key is in
`keys-01`.

The exercises also include a practice run on July, the reference solution with
forty configurations, which is not printed in the book:

```sh
BOOK_GOLDEN_DIRECTORY="$(mktemp -d)" python -m demos.delegated_model_demo --exercise-july
```

July's result is practice evidence, not a new evidence stream. Its output is
deterministic; repeat it in new journal folders to compare.

## What it leaves out

The comparison is offline evidence, and it grants the refund agent no additional
autonomy.
